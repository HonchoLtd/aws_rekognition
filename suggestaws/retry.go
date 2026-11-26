package suggestaws

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"net"
	"time"

	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

var (
	// ErrRetryable means AWS kept failing with a transient error
	// (throttling, 5xx, timeout, connection error) after every transport
	// retry. The caller may try again later, e.g. at the next cycle.
	ErrRetryable = errors.New("suggestaws: transient AWS error, retries exhausted")
	// ErrAuth means the credentials are missing, wrong or not allowed to
	// call the model or guardrail. Retrying will not help.
	ErrAuth = errors.New("suggestaws: AWS rejected the credentials")
	// ErrGuardrailNotConfigured is returned by Screen when no guardrail ID
	// is configured. The caller should turn suggestions off (fail closed).
	ErrGuardrailNotConfigured = errors.New("suggestaws: guardrail not configured")
	// ErrInvalidRequest means the caller passed something the package
	// refuses to send.
	ErrInvalidRequest = errors.New("suggestaws: invalid request")
	// ErrUnexpectedResponse means AWS answered but not in the expected
	// shape (for example Nova Lite returned no submit_terms tool call).
	ErrUnexpectedResponse = errors.New("suggestaws: unexpected AWS response")
)

// Error codes that are worth retrying, as returned by smithy.APIError.
var retryableCodes = map[string]bool{
	"ThrottlingException":         true,
	"ServiceUnavailableException": true,
	"ModelNotReadyException":      true,
	"InternalServerException":     true,
	"ModelTimeoutException":       true,
	"TooManyRequestsException":    true,
}

// Error codes that mean the credentials are the problem.
var authCodes = map[string]bool{
	"AccessDeniedException":               true,
	"UnrecognizedClientException":         true,
	"InvalidSignatureException":           true,
	"ExpiredTokenException":               true,
	"MissingAuthenticationTokenException": true,
}

type errClass int

const (
	classFatal errClass = iota
	classRetryable
	classAuth
)

// classify sorts an AWS error into retry / auth / everything else. The
// caller's own context ending is never retried.
func classify(ctx context.Context, err error) errClass {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return classFatal
	}
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		switch code := apiErr.ErrorCode(); {
		case authCodes[code]:
			return classAuth
		case retryableCodes[code]:
			return classRetryable
		}
	}
	var respErr *smithyhttp.ResponseError
	if errors.As(err, &respErr) && respErr.HTTPStatusCode() >= 500 {
		return classRetryable
	}
	// The request never got an HTTP response: connection refused/reset,
	// DNS failure, TLS handshake or response-header timeout.
	var sendErr *smithyhttp.RequestSendError
	if errors.As(err, &sendErr) {
		return classRetryable
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return classRetryable
	}
	return classFatal
}

// retryPolicy is A's own transport retry; the SDK retryer is turned off
// (see NewAWSConfig) so attempts are not multiplied.
type retryPolicy struct {
	maxAttempts int
	baseDelay   time.Duration
	maxDelay    time.Duration
	// sleep waits d or until ctx ends. Replaced in tests.
	sleep func(ctx context.Context, d time.Duration) error
}

// defaultRetryPolicy: exponential backoff with jitter, start 1 s, cap
// 30 s, at most 6 attempts (section 3.5).
var defaultRetryPolicy = retryPolicy{
	maxAttempts: 6,
	baseDelay:   time.Second,
	maxDelay:    30 * time.Second,
	sleep:       sleepCtx,
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// backoff is the upper bound of the wait before retry number `retry`
// (1-based): base, 2×base, 4×base … capped at maxDelay.
func (p retryPolicy) backoff(retry int) time.Duration {
	d := p.baseDelay
	for i := 1; i < retry && d < p.maxDelay; i++ {
		d *= 2
	}
	if d > p.maxDelay {
		d = p.maxDelay
	}
	return d
}

// jitter returns a random duration in [d/2, d], so concurrent callers do
// not retry in lockstep.
func jitter(d time.Duration) time.Duration {
	half := d / 2
	if half <= 0 {
		return d
	}
	return half + rand.N(half+1)
}

// withRetry runs fn until it succeeds, fails with a non-transient error,
// or has been tried maxAttempts times. Errors come back classified:
// wrapping ErrRetryable, wrapping ErrAuth, or unchanged.
func withRetry(ctx context.Context, p retryPolicy, op string, fn func() error) error {
	for attempt := 1; ; attempt++ {
		err := fn()
		if err == nil {
			if attempt > 1 {
				getLogger().Info("aws call recovered after retry", "op", op, "attempt", attempt)
			}
			return nil
		}
		switch classify(ctx, err) {
		case classAuth:
			return fmt.Errorf("%w: %w", ErrAuth, err)
		case classFatal:
			return err
		}
		if attempt >= p.maxAttempts {
			getLogger().Warn("aws call gave up retrying", "op", op, "attempts", attempt, "error", err)
			return fmt.Errorf("%w (%d attempts): %w", ErrRetryable, attempt, err)
		}
		wait := jitter(p.backoff(attempt))
		getLogger().Warn("aws call hit a transient error, retrying", "op", op, "attempt", attempt, "max_attempts", p.maxAttempts, "retry_in_ms", wait.Milliseconds(), "error", err)
		if serr := p.sleep(ctx, wait); serr != nil {
			return serr
		}
	}
}
