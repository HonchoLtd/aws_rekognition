package modelembedding

import (
	"context"
	"errors"
	"math/rand/v2"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// retryPolicy controls withRetry's exponential backoff. It sits on top of
// the AWS SDK client's own retryer, which already makes a few quick
// attempts; this layer adds longer waits so bursts of concurrent calls
// (see EmbedImages) can ride out Bedrock throttling instead of failing.
type retryPolicy struct {
	maxAttempts int
	baseDelay   time.Duration
}

// defaultRetryPolicy: up to 4 attempts with roughly 1s, 2s, 4s between
// them (each randomized to 50-100% of that, so concurrent workers don't
// retry in lockstep).
var defaultRetryPolicy = retryPolicy{
	maxAttempts: 4,
	baseDelay:   time.Second,
}

// withRetry calls fn, retrying with jittered exponential backoff while it
// keeps failing with a transient error (per isTransient), up to
// policy.maxAttempts total attempts. It returns immediately on success, on
// a non-transient error, or if ctx is canceled while waiting.
func withRetry(ctx context.Context, policy retryPolicy, opName string, fn func() error) error {
	var lastErr error
	delay := policy.baseDelay

	for attempt := 1; attempt <= policy.maxAttempts; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			if attempt > 1 {
				getLogger().Info("bedrock api call recovered after retry", "op", opName, "succeeded_on_attempt", attempt, "max_attempts", policy.maxAttempts)
			}
			return nil
		}
		if !isTransient(lastErr) {
			return lastErr
		}
		if attempt == policy.maxAttempts {
			getLogger().Warn("bedrock api call gave up retrying", "op", opName, "attempts", attempt, "max_attempts", policy.maxAttempts, "error", lastErr)
			return lastErr
		}

		wait := jitter(delay)
		getLogger().Warn("bedrock api call hit a transient error, retrying", "op", opName, "attempt", attempt, "max_attempts", policy.maxAttempts, "retry_in_ms", wait.Milliseconds(), "error", lastErr)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(wait):
		}
		delay *= 2
	}

	return lastErr
}

// jitter returns a random duration in [d/2, d].
func jitter(d time.Duration) time.Duration {
	half := d / 2
	if half <= 0 {
		return d
	}
	return half + rand.N(half+1)
}

// isTransient reports whether err is a Bedrock error worth retrying:
// throttling or a temporary service-side failure, as opposed to bad input
// or missing permissions.
func isTransient(err error) bool {
	var (
		throttling   *types.ThrottlingException
		unavailable  *types.ServiceUnavailableException
		internal     *types.InternalServerException
		notReady     *types.ModelNotReadyException
		modelTimeout *types.ModelTimeoutException
	)
	return errors.As(err, &throttling) ||
		errors.As(err, &unavailable) ||
		errors.As(err, &internal) ||
		errors.As(err, &notReady) ||
		errors.As(err, &modelTimeout)
}
