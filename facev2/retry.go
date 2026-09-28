package facev2

import (
	"context"
	"errors"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

// retryPolicy controls withRetry's exponential backoff for calls that can
// fail with a transient, eventual-consistency-shaped error — specifically
// ResourceNotFoundException right after CreateCollection, which a real-AWS
// smoke test caught hitting SearchFaces within a couple of seconds of a
// fresh collection existing. This mirrors the fixed delays `face`'s
// IndexFace/SearchFacebyFaceId already use for the same class of issue, but
// only pays the cost when a call actually needs it, instead of always
// sleeping up front.
type retryPolicy struct {
	maxAttempts int
	baseDelay   time.Duration
}

// defaultRetryPolicy: up to 3 retries (4 attempts total), with delays of
// 500ms, 1s, 2s between them (~3.5s worst case) — in the same ballpark as
// the 3s fixed delay `face.SearchFacebyFaceId` already uses for this.
var defaultRetryPolicy = retryPolicy{
	maxAttempts: 4,
	baseDelay:   500 * time.Millisecond,
}

// withRetry calls fn, retrying with exponential backoff while it keeps
// failing with a transient error (per isTransient), up to policy.maxAttempts
// total attempts. It returns immediately on a non-transient error, on
// success, or if ctx is canceled while waiting between attempts. opName is
// only used for the retry log line.
func withRetry(ctx context.Context, policy retryPolicy, opName string, fn func() error) error {
	var lastErr error
	delay := policy.baseDelay

	for attempt := 1; attempt <= policy.maxAttempts; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			if attempt > 1 {
				getLogger().Info("rekognition api call recovered after retry", "op", opName, "succeeded_on_attempt", attempt, "max_attempts", policy.maxAttempts)
			}
			return nil
		}
		if !isTransient(lastErr) {
			return lastErr
		}
		if attempt == policy.maxAttempts {
			getLogger().Warn("rekognition api call gave up retrying", "op", opName, "attempts", attempt, "max_attempts", policy.maxAttempts, "error", lastErr)
			return lastErr
		}

		getLogger().Warn("rekognition api call hit a transient error, retrying", "op", opName, "attempt", attempt, "max_attempts", policy.maxAttempts, "retry_in_ms", delay.Milliseconds(), "error", lastErr)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
		delay *= 2
	}

	return lastErr
}

// isTransient reports whether err looks like a transient AWS Rekognition
// error worth retrying: ResourceNotFoundException (the eventual-consistency
// gap right after collection/user creation) or throttling — as opposed to a
// genuine, retry-proof failure like an invalid parameter.
func isTransient(err error) bool {
	var notFound *types.ResourceNotFoundException
	if errors.As(err, &notFound) {
		return true
	}
	var throttling *types.ThrottlingException
	if errors.As(err, &throttling) {
		return true
	}
	var provisionedThroughputExceeded *types.ProvisionedThroughputExceededException
	if errors.As(err, &provisionedThroughputExceeded) {
		return true
	}
	return false
}
