package modelembedding

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// fastRetryPolicy keeps these tests quick: same shape as
// defaultRetryPolicy, without the multi-second real-world delays.
var fastRetryPolicy = retryPolicy{maxAttempts: 4, baseDelay: time.Millisecond}

func TestWithRetry_SucceedsImmediately(t *testing.T) {
	calls := 0
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		return nil
	})
	if err != nil || calls != 1 {
		t.Errorf("withRetry() = %v after %d calls, want nil after 1", err, calls)
	}
}

func TestWithRetry_RetriesTransientThenSucceeds(t *testing.T) {
	calls := 0
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		if calls < 3 {
			return &types.ThrottlingException{Message: aws.String("slow down")}
		}
		return nil
	})
	if err != nil || calls != 3 {
		t.Errorf("withRetry() = %v after %d calls, want nil after 3", err, calls)
	}
}

func TestWithRetry_GivesUpAfterMaxAttempts(t *testing.T) {
	calls := 0
	wantErr := &types.ThrottlingException{Message: aws.String("still throttled")}
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		return wantErr
	})
	if err != error(wantErr) {
		t.Errorf("withRetry() error = %v, want the same *ThrottlingException", err)
	}
	if calls != fastRetryPolicy.maxAttempts {
		t.Errorf("fn called %d times, want %d", calls, fastRetryPolicy.maxAttempts)
	}
}

func TestWithRetry_DoesNotRetryNonTransientError(t *testing.T) {
	calls := 0
	wantErr := &types.AccessDeniedException{Message: aws.String("no")}
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		return wantErr
	})
	if err != error(wantErr) || calls != 1 {
		t.Errorf("withRetry() = %v after %d calls, want AccessDenied after 1", err, calls)
	}
}

func TestWithRetry_StopsOnContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	err := withRetry(ctx, retryPolicy{maxAttempts: 4, baseDelay: time.Hour}, "op", func() error {
		calls++
		cancel()
		return &types.ThrottlingException{Message: aws.String("slow down")}
	})
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Errorf("withRetry() = %v after %d calls, want context.Canceled after 1", err, calls)
	}
}

func TestIsTransient(t *testing.T) {
	tests := []struct {
		err  error
		want bool
	}{
		{&types.ThrottlingException{}, true},
		{&types.ServiceUnavailableException{}, true},
		{&types.InternalServerException{}, true},
		{&types.ModelNotReadyException{}, true},
		{&types.ModelTimeoutException{}, true},
		{fmt.Errorf("wrapped: %w", &types.ThrottlingException{}), true},
		{&types.ValidationException{}, false},
		{&types.AccessDeniedException{}, false},
		{&types.ResourceNotFoundException{}, false},
		{errors.New("plain"), false},
	}
	for _, tt := range tests {
		if got := isTransient(tt.err); got != tt.want {
			t.Errorf("isTransient(%T) = %v, want %v", tt.err, got, tt.want)
		}
	}
}

func TestJitter(t *testing.T) {
	for i := 0; i < 100; i++ {
		if got := jitter(time.Second); got < 500*time.Millisecond || got > time.Second {
			t.Fatalf("jitter(1s) = %v, want within [500ms, 1s]", got)
		}
	}
}
