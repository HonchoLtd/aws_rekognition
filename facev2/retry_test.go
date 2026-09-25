package facev2

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

// fastRetryPolicy keeps these tests quick — same shape as
// defaultRetryPolicy, just without the multi-second real-world delays.
var fastRetryPolicy = retryPolicy{maxAttempts: 4, baseDelay: time.Millisecond}

func TestWithRetry_SucceedsImmediately(t *testing.T) {
	calls := 0
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		return nil
	})
	if err != nil {
		t.Fatalf("withRetry() error = %v, want nil", err)
	}
	if calls != 1 {
		t.Errorf("fn called %d times, want 1 (no retry needed)", calls)
	}
}

func TestWithRetry_RetriesTransientThenSucceeds(t *testing.T) {
	calls := 0
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		if calls < 3 {
			return &types.ResourceNotFoundException{Message: aws32str("collection not found yet")}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("withRetry() error = %v, want nil (should have succeeded on attempt 3)", err)
	}
	if calls != 3 {
		t.Errorf("fn called %d times, want 3", calls)
	}
}

func TestWithRetry_GivesUpAfterMaxAttempts(t *testing.T) {
	calls := 0
	wantErr := &types.ResourceNotFoundException{Message: aws32str("still not found")}
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		return wantErr
	})
	if err != error(wantErr) {
		t.Errorf("withRetry() error = %v, want the same *ResourceNotFoundException instance", err)
	}
	if calls != fastRetryPolicy.maxAttempts {
		t.Errorf("fn called %d times, want %d (maxAttempts, then give up)", calls, fastRetryPolicy.maxAttempts)
	}
}

func TestWithRetry_DoesNotRetryNonTransientError(t *testing.T) {
	calls := 0
	nonTransient := &types.InvalidParameterException{Message: aws32str("bad input")}
	err := withRetry(context.Background(), fastRetryPolicy, "op", func() error {
		calls++
		return nonTransient
	})
	if err != nonTransient {
		t.Errorf("withRetry() error = %v, want %v", err, nonTransient)
	}
	if calls != 1 {
		t.Errorf("fn called %d times, want 1 (non-transient errors shouldn't retry)", calls)
	}
}

func TestWithRetry_StopsOnContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	slowPolicy := retryPolicy{maxAttempts: 5, baseDelay: 50 * time.Millisecond}

	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	err := withRetry(ctx, slowPolicy, "op", func() error {
		calls++
		return &types.ThrottlingException{Message: aws32str("slow down")}
	})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("withRetry() error = %v, want context.Canceled", err)
	}
	if calls >= slowPolicy.maxAttempts {
		t.Errorf("fn called %d times, want fewer than maxAttempts (%d) since context was canceled first", calls, slowPolicy.maxAttempts)
	}
}

func TestIsTransient(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"ResourceNotFoundException", &types.ResourceNotFoundException{}, true},
		{"ThrottlingException", &types.ThrottlingException{}, true},
		{"ProvisionedThroughputExceededException", &types.ProvisionedThroughputExceededException{}, true},
		{"InvalidParameterException", &types.InvalidParameterException{}, false},
		{"wrapped ResourceNotFoundException", fmt.Errorf("call failed: %w", &types.ResourceNotFoundException{}), true},
		{"plain error", errors.New("boom"), false},
		{"nil", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isTransient(tc.err); got != tc.want {
				t.Errorf("isTransient(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func aws32str(s string) *string { return &s }
