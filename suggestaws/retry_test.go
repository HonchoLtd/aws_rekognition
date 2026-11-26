package suggestaws

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

func TestErrors_ThrottlingRetriedThenErrRetryable(t *testing.T) {
	rec := &sleepRecorder{}
	rt := &fakeRuntime{invokeModel: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		return nil, &types.ThrottlingException{Message: aws.String("slow down")}
	}}
	c := newTestClient(rt, Config{})
	c.retry = testPolicy(rec)

	_, err := c.EmbedText(context.Background(), "drum circle")
	if !errors.Is(err, ErrRetryable) {
		t.Fatalf("err = %v, want ErrRetryable", err)
	}
	var te *types.ThrottlingException
	if !errors.As(err, &te) {
		t.Error("ErrRetryable should still wrap the last AWS error")
	}
	if len(rt.invokeInputs) != 6 {
		t.Errorf("attempts = %d, want 6", len(rt.invokeInputs))
	}
	if len(rec.sleeps) != 5 {
		t.Fatalf("sleeps = %d, want 5", len(rec.sleeps))
	}
	// Each wait is jittered into [bound/2, bound] of 1s, 2s, 4s, 8s, 16s.
	for i, d := range rec.sleeps {
		bound := time.Second << i
		if d < bound/2 || d > bound {
			t.Errorf("sleep %d = %v, want within [%v, %v]", i, d, bound/2, bound)
		}
	}
}

func TestErrors_AccessDeniedIsErrAuthAtOnce(t *testing.T) {
	for _, apiErr := range []error{
		&types.AccessDeniedException{Message: aws.String("no")},
		&smithy.GenericAPIError{Code: "UnrecognizedClientException", Message: "bad key"},
		&smithy.GenericAPIError{Code: "InvalidSignatureException", Message: "bad sig"},
	} {
		rt := &fakeRuntime{converse: func(*bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) { return nil, apiErr }}
		c := newTestClient(rt, Config{})
		_, err := c.ProposeTerms(context.Background(), ProposeRequest{Groups: []Group{{Number: 1, Size: 5}}})
		if !errors.Is(err, ErrAuth) {
			t.Errorf("%T: err = %v, want ErrAuth", apiErr, err)
		}
		if len(rt.converseInputs) != 1 {
			t.Errorf("%T: attempts = %d, want 1", apiErr, len(rt.converseInputs))
		}
	}
}

func TestErrors_ValidationNotRetried(t *testing.T) {
	rt := &fakeRuntime{converse: func(*bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
		return nil, &types.ValidationException{Message: aws.String("too many images")}
	}}
	c := newTestClient(rt, Config{})
	_, err := c.ProposeTerms(context.Background(), ProposeRequest{Groups: []Group{{Number: 1, Size: 5}}})
	if err == nil || errors.Is(err, ErrRetryable) || errors.Is(err, ErrAuth) {
		t.Errorf("err = %v, want the plain validation error", err)
	}
	if len(rt.converseInputs) != 1 {
		t.Errorf("attempts = %d, want 1", len(rt.converseInputs))
	}
}

func TestErrors_5xxAndConnectionErrorsRetried(t *testing.T) {
	transient := []error{
		&smithyhttp.ResponseError{Response: &smithyhttp.Response{Response: &http.Response{StatusCode: 502}}, Err: errors.New("bad gateway")},
		&smithyhttp.RequestSendError{Err: errors.New("connection reset by peer")},
		&types.ModelNotReadyException{Message: aws.String("warming")},
		&types.ServiceUnavailableException{Message: aws.String("busy")},
	}
	for _, e := range transient {
		calls := 0
		rt := &fakeRuntime{invokeModel: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
			calls++
			if calls < 3 {
				return nil, e
			}
			return &bedrockruntime.InvokeModelOutput{Body: vectorBody(1024, 1)}, nil
		}}
		if _, err := newTestClient(rt, Config{}).EmbedText(context.Background(), "drum circle"); err != nil || calls != 3 {
			t.Errorf("%T: err = %v after %d calls, want success on attempt 3", e, err, calls)
		}
	}
}

func TestErrors_CallerDeadlineNotRetried(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rt := &fakeRuntime{invokeModel: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		return nil, &smithyhttp.RequestSendError{Err: context.Canceled}
	}}
	_, err := newTestClient(rt, Config{}).EmbedText(ctx, "drum circle")
	if errors.Is(err, ErrRetryable) || len(rt.invokeInputs) != 1 {
		t.Errorf("err = %v after %d calls; a cancelled caller must not be retried", err, len(rt.invokeInputs))
	}
}

func TestBackoffCapsAt30s(t *testing.T) {
	p := defaultRetryPolicy
	want := []time.Duration{1, 2, 4, 8, 16, 30, 30}
	for i, w := range want {
		if got := p.backoff(i + 1); got != w*time.Second {
			t.Errorf("backoff(%d) = %v, want %v", i+1, got, w*time.Second)
		}
	}
}
