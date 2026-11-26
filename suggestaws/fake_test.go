package suggestaws

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// fakeRuntime is a RuntimeAPI whose answers come from the func fields. It
// records every request and is safe for the concurrent calls Screen makes.
type fakeRuntime struct {
	mu sync.Mutex

	converse       func(*bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error)
	applyGuardrail func(*bedrockruntime.ApplyGuardrailInput) (*bedrockruntime.ApplyGuardrailOutput, error)
	invokeModel    func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error)

	converseInputs  []*bedrockruntime.ConverseInput
	guardrailInputs []*bedrockruntime.ApplyGuardrailInput
	invokeInputs    []*bedrockruntime.InvokeModelInput
}

func (f *fakeRuntime) Converse(_ context.Context, in *bedrockruntime.ConverseInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error) {
	f.mu.Lock()
	f.converseInputs = append(f.converseInputs, in)
	f.mu.Unlock()
	return f.converse(in)
}

func (f *fakeRuntime) ApplyGuardrail(_ context.Context, in *bedrockruntime.ApplyGuardrailInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.ApplyGuardrailOutput, error) {
	f.mu.Lock()
	f.guardrailInputs = append(f.guardrailInputs, in)
	f.mu.Unlock()
	return f.applyGuardrail(in)
}

func (f *fakeRuntime) InvokeModel(_ context.Context, in *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	f.mu.Lock()
	f.invokeInputs = append(f.invokeInputs, in)
	f.mu.Unlock()
	return f.invokeModel(in)
}

// noSleep makes retries instant; sleeps records the requested waits.
type sleepRecorder struct {
	mu     sync.Mutex
	sleeps []time.Duration
}

func (s *sleepRecorder) sleep(ctx context.Context, d time.Duration) error {
	s.mu.Lock()
	s.sleeps = append(s.sleeps, d)
	s.mu.Unlock()
	return ctx.Err()
}

func testPolicy(rec *sleepRecorder) retryPolicy {
	p := defaultRetryPolicy
	p.sleep = rec.sleep
	return p
}

func newTestClient(rt RuntimeAPI, cfg Config) *AWSClient {
	c, err := New(rt, cfg)
	if err != nil {
		panic(err)
	}
	c.retry = testPolicy(&sleepRecorder{})
	return c
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
