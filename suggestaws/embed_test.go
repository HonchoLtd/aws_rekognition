package suggestaws

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

func vectorBody(dim int, first float32) []byte {
	v := make([]float32, dim)
	v[0] = first
	return mustJSON(map[string]any{"embeddings": []any{map[string]any{"embeddingType": "TEXT", "embedding": v}}})
}

func TestEmbedText_RequestBodyAndRawVector(t *testing.T) {
	rt := &fakeRuntime{invokeModel: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		return &bedrockruntime.InvokeModelOutput{Body: vectorBody(1024, 3)}, nil
	}}
	c := newTestClient(rt, Config{})
	vec, err := c.EmbedText(context.Background(), "keynote presentation")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != 1024 || vec[0] != 3 {
		t.Errorf("vector len %d first %v, want raw 1024-d vector starting with 3 (not normalised)", len(vec), vec[0])
	}

	in := rt.invokeInputs[0]
	if aws.ToString(in.ModelId) != DefaultEmbedModelID {
		t.Errorf("ModelId = %q", aws.ToString(in.ModelId))
	}
	var got, want any
	_ = json.Unmarshal(in.Body, &got)
	_ = json.Unmarshal([]byte(`{"schemaVersion":"nova-multimodal-embed-v1","taskType":"SINGLE_EMBEDDING",
		"singleEmbeddingParams":{"embeddingDimension":1024,"embeddingPurpose":"IMAGE_RETRIEVAL",
		"text":{"truncationMode":"END","value":"keynote presentation"}}}`), &want)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("body = %s", in.Body)
	}
}

func TestEmbedText_WrongDimensionAndBlank(t *testing.T) {
	rt := &fakeRuntime{invokeModel: func(*bedrockruntime.InvokeModelInput) (*bedrockruntime.InvokeModelOutput, error) {
		return &bedrockruntime.InvokeModelOutput{Body: vectorBody(256, 1)}, nil
	}}
	c := newTestClient(rt, Config{})
	if _, err := c.EmbedText(context.Background(), "drum circle"); !errors.Is(err, ErrUnexpectedResponse) {
		t.Errorf("wrong dimension: err = %v, want ErrUnexpectedResponse", err)
	}
	if _, err := c.EmbedText(context.Background(), "   "); !errors.Is(err, ErrInvalidRequest) {
		t.Errorf("blank: err = %v, want ErrInvalidRequest", err)
	}
}

// slowEmbedder tracks how many EmbedText calls run at once.
type slowEmbedder struct {
	Client
	inFlight, peak atomic.Int32
}

func (s *slowEmbedder) EmbedText(ctx context.Context, text string) ([]float32, error) {
	n := s.inFlight.Add(1)
	defer s.inFlight.Add(-1)
	for {
		p := s.peak.Load()
		if n <= p || s.peak.CompareAndSwap(p, n) {
			break
		}
	}
	time.Sleep(5 * time.Millisecond)
	if text == "boom" {
		return nil, errors.New("boom")
	}
	return []float32{float32(len(text))}, nil
}

func TestEmbedTexts_BoundedConcurrencyAndOrder(t *testing.T) {
	s := &slowEmbedder{}
	texts := []string{"a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg", "hhhhhhhh", "iiiiiiiii"}
	vecs, err := EmbedTexts(context.Background(), s, texts, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if int(v[0]) != len(texts[i]) {
			t.Errorf("vector %d is for the wrong text", i)
		}
	}
	if p := s.peak.Load(); p > DefaultEmbedConcurrency || p < 2 {
		t.Errorf("peak concurrency = %d, want 2..%d", p, DefaultEmbedConcurrency)
	}
	if _, err := EmbedTexts(context.Background(), s, []string{"ok", "boom", "ok2"}, 2); err == nil {
		t.Error("want the error from the failing text")
	}
}
