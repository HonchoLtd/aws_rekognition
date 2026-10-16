package modelembedding

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// fakeEmbedder is an Embedder whose behavior per input is decided by fail
// (keyed by the image bytes as a string, or the S3 key). It tracks peak
// concurrency.
type fakeEmbedder struct {
	delay    time.Duration
	fail     map[string]error
	inFlight atomic.Int32
	peak     atomic.Int32
	calls    atomic.Int32
}

func (f *fakeEmbedder) embed(ctx context.Context, key string) (Embedding, error) {
	f.calls.Add(1)
	n := f.inFlight.Add(1)
	defer f.inFlight.Add(-1)
	for {
		p := f.peak.Load()
		if n <= p || f.peak.CompareAndSwap(p, n) {
			break
		}
	}
	select {
	case <-ctx.Done():
		return Embedding{}, ctx.Err()
	case <-time.After(f.delay):
	}
	if err := f.fail[key]; err != nil {
		return Embedding{}, err
	}
	return Embedding{Vector: []float32{float32(len(key))}, Dimension: 1}, nil
}

func (f *fakeEmbedder) EmbedImage(ctx context.Context, image []byte, _ ...EmbedOption) (Embedding, error) {
	return f.embed(ctx, string(image))
}

func (f *fakeEmbedder) EmbedImageWithBucket(ctx context.Context, _ string, s3Key string, _ ...EmbedOption) (Embedding, error) {
	return f.embed(ctx, s3Key)
}

func (f *fakeEmbedder) EmbedText(ctx context.Context, text string, _ ...EmbedOption) (Embedding, error) {
	return f.embed(ctx, text)
}

func makeItems(n int) []ImageInput {
	items := make([]ImageInput, n)
	for i := range items {
		items[i] = ImageInput{ID: fmt.Sprintf("img-%02d", i), Image: []byte(fmt.Sprintf("img-%02d", i))}
	}
	return items
}

func TestEmbedImages_ReturnsResultsInInputOrderWithBoundedConcurrency(t *testing.T) {
	embedder := &fakeEmbedder{delay: 5 * time.Millisecond}
	items := makeItems(20)

	var mu sync.Mutex
	callbackInFlight := 0
	seen := map[string]bool{}
	results := EmbedImages(context.Background(), embedder, items,
		WithConcurrency(3),
		WithOnResult(func(r ImageResult) {
			mu.Lock()
			callbackInFlight++
			if callbackInFlight > 1 {
				t.Error("OnResult called concurrently")
			}
			seen[r.ID] = true
			callbackInFlight--
			mu.Unlock()
		}),
	)

	if len(results) != len(items) {
		t.Fatalf("got %d results, want %d", len(results), len(items))
	}
	for i, r := range results {
		if r.Err != nil || r.Index != i || r.ID != items[i].ID {
			t.Errorf("results[%d] = {ID:%q Index:%d Err:%v}, want ID %q", i, r.ID, r.Index, r.Err, items[i].ID)
		}
	}
	if len(seen) != len(items) {
		t.Errorf("OnResult saw %d distinct items, want %d", len(seen), len(items))
	}
	if peak := embedder.peak.Load(); peak > 3 || peak < 2 {
		t.Errorf("peak concurrency = %d, want between 2 and 3", peak)
	}
}

func TestEmbedImages_PerItemErrorsDoNotStopOthers(t *testing.T) {
	badImage := &types.ValidationException{Message: aws.String("bad image")}
	embedder := &fakeEmbedder{fail: map[string]error{"img-02": badImage}}
	items := makeItems(5)
	items = append(items, ImageInput{ID: "s3-item", S3Bucket: "b", S3Key: "k.jpg"}, ImageInput{ID: "empty"})

	results := EmbedImages(context.Background(), embedder, items)

	for i, r := range results {
		switch r.ID {
		case "img-02":
			if !errors.Is(r.Err, badImage) {
				t.Errorf("img-02 error = %v, want the ValidationException", r.Err)
			}
		case "empty":
			if !errors.Is(r.Err, ErrEmptyInput) {
				t.Errorf("empty item error = %v, want ErrEmptyInput", r.Err)
			}
		default:
			if r.Err != nil {
				t.Errorf("results[%d] (%s) error = %v, want nil", i, r.ID, r.Err)
			}
		}
	}
	if calls := embedder.calls.Load(); calls != 6 {
		t.Errorf("embedder called %d times, want 6 (the empty item never reaches it)", calls)
	}
}

func TestEmbedImages_FatalErrorAbortsRemainingItems(t *testing.T) {
	denied := &types.AccessDeniedException{Message: aws.String("not allowed")}
	embedder := &fakeEmbedder{delay: time.Millisecond, fail: map[string]error{"img-00": denied}}
	items := makeItems(50)

	results := EmbedImages(context.Background(), embedder, items, WithConcurrency(1))

	var gotDenied *types.AccessDeniedException
	if !errors.As(results[0].Err, &gotDenied) || errors.Is(results[0].Err, ErrAborted) {
		t.Errorf("results[0] error = %v, want the AccessDeniedException itself", results[0].Err)
	}
	for _, r := range results[1:] {
		if !errors.Is(r.Err, ErrAborted) || !errors.As(r.Err, &gotDenied) {
			t.Fatalf("%s error = %v, want ErrAborted wrapping AccessDenied", r.ID, r.Err)
		}
	}
	if calls := embedder.calls.Load(); calls != 1 {
		t.Errorf("embedder called %d times, want 1 (everything after the fatal error is skipped)", calls)
	}
}

func TestEmbedImages_ContextCancellation(t *testing.T) {
	embedder := &fakeEmbedder{delay: time.Hour}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	results := EmbedImages(ctx, embedder, makeItems(10), WithConcurrency(2))
	for _, r := range results {
		if !errors.Is(r.Err, context.DeadlineExceeded) {
			t.Errorf("%s error = %v, want context.DeadlineExceeded", r.ID, r.Err)
		}
	}
}

func TestEmbedImages_Empty(t *testing.T) {
	if results := EmbedImages(context.Background(), &fakeEmbedder{}, nil); len(results) != 0 {
		t.Errorf("got %d results, want 0", len(results))
	}
}

func TestEmbedImages_PassesEmbedOptions(t *testing.T) {
	client := &fakeClient{}
	results := EmbedImages(context.Background(), newTestEmbedder(client),
		[]ImageInput{{ID: "a", S3Bucket: "b", S3Key: "a.png"}},
		WithEmbedOptions(WithDimension(384)),
	)
	if results[0].Err != nil || results[0].Embedding.Dimension != 384 {
		t.Errorf("result = %d dims, %v; want 384 dims", results[0].Embedding.Dimension, results[0].Err)
	}
}

func TestStatsEmbedder(t *testing.T) {
	stats := NewStatsEmbedder(&fakeEmbedder{})
	ctx := context.Background()
	_, _ = stats.EmbedImage(ctx, []byte("a"))
	_, _ = stats.EmbedImage(ctx, []byte("b"))
	_, _ = stats.EmbedImageWithBucket(ctx, "b", "k.jpg")
	_, _ = stats.EmbedText(ctx, "q")

	got := stats.Snapshot()
	want := CallStats{EmbedImage: 2, EmbedImageWithBucket: 1, EmbedText: 1}
	if got != want || got.Total() != 4 {
		t.Errorf("Snapshot() = %+v (total %d), want %+v (total 4)", got, got.Total(), want)
	}
	if got.String() != "EmbedImage=2 EmbedImageWithBucket=1 EmbedText=1 (total=4)" {
		t.Errorf("String() = %q", got.String())
	}
}
