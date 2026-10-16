package modelembedding

import (
	"context"
	"fmt"
	"sync/atomic"
)

// CallStats counts how many times each Embedder method was called. Read it
// via StatsEmbedder.Snapshot(). Retries inside a call are not counted
// separately.
type CallStats struct {
	EmbedImage           int64 `json:"embed_image"`
	EmbedImageWithBucket int64 `json:"embed_image_with_bucket"`
	EmbedText            int64 `json:"embed_text"`
}

// Total is the sum of every counted call, i.e. the number of Bedrock
// InvokeModel requests made (excluding retries).
func (c CallStats) Total() int64 {
	return c.EmbedImage + c.EmbedImageWithBucket + c.EmbedText
}

func (c CallStats) String() string {
	return fmt.Sprintf(
		"EmbedImage=%d EmbedImageWithBucket=%d EmbedText=%d (total=%d)",
		c.EmbedImage, c.EmbedImageWithBucket, c.EmbedText, c.Total(),
	)
}

// StatsEmbedder wraps an Embedder and counts calls made through it without
// changing behavior, e.g.:
//
//	tracked := modelembedding.NewStatsEmbedder(modelembedding.NewNovaEmbedder(client))
//	// ... run your pipeline against `tracked` ...
//	fmt.Println(tracked.Snapshot())
type StatsEmbedder struct {
	embedder             Embedder
	embedImage           atomic.Int64
	embedImageWithBucket atomic.Int64
	embedText            atomic.Int64
}

func NewStatsEmbedder(embedder Embedder) *StatsEmbedder {
	return &StatsEmbedder{embedder: embedder}
}

// Snapshot returns the call counts observed so far. Safe to call while
// other calls are in flight.
func (s *StatsEmbedder) Snapshot() CallStats {
	return CallStats{
		EmbedImage:           s.embedImage.Load(),
		EmbedImageWithBucket: s.embedImageWithBucket.Load(),
		EmbedText:            s.embedText.Load(),
	}
}

func (s *StatsEmbedder) EmbedImage(ctx context.Context, image []byte, opts ...EmbedOption) (Embedding, error) {
	s.embedImage.Add(1)
	return s.embedder.EmbedImage(ctx, image, opts...)
}

func (s *StatsEmbedder) EmbedImageWithBucket(ctx context.Context, s3Bucket string, s3Key string, opts ...EmbedOption) (Embedding, error) {
	s.embedImageWithBucket.Add(1)
	return s.embedder.EmbedImageWithBucket(ctx, s3Bucket, s3Key, opts...)
}

func (s *StatsEmbedder) EmbedText(ctx context.Context, text string, opts ...EmbedOption) (Embedding, error) {
	s.embedText.Add(1)
	return s.embedder.EmbedText(ctx, text, opts...)
}

var _ Embedder = (*StatsEmbedder)(nil)
