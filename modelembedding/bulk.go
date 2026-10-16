package modelembedding

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/aws/smithy-go"
)

// DefaultBulkConcurrency is how many EmbedImages calls run at once when
// WithConcurrency isn't passed. Raise it only as far as your account's
// Bedrock requests-per-minute quota for the model allows.
const DefaultBulkConcurrency = 4

// ErrAborted marks EmbedImages items that were never embedded because an
// earlier item hit an error no other item could succeed past (rejected
// credentials, missing model access). errors.Is(err, ErrAborted) is true
// for them, and errors.As still reaches the underlying AWS error.
var ErrAborted = errors.New("modelembedding: bulk embedding aborted")

// ImageInput is one image for EmbedImages: either Image bytes, or an
// S3Bucket/S3Key pair when Image is empty.
type ImageInput struct {
	// ID is the caller's identifier for this image (file name, DB id, ...),
	// echoed back on its ImageResult.
	ID       string
	Image    []byte
	S3Bucket string
	S3Key    string
}

// ImageResult is the outcome for one ImageInput. Exactly one of Embedding
// and Err is meaningful.
type ImageResult struct {
	ID string
	// Index is the position of the input in the slice given to EmbedImages.
	Index     int
	Embedding Embedding
	Err       error
}

type bulkConfig struct {
	concurrency  int
	embedOptions []EmbedOption
	onResult     func(ImageResult)
}

// BulkOption configures an EmbedImages call.
type BulkOption func(*bulkConfig)

// WithConcurrency sets how many images are embedded at once (default
// DefaultBulkConcurrency).
func WithConcurrency(n int) BulkOption {
	return func(c *bulkConfig) { c.concurrency = n }
}

// WithEmbedOptions passes options through to every EmbedImage /
// EmbedImageWithBucket call.
func WithEmbedOptions(opts ...EmbedOption) BulkOption {
	return func(c *bulkConfig) { c.embedOptions = append(c.embedOptions, opts...) }
}

// WithOnResult registers a callback that receives each result as soon as it
// completes, in completion order. Calls are serialized (never concurrent),
// so the callback can append to a batch or write to a DB without its own
// locking; a slow callback slows down result collection, not the in-flight
// Bedrock calls.
func WithOnResult(fn func(ImageResult)) BulkOption {
	return func(c *bulkConfig) { c.onResult = fn }
}

// EmbedImages embeds items concurrently through embedder and returns one
// result per item, in input order. Failures are per item: one bad image
// doesn't stop the rest. The exceptions are errors that would fail every
// remaining item too (see isFatal); after one of those, items not yet
// started get an ErrAborted error instead of calling Bedrock. If ctx is
// canceled, unfinished items get ctx's error.
//
// It works with any Embedder, including StatsEmbedder and test fakes.
func EmbedImages(ctx context.Context, embedder Embedder, items []ImageInput, opts ...BulkOption) []ImageResult {
	cfg := bulkConfig{concurrency: DefaultBulkConcurrency}
	for _, opt := range opts {
		opt(&cfg)
	}
	workers := min(max(cfg.concurrency, 1), len(items))

	results := make([]ImageResult, len(items))
	if len(items) == 0 {
		return results
	}

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	jobs := make(chan int)
	done := make(chan ImageResult)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				done <- embedOne(ctx, cancel, embedder, idx, items[idx], cfg.embedOptions)
			}
		}()
	}

	go func() {
		defer close(jobs)
		for idx := range items {
			jobs <- idx
		}
	}()

	go func() {
		wg.Wait()
		close(done)
	}()

	succeeded, failed := 0, 0
	for result := range done {
		results[result.Index] = result
		if result.Err != nil {
			failed++
		} else {
			succeeded++
		}
		if cfg.onResult != nil {
			cfg.onResult(result)
		}
	}

	getLogger().Info("bulk embedding finished", "items", len(items), "succeeded", succeeded, "failed", failed, "concurrency", workers)
	return results
}

// embedOne embeds a single item, skipping the Bedrock call entirely once
// ctx is done. A fatal error cancels ctx so the other workers stop too.
func embedOne(ctx context.Context, cancel context.CancelCauseFunc, embedder Embedder, idx int, item ImageInput, opts []EmbedOption) ImageResult {
	result := ImageResult{ID: item.ID, Index: idx}
	if ctx.Err() != nil {
		result.Err = context.Cause(ctx)
		return result
	}

	switch {
	case len(item.Image) > 0:
		result.Embedding, result.Err = embedder.EmbedImage(ctx, item.Image, opts...)
	case item.S3Bucket != "" && item.S3Key != "":
		result.Embedding, result.Err = embedder.EmbedImageWithBucket(ctx, item.S3Bucket, item.S3Key, opts...)
	default:
		result.Err = fmt.Errorf("%w: item %q has neither Image bytes nor S3Bucket/S3Key", ErrEmptyInput, item.ID)
		return result
	}

	if result.Err == nil {
		return result
	}
	if isFatal(result.Err) {
		cancel(fmt.Errorf("%w after item %q: %w", ErrAborted, item.ID, result.Err))
		getLogger().Error("bulk embedding aborted", "id", item.ID, "error", result.Err)
		return result
	}
	if ctx.Err() != nil && (errors.Is(result.Err, context.Canceled) || errors.Is(result.Err, context.DeadlineExceeded)) {
		result.Err = context.Cause(ctx)
	}
	getLogger().Warn("bulk embedding item failed", "id", item.ID, "error", result.Err)
	return result
}

// isFatal reports whether err means no other item can succeed either:
// rejected or expired credentials, missing permission, or a model id that
// doesn't exist (or isn't enabled) in the client's region.
func isFatal(err error) bool {
	var apiErr smithy.APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	switch apiErr.ErrorCode() {
	case "AccessDeniedException",
		"ResourceNotFoundException",
		"UnrecognizedClientException",
		"InvalidSignatureException",
		"ExpiredTokenException":
		return true
	}
	return false
}
