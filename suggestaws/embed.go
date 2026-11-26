package suggestaws

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// DefaultEmbedConcurrency is how many EmbedText calls EmbedTexts runs at
// once (section 3.4).
const DefaultEmbedConcurrency = 4

// purposeImageRetrieval matches how user search queries are embedded, so
// a term is checked the same way a guest's search would be (the photos
// were indexed with GENERIC_INDEX).
const purposeImageRetrieval = "IMAGE_RETRIEVAL"

type embedRequest struct {
	SchemaVersion         string `json:"schemaVersion"`
	TaskType              string `json:"taskType"`
	SingleEmbeddingParams struct {
		EmbeddingDimension int    `json:"embeddingDimension"`
		EmbeddingPurpose   string `json:"embeddingPurpose"`
		Text               struct {
			TruncationMode string `json:"truncationMode"`
			Value          string `json:"value"`
		} `json:"text"`
	} `json:"singleEmbeddingParams"`
}

type embedResponse struct {
	Embeddings []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"embeddings"`
}

// EmbedText implements Client. The vector is returned as Nova sent it;
// normalising it is the caller's job.
func (c *AWSClient) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("%w: EmbedText needs non-blank text", ErrInvalidRequest)
	}
	var req embedRequest
	req.SchemaVersion = "nova-multimodal-embed-v1"
	req.TaskType = "SINGLE_EMBEDDING"
	req.SingleEmbeddingParams.EmbeddingDimension = c.cfg.Dimension
	req.SingleEmbeddingParams.EmbeddingPurpose = purposeImageRetrieval
	req.SingleEmbeddingParams.Text.TruncationMode = "END"
	req.SingleEmbeddingParams.Text.Value = text
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("suggestaws: EmbedText: marshal request: %w", err)
	}

	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(c.cfg.EmbedModelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        body,
	}
	var out *bedrockruntime.InvokeModelOutput
	start := time.Now()
	err = withRetry(ctx, c.retry, "EmbedText", func() error {
		var callErr error
		out, callErr = c.rt.InvokeModel(ctx, input)
		return callErr
	})
	attrs := []any{"op", "EmbedText", "model_id", c.cfg.EmbedModelID, "chars", len(text), "duration_ms", time.Since(start).Milliseconds()}
	if err != nil {
		getLogger().Error("bedrock embed text failed", append(attrs, "error", err)...)
		return nil, fmt.Errorf("suggestaws: EmbedText: %w", err)
	}

	var resp embedResponse
	if out == nil {
		return nil, fmt.Errorf("suggestaws: EmbedText: %w: nil InvokeModel output", ErrUnexpectedResponse)
	}
	if err := json.Unmarshal(out.Body, &resp); err != nil {
		return nil, fmt.Errorf("suggestaws: EmbedText: %w: decode body: %v", ErrUnexpectedResponse, err)
	}
	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("suggestaws: EmbedText: %w: no embeddings in response", ErrUnexpectedResponse)
	}
	vec := resp.Embeddings[0].Embedding
	if len(vec) != c.cfg.Dimension {
		return nil, fmt.Errorf("suggestaws: EmbedText: %w: got %d-d vector, want %d", ErrUnexpectedResponse, len(vec), c.cfg.Dimension)
	}
	getLogger().Debug("bedrock embed text", attrs...)
	return vec, nil
}

// EmbedTexts embeds every text through c, at most concurrency at a time
// (DefaultEmbedConcurrency when <= 0). Vectors come back in input order.
// The first error cancels the remaining calls and is returned.
func EmbedTexts(ctx context.Context, c Client, texts []string, concurrency int) ([][]float32, error) {
	if concurrency <= 0 {
		concurrency = DefaultEmbedConcurrency
	}
	vectors := make([][]float32, len(texts))
	if len(texts) == 0 {
		return vectors, nil
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, concurrency)
	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)
	for i, text := range texts {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
		}
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		go func(i int, text string) {
			defer wg.Done()
			defer func() { <-sem }()
			vec, err := c.EmbedText(ctx, text)
			if err != nil {
				once.Do(func() {
					firstErr = fmt.Errorf("embed %q: %w", text, err)
					cancel()
				})
				return
			}
			vectors[i] = vec
		}(i, text)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return vectors, nil
}
