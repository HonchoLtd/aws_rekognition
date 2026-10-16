// Package modelembedding is a thin SDK wrapper around Amazon Nova
// Multimodal Embeddings (amazon.nova-2-multimodal-embeddings-v1:0) on
// Amazon Bedrock. It does one job: turn an image (bytes or an S3 object) or
// a text query into an embedding vector, and hand that vector back to the
// caller.
//
// Persisting vectors (pgvector, OpenSearch, ...), deduplicating inputs, and
// running similarity search are all the caller's responsibility — the same
// split facev2 uses between the SDK and facev2/store. Each Bedrock
// InvokeModel call embeds exactly one input, so throughput comes from
// running calls concurrently (see EmbedImages), not from where the vectors
// get written.
//
// See: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
package modelembedding

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const (
	// DefaultModelId is the Bedrock model id of Nova Multimodal Embeddings.
	DefaultModelId = "amazon.nova-2-multimodal-embeddings-v1:0"
	// DefaultRegion is where DefaultModelId is offered. The Bedrock client
	// handed to NewNovaEmbedder must be configured for a region that serves
	// the model.
	DefaultRegion = "us-east-1"
	// DefaultDimension is used when WithDimension isn't passed. It is the
	// largest size Nova offers that still fits pgvector's 2000-dimension
	// HNSW limit.
	DefaultDimension = 1024

	// MaxTextChars is Nova's limit on an inline text value.
	MaxTextChars = 8192
	// MaxInlineRequestBytes is Nova's limit on inline content, measured
	// after base64 encoding.
	MaxInlineRequestBytes = 25 * 1024 * 1024

	schemaVersion       = "nova-multimodal-embed-v1"
	taskSingleEmbedding = "SINGLE_EMBEDDING"
)

// Purpose is Nova's embeddingPurpose. Index content with
// PurposeGenericIndex, then embed queries with the *_RETRIEVAL purpose that
// matches what was indexed (PurposeImageRetrieval for a text query against
// an index of images).
type Purpose string

const (
	PurposeGenericIndex      Purpose = "GENERIC_INDEX"
	PurposeGenericRetrieval  Purpose = "GENERIC_RETRIEVAL"
	PurposeTextRetrieval     Purpose = "TEXT_RETRIEVAL"
	PurposeImageRetrieval    Purpose = "IMAGE_RETRIEVAL"
	PurposeDocumentRetrieval Purpose = "DOCUMENT_RETRIEVAL"
	PurposeClassification    Purpose = "CLASSIFICATION"
	PurposeClustering        Purpose = "CLUSTERING"
)

// ImageFormat is one of the image formats Nova accepts.
type ImageFormat string

const (
	ImageFormatJPEG ImageFormat = "jpeg"
	ImageFormatPNG  ImageFormat = "png"
	ImageFormatGIF  ImageFormat = "gif"
	ImageFormatWEBP ImageFormat = "webp"
)

// DetailLevel controls the resolution Nova processes an image at.
// DetailLevelStandard (Nova's default) is right for photos;
// DetailLevelDocument reads text inside images better, and must be paired
// with PurposeDocumentRetrieval queries rather than PurposeImageRetrieval.
type DetailLevel string

const (
	DetailLevelStandard DetailLevel = "STANDARD_IMAGE"
	DetailLevelDocument DetailLevel = "DOCUMENT_IMAGE"
)

// TruncationMode controls what Nova does with text longer than its token
// limit.
type TruncationMode string

const (
	TruncateStart TruncationMode = "START"
	TruncateEnd   TruncationMode = "END"
	TruncateNone  TruncationMode = "NONE"
)

var (
	ErrEmptyInput             = errors.New("modelembedding: empty input")
	ErrInputTooLarge          = errors.New("modelembedding: input exceeds Nova's size limit")
	ErrInvalidDimension       = errors.New("modelembedding: invalid embedding dimension (want 256, 384, 1024, or 3072)")
	ErrUnsupportedImageFormat = errors.New("modelembedding: unsupported image format")
	ErrUnexpectedResponse     = errors.New("modelembedding: unexpected Nova response")
)

var validDimensions = map[int]bool{256: true, 384: true, 1024: true, 3072: true}

// Embedding is a single vector returned by Nova, plus the parameters that
// produced it. Vectors are only comparable with other vectors from the same
// ModelId and Dimension.
type Embedding struct {
	Vector    []float32 `json:"vector"`
	Dimension int       `json:"dimension"`
	ModelId   string    `json:"model_id"`
	Purpose   Purpose   `json:"purpose"`
	// TruncatedCharLength is set only for text that exceeded Nova's token
	// limit: it's the character index after which the text was ignored.
	TruncatedCharLength *int `json:"truncated_char_length,omitempty"`
}

type Embedder interface {
	// EmbedImage embeds raw image bytes (JPEG, PNG, GIF, or WEBP; or any of
	// those plus BMP/TIFF when WithPreprocess is set). Purpose defaults to
	// PurposeGenericIndex.
	EmbedImage(ctx context.Context, image []byte, opts ...EmbedOption) (Embedding, error)

	// EmbedImageWithBucket embeds an image Bedrock reads directly from S3.
	// The format is taken from the key's extension unless WithImageFormat
	// is passed. WithPreprocess is ignored, since the bytes never pass
	// through this process. Purpose defaults to PurposeGenericIndex.
	EmbedImageWithBucket(ctx context.Context, s3Bucket string, s3Key string, opts ...EmbedOption) (Embedding, error)

	// EmbedText embeds a text query of at most MaxTextChars characters.
	// Purpose defaults to PurposeImageRetrieval, for searching an index of
	// images built with EmbedImage.
	EmbedText(ctx context.Context, text string, opts ...EmbedOption) (Embedding, error)
}

// InvokeModelAPI is the subset of *bedrockruntime.Client this package uses.
type InvokeModelAPI interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

type novaEmbedder struct {
	client  InvokeModelAPI
	modelId string
	retry   retryPolicy
}

// NewNovaEmbedder returns an Embedder backed by DefaultModelId. client is
// usually a *bedrockruntime.Client configured for DefaultRegion.
func NewNovaEmbedder(client InvokeModelAPI) Embedder {
	return NewNovaEmbedderWithModelId(client, DefaultModelId)
}

// NewNovaEmbedderWithModelId is NewNovaEmbedder for a different model id or
// inference profile (for example a cross-region "us." profile) that speaks
// the same request schema.
func NewNovaEmbedderWithModelId(client InvokeModelAPI, modelId string) Embedder {
	return &novaEmbedder{client: client, modelId: modelId, retry: defaultRetryPolicy}
}

// EmbedImage implements Embedder.
func (n *novaEmbedder) EmbedImage(ctx context.Context, image []byte, opts ...EmbedOption) (Embedding, error) {
	cfg := newEmbedConfig(PurposeGenericIndex, opts)
	if err := cfg.validate(); err != nil {
		return Embedding{}, err
	}
	if len(image) == 0 {
		return Embedding{}, fmt.Errorf("%w: image has no bytes", ErrEmptyInput)
	}

	originalBytes := len(image)
	format := cfg.imageFormat
	if cfg.preprocess {
		processed, err := PreprocessImage(image, cfg.maxImageSide)
		if err != nil {
			return Embedding{}, err
		}
		image = processed
		format = ImageFormatJPEG
	} else if format == "" {
		detected, err := DetectImageFormat(image)
		if err != nil {
			return Embedding{}, err
		}
		format = detected
	}

	if encoded := base64.StdEncoding.EncodedLen(len(image)); encoded > MaxInlineRequestBytes {
		return Embedding{}, fmt.Errorf("%w: image is %d bytes after base64, limit is %d (use EmbedImageWithBucket or WithPreprocess)", ErrInputTooLarge, encoded, MaxInlineRequestBytes)
	}

	params := singleEmbeddingParams{
		EmbeddingPurpose:   cfg.purpose,
		EmbeddingDimension: cfg.dimension,
		Image: &imageInput{
			Format:      format,
			DetailLevel: cfg.detailLevel,
			Source:      source{Bytes: image},
		},
	}
	return n.invoke(ctx, "EmbedImage", params, "original_bytes", originalBytes, "sent_bytes", len(image), "format", format, "preprocessed", cfg.preprocess)
}

// EmbedImageWithBucket implements Embedder.
func (n *novaEmbedder) EmbedImageWithBucket(ctx context.Context, s3Bucket string, s3Key string, opts ...EmbedOption) (Embedding, error) {
	cfg := newEmbedConfig(PurposeGenericIndex, opts)
	if err := cfg.validate(); err != nil {
		return Embedding{}, err
	}
	if s3Bucket == "" || s3Key == "" {
		return Embedding{}, fmt.Errorf("%w: s3Bucket and s3Key are both required", ErrEmptyInput)
	}

	format := cfg.imageFormat
	if format == "" {
		detected, ok := imageFormatFromExtension(s3Key)
		if !ok {
			return Embedding{}, fmt.Errorf("%w: can't infer format from S3 key %q (pass WithImageFormat)", ErrUnsupportedImageFormat, s3Key)
		}
		format = detected
	}

	uri := "s3://" + s3Bucket + "/" + strings.TrimPrefix(s3Key, "/")
	params := singleEmbeddingParams{
		EmbeddingPurpose:   cfg.purpose,
		EmbeddingDimension: cfg.dimension,
		Image: &imageInput{
			Format:      format,
			DetailLevel: cfg.detailLevel,
			Source:      source{S3Location: &s3Location{URI: uri}},
		},
	}
	return n.invoke(ctx, "EmbedImageWithBucket", params, "s3_uri", uri, "format", format)
}

// EmbedText implements Embedder.
func (n *novaEmbedder) EmbedText(ctx context.Context, text string, opts ...EmbedOption) (Embedding, error) {
	cfg := newEmbedConfig(PurposeImageRetrieval, opts)
	if err := cfg.validate(); err != nil {
		return Embedding{}, err
	}
	if strings.TrimSpace(text) == "" {
		return Embedding{}, fmt.Errorf("%w: text is blank", ErrEmptyInput)
	}
	if chars := utf8.RuneCountInString(text); chars > MaxTextChars {
		return Embedding{}, fmt.Errorf("%w: text is %d characters, limit is %d", ErrInputTooLarge, chars, MaxTextChars)
	}

	params := singleEmbeddingParams{
		EmbeddingPurpose:   cfg.purpose,
		EmbeddingDimension: cfg.dimension,
		Text: &textInput{
			TruncationMode: cfg.truncationMode,
			Value:          text,
		},
	}
	return n.invoke(ctx, "EmbedText", params, "chars", utf8.RuneCountInString(text))
}

// invoke sends params to Bedrock (retrying transient failures) and parses
// the single embedding Nova returns. logAttrs are extra slog key/value
// pairs describing the input.
func (n *novaEmbedder) invoke(ctx context.Context, op string, params singleEmbeddingParams, logAttrs ...any) (Embedding, error) {
	body, err := json.Marshal(novaRequest{
		SchemaVersion:         schemaVersion,
		TaskType:              taskSingleEmbedding,
		SingleEmbeddingParams: params,
	})
	if err != nil {
		return Embedding{}, fmt.Errorf("modelembedding: marshal request: %w", err)
	}

	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(n.modelId),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        body,
	}

	var out *bedrockruntime.InvokeModelOutput
	start := time.Now()
	err = withRetry(ctx, n.retry, op, func() error {
		var callErr error
		out, callErr = n.client.InvokeModel(ctx, input)
		return callErr
	})
	duration := time.Since(start)

	baseAttrs := []any{"op", op, "model_id", n.modelId, "purpose", params.EmbeddingPurpose, "dimension", params.EmbeddingDimension, "duration_ms", duration.Milliseconds()}
	if err != nil {
		getLogger().Error("bedrock api call failed", append(append(baseAttrs, logAttrs...), "error", err)...)
		return Embedding{}, fmt.Errorf("modelembedding: %s: invoke %s: %w", op, n.modelId, err)
	}

	embedding, err := parseResponse(out.Body, params.EmbeddingDimension)
	if err != nil {
		getLogger().Error("bedrock response could not be parsed", append(append(baseAttrs, logAttrs...), "error", err)...)
		return Embedding{}, fmt.Errorf("modelembedding: %s: %w", op, err)
	}
	embedding.ModelId = n.modelId
	embedding.Purpose = params.EmbeddingPurpose

	getLogger().Info("bedrock api call", append(baseAttrs, logAttrs...)...)
	if embedding.TruncatedCharLength != nil {
		getLogger().Warn("text was truncated by Nova", "op", op, "truncated_char_length", *embedding.TruncatedCharLength)
	}
	return embedding, nil
}

// parseResponse extracts the single embedding from an InvokeModel response
// body and checks it has the dimension that was requested.
func parseResponse(body []byte, wantDimension int) (Embedding, error) {
	var resp novaResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return Embedding{}, fmt.Errorf("%w: decode body: %v", ErrUnexpectedResponse, err)
	}
	if len(resp.Embeddings) == 0 {
		return Embedding{}, fmt.Errorf("%w: no embeddings in response", ErrUnexpectedResponse)
	}
	first := resp.Embeddings[0]
	if len(first.Embedding) != wantDimension {
		return Embedding{}, fmt.Errorf("%w: got %d-dimension vector, requested %d", ErrUnexpectedResponse, len(first.Embedding), wantDimension)
	}
	return Embedding{
		Vector:              first.Embedding,
		Dimension:           len(first.Embedding),
		TruncatedCharLength: first.TruncatedCharLength,
	}, nil
}

// imageFormatFromExtension maps a file name or S3 key to the Nova format
// its extension implies.
func imageFormatFromExtension(name string) (ImageFormat, bool) {
	switch strings.ToLower(path.Ext(name)) {
	case ".jpg", ".jpeg":
		return ImageFormatJPEG, true
	case ".png":
		return ImageFormatPNG, true
	case ".gif":
		return ImageFormatGIF, true
	case ".webp":
		return ImageFormatWEBP, true
	}
	return "", false
}

type novaRequest struct {
	SchemaVersion         string                `json:"schemaVersion"`
	TaskType              string                `json:"taskType"`
	SingleEmbeddingParams singleEmbeddingParams `json:"singleEmbeddingParams"`
}

type singleEmbeddingParams struct {
	EmbeddingPurpose   Purpose     `json:"embeddingPurpose"`
	EmbeddingDimension int         `json:"embeddingDimension"`
	Text               *textInput  `json:"text,omitempty"`
	Image              *imageInput `json:"image,omitempty"`
}

type textInput struct {
	TruncationMode TruncationMode `json:"truncationMode"`
	Value          string         `json:"value"`
}

type imageInput struct {
	Format      ImageFormat `json:"format"`
	DetailLevel DetailLevel `json:"detailLevel,omitempty"`
	Source      source      `json:"source"`
}

// source holds exactly one of Bytes (base64-encoded by encoding/json) or
// S3Location.
type source struct {
	Bytes      []byte      `json:"bytes,omitempty"`
	S3Location *s3Location `json:"s3Location,omitempty"`
}

type s3Location struct {
	URI string `json:"uri"`
}

type novaResponse struct {
	Embeddings []struct {
		EmbeddingType       string    `json:"embeddingType"`
		Embedding           []float32 `json:"embedding"`
		TruncatedCharLength *int      `json:"truncatedCharLength"`
	} `json:"embeddings"`
}
