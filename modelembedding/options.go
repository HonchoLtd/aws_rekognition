package modelembedding

import "fmt"

// embedConfig holds the optional parameters shared by every Embedder
// method. Options that don't apply to a method (e.g. WithTruncationMode on
// EmbedImage) are ignored by it.
type embedConfig struct {
	dimension      int
	purpose        Purpose
	imageFormat    ImageFormat
	detailLevel    DetailLevel
	truncationMode TruncationMode
	preprocess     bool
	maxImageSide   int
}

// EmbedOption configures a single EmbedImage / EmbedImageWithBucket /
// EmbedText call.
type EmbedOption func(*embedConfig)

func newEmbedConfig(defaultPurpose Purpose, opts []EmbedOption) embedConfig {
	cfg := embedConfig{
		dimension:      DefaultDimension,
		purpose:        defaultPurpose,
		truncationMode: TruncateEnd,
		maxImageSide:   DefaultMaxImageSide,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func (c embedConfig) validate() error {
	if !validDimensions[c.dimension] {
		return fmt.Errorf("%w: got %d", ErrInvalidDimension, c.dimension)
	}
	if c.preprocess && c.maxImageSide <= 0 {
		return fmt.Errorf("modelembedding: max image side must be positive, got %d", c.maxImageSide)
	}
	return nil
}

// WithDimension sets the vector size: 256, 384, 1024 (default), or 3072.
// Every vector stored in one index must use the same dimension.
func WithDimension(dimension int) EmbedOption {
	return func(c *embedConfig) { c.dimension = dimension }
}

// WithPurpose overrides the default embeddingPurpose (PurposeGenericIndex
// for images, PurposeImageRetrieval for text).
func WithPurpose(purpose Purpose) EmbedOption {
	return func(c *embedConfig) { c.purpose = purpose }
}

// WithImageFormat skips format detection. For EmbedImageWithBucket it is
// required when the S3 key has no recognizable extension.
func WithImageFormat(format ImageFormat) EmbedOption {
	return func(c *embedConfig) { c.imageFormat = format }
}

// WithDetailLevel sets the image detailLevel. Nova defaults to
// DetailLevelStandard when omitted.
func WithDetailLevel(level DetailLevel) EmbedOption {
	return func(c *embedConfig) { c.detailLevel = level }
}

// WithTruncationMode sets how over-long text is handled. Defaults to
// TruncateEnd; TruncateNone makes Nova fail instead.
func WithTruncationMode(mode TruncationMode) EmbedOption {
	return func(c *embedConfig) { c.truncationMode = mode }
}

// WithPreprocess makes EmbedImage run PreprocessImage before uploading:
// apply the JPEG EXIF orientation, flatten transparency onto white, shrink
// so the longest side is at most DefaultMaxImageSide, and re-encode as
// JPEG. It also lets EmbedImage accept BMP and TIFF input.
func WithPreprocess() EmbedOption {
	return func(c *embedConfig) { c.preprocess = true }
}

// WithPreprocessMaxSide is WithPreprocess with a custom longest-side limit
// in pixels.
func WithPreprocessMaxSide(maxSide int) EmbedOption {
	return func(c *embedConfig) {
		c.preprocess = true
		c.maxImageSide = maxSide
	}
}
