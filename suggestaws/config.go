package suggestaws

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

const (
	// DefaultRegion is where both Nova models and Guardrails are offered.
	DefaultRegion = "us-east-1"
	// DefaultLiteModelID is the cross-region inference profile of Nova 2 Lite.
	DefaultLiteModelID = "us.amazon.nova-2-lite-v1:0"
	// DefaultEmbedModelID is Nova Multimodal Embeddings, the model the album's
	// image vectors were indexed with.
	DefaultEmbedModelID = "amazon.nova-2-multimodal-embeddings-v1:0"
	// DefaultDimension must match the stored image vectors.
	DefaultDimension = 1024
	// DefaultGuardrailVersion screens against the working draft. Pin a
	// numbered version in production.
	DefaultGuardrailVersion = "DRAFT"
)

var validDimensions = map[int]bool{256: true, 384: true, 1024: true, 3072: true}

// Config is everything the package needs besides AWS credentials. See
// section 3.6 of the design doc.
type Config struct {
	Region       string // AWS_REGION
	LiteModelID  string // NOVA_LITE_MODEL_ID
	EmbedModelID string // NOVA_MODEL_ID
	Dimension    int    // NOVA_DIM
	// GuardrailID is GUARDRAIL_ID. When empty, Screen returns
	// ErrGuardrailNotConfigured and the caller should turn suggestions off
	// (fail closed).
	GuardrailID      string
	GuardrailVersion string // GUARDRAIL_VERSION
}

// DefaultConfig returns the documented defaults, with no guardrail set.
func DefaultConfig() Config {
	return Config{
		Region:           DefaultRegion,
		LiteModelID:      DefaultLiteModelID,
		EmbedModelID:     DefaultEmbedModelID,
		Dimension:        DefaultDimension,
		GuardrailVersion: DefaultGuardrailVersion,
	}
}

// ConfigFromEnv reads AWS_REGION, NOVA_LITE_MODEL_ID, NOVA_MODEL_ID,
// NOVA_DIM, GUARDRAIL_ID and GUARDRAIL_VERSION, falling back to
// DefaultConfig for anything unset.
func ConfigFromEnv() (Config, error) {
	cfg := DefaultConfig()
	setString := func(dst *string, key string) {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" {
			*dst = v
		}
	}
	setString(&cfg.Region, "AWS_REGION")
	setString(&cfg.LiteModelID, "NOVA_LITE_MODEL_ID")
	setString(&cfg.EmbedModelID, "NOVA_MODEL_ID")
	setString(&cfg.GuardrailID, "GUARDRAIL_ID")
	setString(&cfg.GuardrailVersion, "GUARDRAIL_VERSION")
	if v := strings.TrimSpace(os.Getenv("NOVA_DIM")); v != "" {
		dim, err := strconv.Atoi(v)
		if err != nil {
			return Config{}, fmt.Errorf("suggestaws: NOVA_DIM %q is not an integer", v)
		}
		cfg.Dimension = dim
	}
	return cfg, cfg.validate()
}

// withDefaults fills empty fields with the documented defaults, so a
// partially filled Config literal behaves like DefaultConfig.
func (c Config) withDefaults() Config {
	d := DefaultConfig()
	if c.Region == "" {
		c.Region = d.Region
	}
	if c.LiteModelID == "" {
		c.LiteModelID = d.LiteModelID
	}
	if c.EmbedModelID == "" {
		c.EmbedModelID = d.EmbedModelID
	}
	if c.Dimension == 0 {
		c.Dimension = d.Dimension
	}
	if c.GuardrailVersion == "" {
		c.GuardrailVersion = d.GuardrailVersion
	}
	return c
}

func (c Config) validate() error {
	if !validDimensions[c.Dimension] {
		return fmt.Errorf("suggestaws: NOVA_DIM must be 256, 384, 1024 or 3072, got %d", c.Dimension)
	}
	return nil
}
