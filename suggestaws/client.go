// Package suggestaws is System A of the Smart Search Suggestions design: a
// stateless package that only talks to AWS. It exposes three operations:
//
//   - ProposeTerms asks Nova 2 Lite for search terms for numbered groups of
//     photos (Bedrock Converse with a forced tool call).
//   - Screen checks terms against a Bedrock Guardrail.
//   - EmbedText embeds a term with Nova Multimodal Embeddings
//     (IMAGE_RETRIEVAL), so it can be compared with the album's image
//     vectors.
//
// Each operation reports its own usage so the caller can cost a cycle.
// The package holds no state beyond AWS clients and config: it never
// touches a database, never clusters, and never decides which term wins.
// It knows nothing about albums or clusters beyond "numbered groups of
// images"; mapping group numbers back to cluster IDs is the caller's job
// (System B).
//
// EnsureGuardrail (guardrail.go) creates or updates the guardrail policy
// and is meant for a one-off admin command, not the request path.
package suggestaws

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awshttp "github.com/aws/aws-sdk-go-v2/aws/transport/http"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Client is the interface System B depends on. *AWSClient implements it;
// tests in B use a fake.
type Client interface {
	ProposeTerms(ctx context.Context, req ProposeRequest) (ProposeResult, error)
	Screen(ctx context.Context, terms []string) (ScreenResult, error)
	// EmbedText embeds text with purpose IMAGE_RETRIEVAL and returns the
	// raw (not normalised) vector of Config.Dimension floats.
	EmbedText(ctx context.Context, text string) ([]float32, error)
}

// ProposeRequest is one Nova 2 Lite call.
type ProposeRequest struct {
	// Context lists chips that already have a term (one photo each). They
	// are shown so Lite avoids repeating them; Lite is not asked to name
	// them.
	Context []NamedPhoto
	// Groups need a term, largest first. Group i must have Number i+1.
	Groups []Group
	// Temperature: 0 on the first attempt, 0.7 on retries. Range [0, 1].
	Temperature float32
}

// Group is one set of photos that needs a term.
type Group struct {
	Number int      // 1-based position in this request, NOT the cluster ID
	Size   int      // photo count shown to the model
	JPEGs  [][]byte // representative photos, already resized and JPEG-encoded
	Avoid  []string // rejected terms for this group (blocked or failed earlier)
}

// NamedPhoto is one already-named chip shown as context.
type NamedPhoto struct {
	Term string
	JPEG []byte
}

// ProposeResult is what Lite answered.
type ProposeResult struct {
	// Terms maps group number to Lite's raw terms, best first. An empty,
	// non-nil slice means Lite deliberately refused the group. A group
	// missing from the map got no answer at all.
	Terms        map[int][]string
	InputTokens  int
	OutputTokens int
}

// ScreenResult is the guardrail's verdict on a list of terms.
type ScreenResult struct {
	Allowed []string          // in input order, de-duplicated
	Blocked map[string]string // term -> reasons, e.g. "topic:PersonalAttributes,word:women"
	Calls   int               // billed ApplyGuardrail calls
}

// RuntimeAPI is the subset of *bedrockruntime.Client the package uses.
type RuntimeAPI interface {
	Converse(ctx context.Context, params *bedrockruntime.ConverseInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseOutput, error)
	ApplyGuardrail(ctx context.Context, params *bedrockruntime.ApplyGuardrailInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.ApplyGuardrailOutput, error)
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

// AWSClient implements Client against Amazon Bedrock. It is safe for
// concurrent use.
type AWSClient struct {
	rt    RuntimeAPI
	cfg   Config
	retry retryPolicy
}

var _ Client = (*AWSClient)(nil)

// New wraps an existing Bedrock runtime client. Empty Config fields take
// their defaults. Build rt with NewAWSConfig so the SDK's own retries are
// off and the timeouts match the design.
func New(rt RuntimeAPI, cfg Config) (*AWSClient, error) {
	if rt == nil {
		return nil, fmt.Errorf("%w: nil Bedrock runtime client", ErrInvalidRequest)
	}
	cfg = cfg.withDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &AWSClient{rt: rt, cfg: cfg, retry: defaultRetryPolicy}, nil
}

// NewFromAWSConfig builds the Bedrock runtime client from awsCfg (usually
// from NewAWSConfig) and wraps it.
func NewFromAWSConfig(awsCfg aws.Config, cfg Config) (*AWSClient, error) {
	return New(bedrockruntime.NewFromConfig(awsCfg), cfg)
}

// Config returns the effective configuration, defaults filled in.
func (c *AWSClient) Config() Config { return c.cfg }

// Transport settings from section 3.5.
const (
	connectTimeout = 10 * time.Second
	readTimeout    = 60 * time.Second
	keepAlive      = 30 * time.Second
	maxConns       = 32
)

// NewHTTPClient returns the HTTP client the design calls for: connect
// timeout 10 s, read (response) timeout 60 s, TCP keepalive on, a pool of
// 32 connections.
func NewHTTPClient() *awshttp.BuildableClient {
	return awshttp.NewBuildableClient().
		WithDialerOptions(func(d *net.Dialer) {
			d.Timeout = connectTimeout
			d.KeepAlive = keepAlive
		}).
		WithTransportOptions(func(t *http.Transport) {
			t.TLSHandshakeTimeout = connectTimeout
			// Converse is not streamed, so the headers arrive only once
			// the model has finished: this is effectively the read timeout.
			t.ResponseHeaderTimeout = readTimeout
			t.MaxIdleConns = maxConns
			t.MaxIdleConnsPerHost = maxConns
			t.MaxConnsPerHost = maxConns
		})
}

// NewAWSConfig loads the default AWS config (credential chain, profile …)
// for region with NewHTTPClient and the SDK retryer turned off, because
// this package retries by itself. Extra options (for example static
// credentials) are applied last.
func NewAWSConfig(ctx context.Context, region string, optFns ...func(*awsconfig.LoadOptions) error) (aws.Config, error) {
	if region == "" {
		region = DefaultRegion
	}
	opts := append([]func(*awsconfig.LoadOptions) error{
		awsconfig.WithRegion(region),
		awsconfig.WithHTTPClient(NewHTTPClient()),
		awsconfig.WithRetryer(func() aws.Retryer { return aws.NopRetryer{} }),
	}, optFns...)
	return awsconfig.LoadDefaultConfig(ctx, opts...)
}
