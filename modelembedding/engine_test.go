package modelembedding

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsv2_config "github.com/aws/aws-sdk-go-v2/config"
	awsv2_credentials "github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/joho/godotenv"
)

// fakeClient is an InvokeModelAPI that records every request and answers
// from errs (consumed in order) and then with a vector of the requested
// dimension.
type fakeClient struct {
	inputs   []*bedrockruntime.InvokeModelInput
	errs     []error
	respBody func(req novaRequest) []byte
}

func (f *fakeClient) InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	f.inputs = append(f.inputs, params)
	if len(f.errs) > 0 {
		err := f.errs[0]
		f.errs = f.errs[1:]
		return nil, err
	}
	var req novaRequest
	if err := json.Unmarshal(params.Body, &req); err != nil {
		return nil, err
	}
	if f.respBody != nil {
		return &bedrockruntime.InvokeModelOutput{Body: f.respBody(req)}, nil
	}
	return &bedrockruntime.InvokeModelOutput{Body: vectorResponse(req.SingleEmbeddingParams.EmbeddingDimension, nil)}, nil
}

// lastRequest decodes the body of the most recent InvokeModel call into a
// generic map, so assertions check the exact JSON field names Nova expects.
func (f *fakeClient) lastRequest(t *testing.T) map[string]any {
	t.Helper()
	if len(f.inputs) == 0 {
		t.Fatal("InvokeModel was never called")
	}
	var body map[string]any
	if err := json.Unmarshal(f.inputs[len(f.inputs)-1].Body, &body); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	return body
}

func vectorResponse(dimension int, truncatedCharLength *int) []byte {
	vector := make([]float32, dimension)
	for i := range vector {
		vector[i] = float32(i) / float32(dimension)
	}
	entry := map[string]any{"embeddingType": "IMAGE", "embedding": vector}
	if truncatedCharLength != nil {
		entry["truncatedCharLength"] = *truncatedCharLength
	}
	body, _ := json.Marshal(map[string]any{"embeddings": []any{entry}})
	return body
}

func newTestEmbedder(client InvokeModelAPI) *novaEmbedder {
	return &novaEmbedder{client: client, modelId: DefaultModelId, retry: fastRetryPolicy}
}

func pngBytes(t *testing.T, width, height int, fill color.Color) []byte {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, fill)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	return buf.Bytes()
}

func params(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	p, ok := body["singleEmbeddingParams"].(map[string]any)
	if !ok {
		t.Fatalf("singleEmbeddingParams missing from request: %v", body)
	}
	return p
}

func TestEmbedImage_BuildsNovaRequestAndParsesResponse(t *testing.T) {
	client := &fakeClient{}
	img := pngBytes(t, 4, 4, color.NRGBA{R: 255, A: 255})

	got, err := newTestEmbedder(client).EmbedImage(context.Background(), img)
	if err != nil {
		t.Fatalf("EmbedImage() error = %v", err)
	}

	input := client.inputs[0]
	if aws.ToString(input.ModelId) != DefaultModelId {
		t.Errorf("ModelId = %q, want %q", aws.ToString(input.ModelId), DefaultModelId)
	}
	if aws.ToString(input.ContentType) != "application/json" {
		t.Errorf("ContentType = %q, want application/json", aws.ToString(input.ContentType))
	}

	body := client.lastRequest(t)
	if body["schemaVersion"] != "nova-multimodal-embed-v1" || body["taskType"] != "SINGLE_EMBEDDING" {
		t.Errorf("schemaVersion/taskType = %v/%v", body["schemaVersion"], body["taskType"])
	}
	p := params(t, body)
	if p["embeddingPurpose"] != "GENERIC_INDEX" {
		t.Errorf("embeddingPurpose = %v, want GENERIC_INDEX", p["embeddingPurpose"])
	}
	if p["embeddingDimension"] != float64(1024) {
		t.Errorf("embeddingDimension = %v, want 1024", p["embeddingDimension"])
	}
	if _, hasText := p["text"]; hasText {
		t.Error("image request unexpectedly carries a text field")
	}
	imageField := p["image"].(map[string]any)
	if imageField["format"] != "png" {
		t.Errorf("image.format = %v, want png (detected from bytes)", imageField["format"])
	}
	if _, hasDetail := imageField["detailLevel"]; hasDetail {
		t.Error("detailLevel should be omitted unless WithDetailLevel is passed")
	}
	sent := imageField["source"].(map[string]any)["bytes"].(string)
	if sent != base64.StdEncoding.EncodeToString(img) {
		t.Error("image.source.bytes is not the base64 of the original image")
	}

	if got.Dimension != 1024 || len(got.Vector) != 1024 {
		t.Errorf("Dimension = %d, len(Vector) = %d, want 1024", got.Dimension, len(got.Vector))
	}
	if got.ModelId != DefaultModelId || got.Purpose != PurposeGenericIndex {
		t.Errorf("ModelId/Purpose = %q/%q", got.ModelId, got.Purpose)
	}
	if got.Vector[512] != 0.5 {
		t.Errorf("Vector[512] = %v, want 0.5", got.Vector[512])
	}
}

func TestEmbedImage_Options(t *testing.T) {
	client := &fakeClient{}
	img := pngBytes(t, 2, 2, color.White)

	got, err := newTestEmbedder(client).EmbedImage(context.Background(), img,
		WithDimension(256),
		WithPurpose(PurposeClustering),
		WithDetailLevel(DetailLevelDocument),
		WithImageFormat(ImageFormatPNG),
	)
	if err != nil {
		t.Fatalf("EmbedImage() error = %v", err)
	}
	p := params(t, client.lastRequest(t))
	if p["embeddingDimension"] != float64(256) || p["embeddingPurpose"] != "CLUSTERING" {
		t.Errorf("dimension/purpose = %v/%v, want 256/CLUSTERING", p["embeddingDimension"], p["embeddingPurpose"])
	}
	if p["image"].(map[string]any)["detailLevel"] != "DOCUMENT_IMAGE" {
		t.Errorf("detailLevel = %v, want DOCUMENT_IMAGE", p["image"].(map[string]any)["detailLevel"])
	}
	if got.Dimension != 256 || got.Purpose != PurposeClustering {
		t.Errorf("Embedding = %d/%q, want 256/CLUSTERING", got.Dimension, got.Purpose)
	}
}

func TestEmbedImage_Preprocess(t *testing.T) {
	client := &fakeClient{}
	img := pngBytes(t, 2000, 1000, color.NRGBA{B: 255, A: 255})

	if _, err := newTestEmbedder(client).EmbedImage(context.Background(), img, WithPreprocess()); err != nil {
		t.Fatalf("EmbedImage() error = %v", err)
	}
	imageField := params(t, client.lastRequest(t))["image"].(map[string]any)
	if imageField["format"] != "jpeg" {
		t.Errorf("format = %v, want jpeg after preprocessing", imageField["format"])
	}
	sent, _ := base64.StdEncoding.DecodeString(imageField["source"].(map[string]any)["bytes"].(string))
	cfg, format, err := image.DecodeConfig(bytes.NewReader(sent))
	if err != nil {
		t.Fatalf("sent bytes don't decode: %v", err)
	}
	if format != "jpeg" || cfg.Width != 1024 || cfg.Height != 512 {
		t.Errorf("sent image = %s %dx%d, want jpeg 1024x512", format, cfg.Width, cfg.Height)
	}
}

func TestEmbedImage_RejectsBadInputWithoutCallingBedrock(t *testing.T) {
	tests := []struct {
		name    string
		image   []byte
		opts    []EmbedOption
		wantErr error
	}{
		{"empty", nil, nil, ErrEmptyInput},
		{"unsupported format", []byte("definitely not an image"), nil, ErrUnsupportedImageFormat},
		{"invalid dimension", []byte{0xFF, 0xD8, 0xFF}, []EmbedOption{WithDimension(512)}, ErrInvalidDimension},
		{"undecodable with preprocess", []byte("nope"), []EmbedOption{WithPreprocess()}, ErrUnsupportedImageFormat},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &fakeClient{}
			_, err := newTestEmbedder(client).EmbedImage(context.Background(), tt.image, tt.opts...)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("error = %v, want %v", err, tt.wantErr)
			}
			if len(client.inputs) != 0 {
				t.Errorf("InvokeModel called %d times, want 0", len(client.inputs))
			}
		})
	}
}

func TestEmbedImageWithBucket(t *testing.T) {
	client := &fakeClient{}
	if _, err := newTestEmbedder(client).EmbedImageWithBucket(context.Background(), "my-bucket", "/events/42/photo.JPG"); err != nil {
		t.Fatalf("EmbedImageWithBucket() error = %v", err)
	}
	imageField := params(t, client.lastRequest(t))["image"].(map[string]any)
	if imageField["format"] != "jpeg" {
		t.Errorf("format = %v, want jpeg from .JPG extension", imageField["format"])
	}
	src := imageField["source"].(map[string]any)
	if _, hasBytes := src["bytes"]; hasBytes {
		t.Error("S3 request unexpectedly carries inline bytes")
	}
	if uri := src["s3Location"].(map[string]any)["uri"]; uri != "s3://my-bucket/events/42/photo.JPG" {
		t.Errorf("s3Location.uri = %v", uri)
	}
}

func TestEmbedImageWithBucket_Format(t *testing.T) {
	client := &fakeClient{}
	e := newTestEmbedder(client)

	if _, err := e.EmbedImageWithBucket(context.Background(), "b", "no-extension"); !errors.Is(err, ErrUnsupportedImageFormat) {
		t.Errorf("unknown extension error = %v, want ErrUnsupportedImageFormat", err)
	}
	if _, err := e.EmbedImageWithBucket(context.Background(), "b", "no-extension", WithImageFormat(ImageFormatWEBP)); err != nil {
		t.Fatalf("with explicit format error = %v", err)
	}
	if f := params(t, client.lastRequest(t))["image"].(map[string]any)["format"]; f != "webp" {
		t.Errorf("format = %v, want webp", f)
	}
	if _, err := e.EmbedImageWithBucket(context.Background(), "", "a.jpg"); !errors.Is(err, ErrEmptyInput) {
		t.Errorf("missing bucket error = %v, want ErrEmptyInput", err)
	}
}

func TestEmbedText(t *testing.T) {
	truncatedAt := 42
	client := &fakeClient{respBody: func(req novaRequest) []byte {
		return vectorResponse(req.SingleEmbeddingParams.EmbeddingDimension, &truncatedAt)
	}}

	got, err := newTestEmbedder(client).EmbedText(context.Background(), "a dog on the beach")
	if err != nil {
		t.Fatalf("EmbedText() error = %v", err)
	}
	p := params(t, client.lastRequest(t))
	if p["embeddingPurpose"] != "IMAGE_RETRIEVAL" {
		t.Errorf("embeddingPurpose = %v, want IMAGE_RETRIEVAL", p["embeddingPurpose"])
	}
	if _, hasImage := p["image"]; hasImage {
		t.Error("text request unexpectedly carries an image field")
	}
	text := p["text"].(map[string]any)
	if text["value"] != "a dog on the beach" || text["truncationMode"] != "END" {
		t.Errorf("text = %v, want value + truncationMode END", text)
	}
	if got.Purpose != PurposeImageRetrieval {
		t.Errorf("Purpose = %q, want IMAGE_RETRIEVAL", got.Purpose)
	}
	if got.TruncatedCharLength == nil || *got.TruncatedCharLength != 42 {
		t.Errorf("TruncatedCharLength = %v, want 42", got.TruncatedCharLength)
	}
}

func TestEmbedText_RejectsBadInputWithoutCallingBedrock(t *testing.T) {
	client := &fakeClient{}
	e := newTestEmbedder(client)

	if _, err := e.EmbedText(context.Background(), "   "); !errors.Is(err, ErrEmptyInput) {
		t.Errorf("blank text error = %v, want ErrEmptyInput", err)
	}
	if _, err := e.EmbedText(context.Background(), strings.Repeat("é", MaxTextChars+1)); !errors.Is(err, ErrInputTooLarge) {
		t.Errorf("long text error = %v, want ErrInputTooLarge", err)
	}
	if _, err := e.EmbedText(context.Background(), strings.Repeat("é", MaxTextChars)); err != nil {
		t.Errorf("text at exactly the limit error = %v, want nil", err)
	}
	if len(client.inputs) != 1 {
		t.Errorf("InvokeModel called %d times, want 1", len(client.inputs))
	}
}

func TestEmbed_RetriesThrottlingThenSucceeds(t *testing.T) {
	client := &fakeClient{errs: []error{
		&types.ThrottlingException{Message: aws.String("slow down")},
		&types.ServiceUnavailableException{Message: aws.String("try again")},
	}}
	if _, err := newTestEmbedder(client).EmbedText(context.Background(), "query"); err != nil {
		t.Fatalf("EmbedText() error = %v, want success on attempt 3", err)
	}
	if len(client.inputs) != 3 {
		t.Errorf("InvokeModel called %d times, want 3", len(client.inputs))
	}
}

func TestEmbed_DoesNotRetryValidationAndKeepsErrorType(t *testing.T) {
	client := &fakeClient{errs: []error{&types.ValidationException{Message: aws.String("bad image")}}}
	_, err := newTestEmbedder(client).EmbedText(context.Background(), "query")
	var validation *types.ValidationException
	if !errors.As(err, &validation) {
		t.Errorf("error = %v, want it to wrap *types.ValidationException", err)
	}
	if len(client.inputs) != 1 {
		t.Errorf("InvokeModel called %d times, want 1", len(client.inputs))
	}
}

func TestParseResponse(t *testing.T) {
	if _, err := parseResponse([]byte(`{"embeddings":[]}`), 256); !errors.Is(err, ErrUnexpectedResponse) {
		t.Errorf("no embeddings error = %v, want ErrUnexpectedResponse", err)
	}
	if _, err := parseResponse(vectorResponse(384, nil), 256); !errors.Is(err, ErrUnexpectedResponse) {
		t.Errorf("dimension mismatch error = %v, want ErrUnexpectedResponse", err)
	}
	if _, err := parseResponse([]byte(`not json`), 256); !errors.Is(err, ErrUnexpectedResponse) {
		t.Errorf("bad JSON error = %v, want ErrUnexpectedResponse", err)
	}
	got, err := parseResponse(vectorResponse(256, nil), 256)
	if err != nil || got.Dimension != 256 || got.TruncatedCharLength != nil {
		t.Errorf("parseResponse() = %+v, %v", got.Dimension, err)
	}
}

func TestImageFormatFromExtension(t *testing.T) {
	tests := map[string]ImageFormat{"a.jpg": ImageFormatJPEG, "b/c.JPEG": ImageFormatJPEG, "d.png": ImageFormatPNG, "e.gif": ImageFormatGIF, "f.webp": ImageFormatWEBP}
	for name, want := range tests {
		if got, ok := imageFormatFromExtension(name); !ok || got != want {
			t.Errorf("imageFormatFromExtension(%q) = %q, %v; want %q", name, got, ok, want)
		}
	}
	for _, name := range []string{"a.bmp", "a.heic", "noext"} {
		if _, ok := imageFormatFromExtension(name); ok {
			t.Errorf("imageFormatFromExtension(%q) ok = true, want false", name)
		}
	}
}

// loadBedrockClient builds a real client from ./.env, skipping the test
// when the file is absent. Region defaults to DefaultRegion, since Nova
// embeddings aren't offered everywhere Rekognition is.
func loadBedrockClient(t *testing.T) (*bedrockruntime.Client, Env) {
	t.Helper()
	envMap, err := godotenv.Read(".env")
	if err != nil {
		t.Skipf("skipping real-Bedrock test: no modelembedding/.env (%v)", err)
	}
	raw, _ := json.Marshal(envMap)
	var env Env
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatalf("parse .env: %v", err)
	}
	if env.AwsRegion == "" {
		env.AwsRegion = DefaultRegion
	}
	loadOpts := []func(*awsv2_config.LoadOptions) error{awsv2_config.WithRegion(env.AwsRegion)}
	if env.AwsAccessKeyID != "" {
		loadOpts = append(loadOpts, awsv2_config.WithCredentialsProvider(
			awsv2_credentials.NewStaticCredentialsProvider(env.AwsAccessKeyID, env.AwsSecretAccessKey, "")))
	}
	cfg, err := awsv2_config.LoadDefaultConfig(context.Background(), loadOpts...)
	if err != nil {
		t.Fatalf("load AWS config: %v", err)
	}
	return bedrockruntime.NewFromConfig(cfg), env
}

func TestNovaEmbedder_RealBedrock(t *testing.T) {
	client, _ := loadBedrockClient(t)
	paths, _ := filepath.Glob("../cmd/input_image/*.jpeg")
	if len(paths) == 0 {
		t.Skip("skipping real-Bedrock test: no images in cmd/input_image")
	}
	img, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	embedder := NewNovaEmbedder(client)

	imageEmbedding, err := embedder.EmbedImage(ctx, img, WithPreprocess())
	if err != nil {
		t.Fatalf("EmbedImage() error = %v", err)
	}
	textEmbedding, err := embedder.EmbedText(ctx, "a photo of a man")
	if err != nil {
		t.Fatalf("EmbedText() error = %v", err)
	}
	for name, e := range map[string]Embedding{"image": imageEmbedding, "text": textEmbedding} {
		if len(e.Vector) != DefaultDimension {
			t.Errorf("%s vector has %d dimensions, want %d", name, len(e.Vector), DefaultDimension)
		}
	}
	fmt.Printf("%s vs %q cosine = %.4f\n", filepath.Base(paths[0]), "a photo of a man", cosine(imageEmbedding.Vector, textEmbedding.Vector))
}

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
