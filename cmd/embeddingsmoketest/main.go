// Command embeddingsmoketest is a real-AWS smoke test for the
// modelembedding package: it embeds every image in a folder with Amazon
// Nova (through modelembedding.EmbedImages), embeds one or more text
// queries, and prints each query's images ranked by cosine similarity,
// plus an image-to-image similarity matrix. Nothing is written to a
// database; ranking happens in memory, the way pgvector would do it.
//
// Usage:
//
//	go run ./cmd/embeddingsmoketest -env facev2/.env \
//	    -input ./cmd/input_image -query "Dwayne Johnson" -query "Jason Statham"
//
// The IAM identity in -env needs bedrock:InvokeModel on
// amazon.nova-2-multimodal-embeddings-v1:0. If -env doesn't exist, the
// standard AWS credential chain (~/.aws, AWS_PROFILE, ...) is used. Every
// image and query is one billed Bedrock call.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/HonchoLtd/aws_rekognition/modelembedding"
	awsv2_config "github.com/aws/aws-sdk-go-v2/config"
	awsv2_credentials "github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/joho/godotenv"
)

var imageExtensions = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".gif": true, ".webp": true, ".bmp": true, ".tif": true, ".tiff": true,
}

// queryFlags collects every -query occurrence.
type queryFlags []string

func (q *queryFlags) String() string     { return strings.Join(*q, ", ") }
func (q *queryFlags) Set(v string) error { *q = append(*q, v); return nil }

func main() {
	var queries queryFlags
	var (
		inputDir    = flag.String("input", "./cmd/input_image", "folder of images to embed")
		envPath     = flag.String("env", "./modelembedding/.env", "path to a .env with AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (falls back to the default AWS credential chain if missing)")
		region      = flag.String("region", modelembedding.DefaultRegion, "Bedrock region; overrides AWS_REGION from -env because Nova embeddings aren't offered in every region")
		dimension   = flag.Int("dim", modelembedding.DefaultDimension, "embedding dimension: 256, 384, 1024, or 3072")
		concurrency = flag.Int("concurrency", modelembedding.DefaultBulkConcurrency, "images embedded at once")
		preprocess  = flag.Bool("preprocess", true, "convert to JPEG, apply EXIF rotation, and shrink to -max-side before upload")
		maxSide     = flag.Int("max-side", modelembedding.DefaultMaxImageSide, "longest image side in pixels when -preprocess is on")
		topK        = flag.Int("top", 5, "results to print per query")
		matrix      = flag.Bool("matrix", true, "print the image-to-image cosine similarity matrix")
		logLevel    = flag.String("log-level", "warn", "log verbosity: debug, info, warn, or error")
	)
	flag.Var(&queries, "query", "text query to search the images with (repeatable)")
	flag.Parse()

	if len(queries) == 0 {
		queries = queryFlags{"Dwayne Johnson", "Jason Statham", "a bald man smiling"}
	}
	if err := configureLogging(*logLevel); err != nil {
		log.Fatalf("embeddingsmoketest: %v", err)
	}

	embedOpts := []modelembedding.EmbedOption{modelembedding.WithDimension(*dimension)}
	if *preprocess {
		embedOpts = append(embedOpts, modelembedding.WithPreprocessMaxSide(*maxSide))
	}

	if err := run(*inputDir, *envPath, *region, queries, *concurrency, *topK, *matrix, embedOpts); err != nil {
		log.Fatalf("embeddingsmoketest: %v", err)
	}
}

func configureLogging(level string) error {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		return fmt.Errorf("invalid -log-level %q: %w", level, err)
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})))
	return nil
}

func run(inputDir, envPath, region string, queries []string, concurrency, topK int, printMatrix bool, embedOpts []modelembedding.EmbedOption) error {
	ctx := context.Background()

	client, err := loadBedrockClient(ctx, envPath, region)
	if err != nil {
		return fmt.Errorf("load AWS client: %w", err)
	}
	embedder := modelembedding.NewStatsEmbedder(modelembedding.NewNovaEmbedder(client))

	items, err := readImages(inputDir)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("no images found in %s", inputDir)
	}
	fmt.Printf("Embedding %d image(s) from %s in %s (concurrency %d)\n", len(items), inputDir, region, concurrency)

	start := time.Now()
	results := modelembedding.EmbedImages(ctx, embedder, items,
		modelembedding.WithConcurrency(concurrency),
		modelembedding.WithEmbedOptions(embedOpts...),
		modelembedding.WithOnResult(func(r modelembedding.ImageResult) {
			if r.Err != nil {
				fmt.Printf("  FAIL %-24s %v\n", r.ID, r.Err)
				return
			}
			fmt.Printf("  ok   %-24s %d dims\n", r.ID, r.Embedding.Dimension)
		}),
	)
	imageDuration := time.Since(start)

	var indexed []modelembedding.ImageResult
	for _, r := range results {
		if r.Err == nil {
			indexed = append(indexed, r)
		}
	}
	fmt.Printf("Embedded %d/%d image(s) in %s (%.0f ms per image wall-clock)\n\n",
		len(indexed), len(items), imageDuration.Round(time.Millisecond), float64(imageDuration.Milliseconds())/float64(len(items)))
	if len(indexed) == 0 {
		return fmt.Errorf("every image failed; first error: %w", results[0].Err)
	}

	for _, query := range queries {
		queryStart := time.Now()
		queryEmbedding, err := embedder.EmbedText(ctx, query, embedOpts...)
		if err != nil {
			fmt.Printf("Query %q failed: %v\n\n", query, err)
			continue
		}
		fmt.Printf("Query %q (embedded in %s)\n", query, time.Since(queryStart).Round(time.Millisecond))
		for rank, hit := range rank(queryEmbedding.Vector, indexed, topK) {
			fmt.Printf("  #%d  %.4f  %s\n", rank+1, hit.score, hit.id)
		}
		fmt.Println()
	}

	if printMatrix {
		printSimilarityMatrix(indexed)
	}

	fmt.Printf("Bedrock calls: %s\n", embedder.Snapshot())
	return nil
}

func loadBedrockClient(ctx context.Context, envPath, region string) (*bedrockruntime.Client, error) {
	loadOpts := []func(*awsv2_config.LoadOptions) error{awsv2_config.WithRegion(region)}

	envMap, err := godotenv.Read(envPath)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		fmt.Printf("%s not found; using the default AWS credential chain\n", envPath)
	case err != nil:
		return nil, fmt.Errorf("reading env file %q: %w", envPath, err)
	default:
		raw, _ := json.Marshal(envMap)
		var env modelembedding.Env
		if err := json.Unmarshal(raw, &env); err != nil {
			return nil, err
		}
		if env.AwsAccessKeyID != "" {
			loadOpts = append(loadOpts, awsv2_config.WithCredentialsProvider(
				awsv2_credentials.NewStaticCredentialsProvider(env.AwsAccessKeyID, env.AwsSecretAccessKey, "")))
		}
	}

	cfg, err := awsv2_config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return nil, fmt.Errorf("loading AWS SDK config: %w", err)
	}
	return bedrockruntime.NewFromConfig(cfg), nil
}

func readImages(dir string) ([]modelembedding.ImageInput, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	var items []modelembedding.ImageInput
	for _, entry := range entries {
		if entry.IsDir() || !imageExtensions[strings.ToLower(filepath.Ext(entry.Name()))] {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		items = append(items, modelembedding.ImageInput{ID: entry.Name(), Image: data})
	}
	return items, nil
}

type hit struct {
	id    string
	score float64
}

func rank(query []float32, indexed []modelembedding.ImageResult, topK int) []hit {
	hits := make([]hit, 0, len(indexed))
	for _, r := range indexed {
		hits = append(hits, hit{id: r.ID, score: cosine(query, r.Embedding.Vector)})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].score > hits[j].score })
	if topK > 0 && len(hits) > topK {
		hits = hits[:topK]
	}
	return hits
}

func printSimilarityMatrix(indexed []modelembedding.ImageResult) {
	fmt.Println("Image-to-image cosine similarity")
	const width = 8
	fmt.Printf("%-24s", "")
	for i := range indexed {
		fmt.Printf("%*s", width, fmt.Sprintf("[%d]", i))
	}
	fmt.Println()
	for i, a := range indexed {
		fmt.Printf("%-24s", fmt.Sprintf("[%d] %s", i, a.ID))
		for _, b := range indexed {
			fmt.Printf("%*.3f", width, cosine(a.Embedding.Vector, b.Embedding.Vector))
		}
		fmt.Println()
	}
	fmt.Println()
}

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
