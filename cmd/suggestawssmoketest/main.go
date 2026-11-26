// Command suggestawssmoketest makes one real call of each suggestaws
// operation (design doc section 12.1, "A: smoke"), for well under $0.001:
//
//  1. ProposeTerms: up to 4 photos from -input as one group, temperature 0.
//  2. Screen: Lite's terms plus -probe (a term the guardrail should block).
//  3. EmbedText: the first allowed term, IMAGE_RETRIEVAL.
//
// Usage:
//
//	go run ./cmd/suggestawssmoketest -env facev2/.env -input ./cmd/input_image -guardrail-id <id>
//
// GUARDRAIL_ID / GUARDRAIL_VERSION and the other suggestaws settings are
// also read from the environment and from -env. Without a guardrail the
// Screen step is skipped and reported, the same way the suggestion
// service fails closed.
package main

import (
	"context"
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
	"github.com/HonchoLtd/aws_rekognition/suggestaws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/joho/godotenv"
)

// Unit prices from design doc section 11.1 (unconfirmed third-party
// figures; good enough to sanity-check a smoke run).
const (
	liteInputPerToken  = 0.33 / 1e6
	liteOutputPerToken = 2.75 / 1e6
	guardrailPerCall   = 0.0003
	embedPerTerm       = 4 * 0.135 / 1e6
)

var imageExtensions = map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".webp": true, ".gif": true, ".bmp": true, ".tif": true, ".tiff": true}

func main() {
	var (
		envPath          = flag.String("env", "./facev2/.env", "path to a .env with AWS credentials and optional suggestaws settings")
		inputDir         = flag.String("input", "./cmd/input_image", "folder of photos; the first 4 (by name) form group 1")
		guardrailID      = flag.String("guardrail-id", "", "overrides GUARDRAIL_ID")
		guardrailVersion = flag.String("guardrail-version", "", "overrides GUARDRAIL_VERSION")
		probe            = flag.String("probe", "drunk guests", "a term the guardrail should block, screened with Lite's terms")
		logLevel         = flag.String("log-level", "info", "debug, info, warn or error")
	)
	flag.Parse()

	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(*logLevel)); err != nil {
		log.Fatalf("suggestawssmoketest: invalid -log-level: %v", err)
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl})))

	if err := run(*envPath, *inputDir, *guardrailID, *guardrailVersion, *probe); err != nil {
		log.Fatalf("suggestawssmoketest: %v", err)
	}
}

func run(envPath, inputDir, guardrailID, guardrailVersion, probe string) error {
	// The design's cycle deadline (GENERATION_TIMEOUT).
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	credOpts, err := loadEnv(envPath)
	if err != nil {
		return err
	}
	cfg, err := suggestaws.ConfigFromEnv()
	if err != nil {
		return err
	}
	if guardrailID != "" {
		cfg.GuardrailID = guardrailID
	}
	if guardrailVersion != "" {
		cfg.GuardrailVersion = guardrailVersion
	}
	awsCfg, err := suggestaws.NewAWSConfig(ctx, cfg.Region, credOpts...)
	if err != nil {
		return fmt.Errorf("load AWS config: %w", err)
	}
	client, err := suggestaws.NewFromAWSConfig(awsCfg, cfg)
	if err != nil {
		return err
	}

	jpegs, names, err := loadPhotos(inputDir, 4)
	if err != nil {
		return err
	}
	fmt.Printf("Region %s · Lite %s · embed %s (%d-d)\n\n", cfg.Region, cfg.LiteModelID, cfg.EmbedModelID, cfg.Dimension)

	// 1. ProposeTerms
	start := time.Now()
	proposed, err := client.ProposeTerms(ctx, suggestaws.ProposeRequest{
		Groups: []suggestaws.Group{{Number: 1, Size: len(jpegs), JPEGs: jpegs}},
	})
	if err != nil {
		return fmt.Errorf("ProposeTerms: %w", err)
	}
	terms := proposed.Terms[1]
	fmt.Printf("1. ProposeTerms (%s) on %s\n   terms: %q\n   tokens: %d in / %d out\n\n",
		time.Since(start).Round(time.Millisecond), strings.Join(names, ", "), terms, proposed.InputTokens, proposed.OutputTokens)
	cost := float64(proposed.InputTokens)*liteInputPerToken + float64(proposed.OutputTokens)*liteOutputPerToken

	// 2. Screen
	allowed := terms
	start = time.Now()
	screened, err := client.Screen(ctx, append(append([]string{}, terms...), probe))
	switch {
	case errors.Is(err, suggestaws.ErrGuardrailNotConfigured):
		fmt.Println("2. Screen skipped: no GUARDRAIL_ID (run ./cmd/suggestguardrail first). The service would turn suggestions off.")
		fmt.Println()
	case err != nil:
		return fmt.Errorf("Screen: %w", err)
	default:
		fmt.Printf("2. Screen (%s, %s@%s)\n   allowed: %q\n", time.Since(start).Round(time.Millisecond), cfg.GuardrailID, cfg.GuardrailVersion, screened.Allowed)
		for _, t := range sortedKeys(screened.Blocked) {
			fmt.Printf("   blocked: %q (%s)\n", t, screened.Blocked[t])
		}
		if _, ok := screened.Blocked[probe]; !ok {
			fmt.Printf("   WARNING: probe %q was not blocked; check the guardrail policy\n", probe)
		}
		fmt.Printf("   calls: %d\n\n", screened.Calls)
		cost += float64(screened.Calls) * guardrailPerCall
		allowed = screened.Allowed
	}

	// 3. EmbedText
	if len(allowed) == 0 || allowed[0] == probe {
		fmt.Println("3. EmbedText skipped: no allowed term from Lite")
	} else {
		start = time.Now()
		vec, err := client.EmbedText(ctx, allowed[0])
		if err != nil {
			return fmt.Errorf("EmbedText: %w", err)
		}
		fmt.Printf("3. EmbedText (%s) %q -> %d-d vector, norm %.4f, first %v\n\n",
			time.Since(start).Round(time.Millisecond), allowed[0], len(vec), norm(vec), vec[:min(4, len(vec))])
		cost += embedPerTerm
	}

	fmt.Printf("Estimated cost of this run: $%.6f\n", cost)
	return nil
}

// loadEnv exports every key of envPath that is not already set (so
// GUARDRAIL_ID etc. reach ConfigFromEnv) and returns static credentials
// when it has AWS_ACCESS_KEY_ID.
func loadEnv(envPath string) ([]func(*awsconfig.LoadOptions) error, error) {
	env, err := godotenv.Read(envPath)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		fmt.Printf("%s not found; using the process environment and default AWS credential chain\n", envPath)
		return nil, nil
	case err != nil:
		return nil, fmt.Errorf("reading env file %q: %w", envPath, err)
	}
	for k, v := range env {
		if _, set := os.LookupEnv(k); !set && k != "AWS_REGION" {
			os.Setenv(k, v)
		}
	}
	if env["AWS_ACCESS_KEY_ID"] == "" {
		return nil, nil
	}
	return []func(*awsconfig.LoadOptions) error{awsconfig.WithCredentialsProvider(
		credentials.NewStaticCredentialsProvider(env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"], env["AWS_SESSION_TOKEN"]))}, nil
}

// loadPhotos reads up to n images from dir (sorted by name) as 384 px
// JPEGs, the size System B sends.
func loadPhotos(dir string, n int) ([][]byte, []string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	var jpegs [][]byte
	var names []string
	for _, e := range entries {
		if len(jpegs) == n {
			break
		}
		if e.IsDir() || !imageExtensions[strings.ToLower(filepath.Ext(e.Name()))] {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, nil, err
		}
		jpeg, err := modelembedding.PreprocessImage(raw, 384)
		if err != nil {
			fmt.Printf("skipping %s: %v\n", e.Name(), err)
			continue
		}
		jpegs = append(jpegs, jpeg)
		names = append(names, e.Name())
	}
	if len(jpegs) == 0 {
		return nil, nil, fmt.Errorf("no usable images in %s", dir)
	}
	return jpegs, names, nil
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}
