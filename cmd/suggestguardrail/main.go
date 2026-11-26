// Command suggestguardrail creates or updates the Bedrock Guardrail that
// suggestaws.Screen uses (design doc section 3.3). Run it once per account
// and region, and again whenever suggestaws.GuardrailPolicy changes.
//
// Usage:
//
//	go run ./cmd/suggestguardrail -env facev2/.env
//	go run ./cmd/suggestguardrail -env facev2/.env -publish   # also snapshot a numbered version
//
// The IAM identity needs bedrock:ListGuardrails, bedrock:CreateGuardrail
// and bedrock:UpdateGuardrail (plus bedrock:CreateGuardrailVersion with
// -publish). Set the printed GUARDRAIL_ID / GUARDRAIL_VERSION in the
// suggestion service's environment.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"time"

	"github.com/HonchoLtd/aws_rekognition/suggestaws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/joho/godotenv"
)

func main() {
	var (
		envPath  = flag.String("env", "./facev2/.env", "path to a .env with AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY (falls back to the default AWS credential chain if missing)")
		region   = flag.String("region", suggestaws.DefaultRegion, "Bedrock region; must be the region the suggestion service calls")
		name     = flag.String("name", suggestaws.DefaultGuardrailName, "guardrail name; an existing guardrail with this name is updated")
		publish  = flag.Bool("publish", false, "after updating the DRAFT, create a numbered version to pin in production")
		describe = flag.String("version-description", "", "description for the version created by -publish")
	)
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	opts, err := credentialOptions(*envPath)
	if err != nil {
		log.Fatalf("suggestguardrail: %v", err)
	}
	awsCfg, err := suggestaws.NewAWSConfig(ctx, *region, opts...)
	if err != nil {
		log.Fatalf("suggestguardrail: load AWS config: %v", err)
	}
	admin := bedrock.NewFromConfig(awsCfg)

	ref, err := suggestaws.EnsureGuardrail(ctx, admin, *name)
	if err != nil {
		log.Fatalf("suggestguardrail: %v", err)
	}
	action := "updated"
	if ref.Created {
		action = "created"
	}
	fmt.Printf("Guardrail %q %s in %s\n  id:  %s\n  arn: %s\n", *name, action, *region, ref.ID, ref.ARN)

	version := ref.Version
	if *publish {
		version, err = suggestaws.PublishGuardrailVersion(ctx, admin, ref.ID, *describe)
		if err != nil {
			log.Fatalf("suggestguardrail: %v", err)
		}
		fmt.Printf("  published version %s\n", version)
	}
	fmt.Printf("\nSet in the suggestion service:\n  GUARDRAIL_ID=%s\n  GUARDRAIL_VERSION=%s\n", ref.ID, version)
}

// credentialOptions returns static credentials from envPath when it sets
// AWS_ACCESS_KEY_ID, and nothing (default credential chain) otherwise.
func credentialOptions(envPath string) ([]func(*awsconfig.LoadOptions) error, error) {
	env, err := godotenv.Read(envPath)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		fmt.Printf("%s not found; using the default AWS credential chain\n", envPath)
		return nil, nil
	case err != nil:
		return nil, fmt.Errorf("reading env file %q: %w", envPath, err)
	}
	if env["AWS_ACCESS_KEY_ID"] == "" {
		return nil, nil
	}
	return []func(*awsconfig.LoadOptions) error{awsconfig.WithCredentialsProvider(
		credentials.NewStaticCredentialsProvider(env["AWS_ACCESS_KEY_ID"], env["AWS_SECRET_ACCESS_KEY"], env["AWS_SESSION_TOKEN"]))}, nil
}
