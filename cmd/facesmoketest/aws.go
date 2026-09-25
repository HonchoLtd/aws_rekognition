package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/HonchoLtd/aws_rekognition/facev2"
	awsv2_config "github.com/aws/aws-sdk-go-v2/config"
	awsv2_credentials "github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/joho/godotenv"
)

// loadRekognitionClient builds a *rekognition.Client from a .env file with
// the same AWS_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY shape used by
// facev2's own tests (facev2.Env).
func loadRekognitionClient(envFilePath string) (*rekognition.Client, error) {
	envMarshal, err := godotenv.Read(envFilePath)
	if err != nil {
		return nil, fmt.Errorf("reading env file %q: %w", envFilePath, err)
	}

	marshalByte, err := json.Marshal(envMarshal)
	if err != nil {
		return nil, err
	}

	var envStruct facev2.Env
	if err := json.Unmarshal(marshalByte, &envStruct); err != nil {
		return nil, err
	}
	if envStruct.AwsRegion == "" || envStruct.AwsAccessKeyID == "" || envStruct.AwsSecretAccessKey == "" {
		return nil, fmt.Errorf("%s is missing AWS_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY", envFilePath)
	}

	credentialsProvider := awsv2_credentials.NewStaticCredentialsProvider(envStruct.AwsAccessKeyID, envStruct.AwsSecretAccessKey, "")
	cfg, err := awsv2_config.LoadDefaultConfig(context.Background(),
		awsv2_config.WithRegion(envStruct.AwsRegion),
		awsv2_config.WithCredentialsProvider(credentialsProvider),
	)
	if err != nil {
		return nil, fmt.Errorf("loading AWS SDK config: %w", err)
	}

	return rekognition.NewFromConfig(cfg), nil
}
