package text_detection

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"testing"

	awsv2_config "github.com/aws/aws-sdk-go-v2/config"
	awsv2_credentials "github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/joho/godotenv"
)

func loadAwsRekognition() (*rekognition.Client, error) {
	envFilePath := ".env" // Provide the env file path

	envMarshal, err := godotenv.Read(envFilePath)
	if err != nil {
		fmt.Printf("Found this error: %v", err)
		return nil, err
	}

	marshalByte, err := json.Marshal(envMarshal)
	if err != nil {
		return nil, err
	}

	var envStruct Env
	if err := json.Unmarshal(marshalByte, &envStruct); err != nil {
		return nil, err
	}

	// AWS SDK v2
	awsV2Credentials := awsv2_credentials.NewStaticCredentialsProvider(envStruct.AwsAccessKeyID, envStruct.AwsSecretAccessKey, "")
	awsV2Cfg, err := awsv2_config.LoadDefaultConfig(context.Background(),
		awsv2_config.WithRegion(envStruct.AwsRegion),
		awsv2_config.WithCredentialsProvider(awsV2Credentials),
	)
	if err != nil {
		log.Fatalf("unable to load SDK config, %v", err)
	}
	rekognitionClient := rekognition.NewFromConfig(awsV2Cfg)
	return rekognitionClient, nil

}

func TestDetectText(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()

	// Create an instance of RekognitionFaceIndexer
	textDetector := &rekognitionTextDetector{
		client: rekognitionClient,
	}

	// Sample input: imageBytes, externalImageId, collectionId
	imagePath := "0002992_running_600.png" // Replace with your input image path

	// Read image bytes from the file
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("failed to read image file: %v", err)
	}

	// Create a context
	ctx := context.TODO()
	result, err := textDetector.DetectText(ctx, imageBytes)
	if err != nil {
		log.Fatalf("error indexing face: %v", err)
	}
	log.Printf("Detected Texts: %v", result)
}
