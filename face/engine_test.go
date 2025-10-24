package face

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

func TestIndexFace(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()

	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	// Sample input: imageBytes, externalImageId, collectionId
	imagePath := "3persons.png"                // Replace with your input image path
	externalImageId := "sample_3persons"       // Replace with your designated image id
	collectionId := "675c4c8cf3bf5db0b14a04ce" // Replace with your eventId

	// Read image bytes from the file
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("failed to read image file: %v", err)
	}

	// Create a context
	ctx := context.TODO()
	err = faceIndexer.IndexFace(ctx, imageBytes, externalImageId, collectionId)
	if err != nil {
		log.Fatalf("error indexing face: %v", err)
	}
}

func TestSearchFace(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()
	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	// Sample input: imageBytes, externalImageId, collectionId
	// imagePath := "photo_0-0.jpeg"                        // Replace with your input image path
	// collectionId := "new_event_66504ef59a3df2b11c092443" // Replace with your eventId
	// imagePath := "debug_13_12_2024.png"        // Replace with your input image path
	imagePath := "3persons.png"
	collectionId := "675fb398f4bf5db0b14a05cd" // Replace with your eventId
	// Read image bytes from the file
	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		log.Fatalf("failed to read image file: %v", err)
	}

	// Create a context
	ctx := context.TODO()
	faceId, matchedExternalImageIds, croppedFaceBytes, err := faceIndexer.SearchAndIndexSelfieFace(ctx, imageBytes, collectionId)
	log.Println(faceId)
	log.Println(matchedExternalImageIds)
	if err != nil {
		log.Fatalf("error searching for face: %v", err)
	}
	fmt.Println("Face ID Of Selfie: ", faceId)
	fmt.Println("Matched External Image IDs:", matchedExternalImageIds)
	// --- Save cropped face result ---
	outputFile := fmt.Sprintf("cropped_face_%s.jpeg", faceId)
	err = os.WriteFile(outputFile, croppedFaceBytes, 0644)
	if err != nil {
		log.Fatalf("failed to save cropped face image: %v", err)
	}

	fmt.Printf("✅ Cropped face saved successfully as %s\n", outputFile)
}

func TestIndexFaceWithBucket(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()
	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	// Example S3 bucket, image key, and externalImageId, collectionId
	s3Bucket := "prod.pronto.ubersnap"                                                           // replace with your bucket name
	s3Key := "event/6717bc0b7d67ae8adbf77273/media/94f05cdd-9aa8-4bef-ae6d-04e16b62a6c0/raw.jpg" // replace with your path to image
	externalImageId := "sample_image_id_with_bucket"                                             // Replace with your designated image id
	collectionId := "new_event_66504ef59a3df2b11c092443"                                         // Replace with your eventId

	// Create a context
	ctx := context.TODO()
	err := faceIndexer.IndexFaceWithBucket(ctx, s3Bucket, s3Key, externalImageId, collectionId)
	if err != nil {
		log.Fatalf("error indexing face: %v", err)
	}
}

func TestSearchFaceWithBucket(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()

	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	// Example S3 bucket, image key, and externalImageId, collectionId
	s3Bucket := "prod.pronto.ubersnap"                                                           // replace with your bucket name
	s3Key := "event/6717bc0b7d67ae8adbf77273/media/94f05cdd-9aa8-4bef-ae6d-04e16b62a6c0/raw.jpg" // replace with your path to image                      // Replace with your input image path
	collectionId := "new_event_66504ef59a3df2b11c092443"                                         // Replace with your eventId

	// Create a context
	ctx := context.TODO()

	// Call the SearchFace method using S3Object
	matchedExternalImageIds, err := faceIndexer.SearchFaceWithBucket(ctx, s3Bucket, s3Key, collectionId)
	if err != nil {
		log.Fatalf("error searching for face: %v", err)
	}

	fmt.Println("Matched External Image IDs:", matchedExternalImageIds)
}

func TestSearchFacebyFaceId(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()

	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	// Sample input: imageBytes, externalImageId, collectionId
	faceId := "f29ef5ec-37df-42dc-8bfa-cf4100b31dd6" // Replace with your input image path
	collectionId := "675c4c8cf3bf5db0b14a04ce"       // Replace with your eventId

	// Create a context
	ctx := context.TODO()
	matchedExternalImageIds, err := faceIndexer.SearchFacebyFaceId(ctx, faceId, collectionId)
	if err != nil {
		log.Fatalf("error searching for face: %v", err)
	}
	fmt.Println("Face ID Of Selfie: ", faceId)
	fmt.Println("Matched External Image IDs:", matchedExternalImageIds)
}

func TestDeleteFacebyFaceIds(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()

	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	// Sample input: imageBytes, externalImageId, collectionId
	faceIds := []string{"98c0623c-cca1-348c-8996-ae4bfdb85420"} // Replace with your faceIds
	collectionId := "675c4c8cf3bf5db0b14a04ce"                  // Replace with your eventId

	// Create a context
	ctx := context.TODO()
	unsuccessfulFaces, err := faceIndexer.DeleteFacebyFaceIds(ctx, faceIds, collectionId)
	if err != nil {
		log.Fatalf("error deleting the faces: %v", err)
	}
	fmt.Println("Unsuccessfull Deleted Face Ids: ", unsuccessfulFaces)
}

func TestListFace(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()

	// Create an instance of RekognitionFaceIndexer
	faceIndexer := &rekognitionFaceIndexer{
		client: rekognitionClient,
	}

	collectionId := "675c4c8cf3bf5db0b14a04ce" // Replace with your eventId

	// Create a context
	ctx := context.TODO()
	listFaces, err := faceIndexer.listFace(ctx, collectionId)
	if err != nil {
		log.Fatalf("error deleting the faces: %v", err)
	}
	for _, faceID := range listFaces {
		fmt.Println("FaceId:", faceID)
	}
}
