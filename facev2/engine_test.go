package facev2

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
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
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

// TestExtractIndexedFaces is a pure unit test (no AWS credentials/network
// needed) that pins down how a raw IndexFaces response
// (resp.FaceRecords[].Face) is mapped into the exported IndexedFace metadata.
func TestExtractIndexedFaces(t *testing.T) {
	faceId1 := "11111111-1111-1111-1111-111111111111"
	faceId2 := "22222222-2222-2222-2222-222222222222"

	faceRecords := []types.FaceRecord{
		{
			Face: &types.Face{
				FaceId: &faceId1,
				BoundingBox: &types.BoundingBox{
					Width:  aws32(0.25),
					Height: aws32(0.5),
					Left:   aws32(0.1),
					Top:    aws32(0.2),
				},
			},
		},
		{
			// A second face in the same image, to make sure IndexFace can
			// surface metadata for every detected face, not just the first.
			Face: &types.Face{
				FaceId: &faceId2,
				BoundingBox: &types.BoundingBox{
					Width:  aws32(0.3),
					Height: aws32(0.4),
					Left:   aws32(0.5),
					Top:    aws32(0.6),
				},
			},
		},
		{
			// A record missing BoundingBox should be skipped defensively.
			Face: &types.Face{
				FaceId: &faceId1,
			},
		},
		{
			// A record with a nil Face should also be skipped.
			Face: nil,
		},
	}

	got := extractIndexedFaces(faceRecords)

	// None of these records set FaceDetail, so Pose/Quality/Occluded come
	// back as their zero-value (perfectly frontal, zero-quality, neutral
	// occlusion) defensive default — see toFacePose/toFaceQuality/
	// toFaceOccluded — which nets out to a fixed Score of 60 (0.4*100 pose
	// + 0.4*50 occl-neutral + 0.2*0 quality). See
	// TestExtractIndexedFaces_ParsesPoseAndQuality for the real path,
	// where FaceDetail is populated.
	want := []IndexedFace{
		{
			FaceId:      faceId1,
			BoundingBox: FaceBoundingBox{Width: 0.25, Height: 0.5, Left: 0.1, Top: 0.2},
			Score:       60,
		},
		{
			FaceId:      faceId2,
			BoundingBox: FaceBoundingBox{Width: 0.3, Height: 0.4, Left: 0.5, Top: 0.6},
			Score:       60,
		},
	}

	if len(got) != len(want) {
		t.Fatalf("extractIndexedFaces() returned %d faces, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		// Score is compared with a tolerance (see approxEqual in
		// score_test.go): the weight constants aren't exactly representable
		// in float32, so e.g. 0.6*100 comes back as 60.000004, not 60.
		g, w := got[i], want[i]
		g.Score, w.Score = 0, 0
		if g != w || !approxEqual(got[i].Score, want[i].Score) {
			t.Errorf("extractIndexedFaces()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestExtractIndexedFaces_ParsesFaceOccluded confirms
// FaceRecords[].FaceDetail.FaceOccluded is extracted onto
// IndexedFace.Occluded when present (surfaced only because IndexFace
// requests FACE_OCCLUDED explicitly — it's NOT in AWS's DEFAULT attribute
// set), and defaults to the zero-value when absent.
func TestExtractIndexedFaces_ParsesFaceOccluded(t *testing.T) {
	faceIdOccluded := "44444444-4444-4444-4444-444444444444"
	faceIdMissing := "55555555-5555-5555-5555-555555555555"

	faceRecords := []types.FaceRecord{
		{
			Face: &types.Face{
				FaceId: &faceIdOccluded,
				BoundingBox: &types.BoundingBox{
					Width: aws32(0.2), Height: aws32(0.2), Left: aws32(0.1), Top: aws32(0.1),
				},
			},
			FaceDetail: &types.FaceDetail{
				FaceOccluded: &types.FaceOccluded{
					Value:      true,
					Confidence: aws32(88.4),
				},
			},
		},
		{
			// FaceDetail present, but FaceOccluded nil (e.g. a record from an
			// older API response, or from a call that didn't ask for the
			// FACE_OCCLUDED attribute) — must fall back to the zero-value
			// (not occluded, zero confidence), not panic.
			Face: &types.Face{
				FaceId: &faceIdMissing,
				BoundingBox: &types.BoundingBox{
					Width: aws32(0.2), Height: aws32(0.2), Left: aws32(0.1), Top: aws32(0.1),
				},
			},
			FaceDetail: &types.FaceDetail{},
		},
	}

	got := extractIndexedFaces(faceRecords)
	if len(got) != 2 {
		t.Fatalf("extractIndexedFaces() returned %d faces, want 2: %+v", len(got), got)
	}
	if got[0].Occluded != (FaceOccluded{Value: true, Confidence: 88.4}) {
		t.Errorf("Occluded[0] = %+v, want {true, 88.4}", got[0].Occluded)
	}
	if got[1].Occluded != (FaceOccluded{}) {
		t.Errorf("Occluded[1] = %+v, want zero-value (nil FaceOccluded)", got[1].Occluded)
	}
}

// TestExtractIndexedFaces_ParsesPoseAndQuality confirms
// FaceRecords[].FaceDetail's Pose/Quality are extracted and folded into the
// same Score formula ComputeFaceScore exposes directly (see score_test.go
// for that formula's own dedicated cases, including how the FaceOccluded
// third input contributes to the score).
func TestExtractIndexedFaces_ParsesPoseAndQuality(t *testing.T) {
	faceId := "33333333-3333-3333-3333-333333333333"

	faceRecords := []types.FaceRecord{
		{
			Face: &types.Face{
				FaceId: &faceId,
				BoundingBox: &types.BoundingBox{
					Width: aws32(0.2), Height: aws32(0.2), Left: aws32(0.1), Top: aws32(0.1),
				},
			},
			FaceDetail: &types.FaceDetail{
				Pose: &types.Pose{
					Yaw:   aws32(10),
					Pitch: aws32(-5),
					Roll:  aws32(90), // extreme, but must not affect the score at all
				},
				Quality: &types.ImageQuality{
					Brightness: aws32(80),
					Sharpness:  aws32(90),
				},
			},
		},
	}

	got := extractIndexedFaces(faceRecords)
	if len(got) != 1 {
		t.Fatalf("extractIndexedFaces() returned %d faces, want 1: %+v", len(got), got)
	}

	wantPose := FacePose{Yaw: 10, Pitch: -5, Roll: 90}
	wantQuality := FaceQuality{Brightness: 80, Sharpness: 90}
	if got[0].Pose != wantPose {
		t.Errorf("Pose = %+v, want %+v", got[0].Pose, wantPose)
	}
	if got[0].Quality != wantQuality {
		t.Errorf("Quality = %+v, want %+v", got[0].Quality, wantQuality)
	}

	// FaceDetail didn't include a FaceOccluded, so extractIndexedFaces uses
	// the zero-value (Value=false, Confidence=0 → neutral 50) — verify the
	// end-to-end Score matches ComputeFaceScore called with that same input.
	wantScore := ComputeFaceScore(wantPose, wantQuality, FaceOccluded{})
	if got[0].Score != wantScore {
		t.Errorf("Score = %v, want %v (ComputeFaceScore(Pose, Quality, FaceOccluded{}))", got[0].Score, wantScore)
	}
}

// TestExtractIndexedFaces_Empty ensures an empty FaceRecords slice yields an
// empty (non-nil) result rather than nil, so callers can safely range over it.
func TestExtractIndexedFaces_Empty(t *testing.T) {
	got := extractIndexedFaces(nil)
	if got == nil {
		t.Fatalf("extractIndexedFaces(nil) = nil, want empty non-nil slice")
	}
	if len(got) != 0 {
		t.Fatalf("extractIndexedFaces(nil) = %+v, want empty slice", got)
	}
}

func aws32(v float32) *float32 {
	return &v
}

// The tests below hit the real AWS Rekognition API and therefore require a
// populated .env (AWS_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY) plus,
// for TestIndexFace, a local sample image. They mirror face/engine_test.go's
// existing integration-test conventions.

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
	indexedFaces, err := faceIndexer.IndexFace(ctx, imageBytes, externalImageId, collectionId)
	if err != nil {
		log.Fatalf("error indexing face: %v", err)
	}
	if len(indexedFaces) == 0 {
		log.Fatalf("expected at least one indexed face, got none")
	}
	for _, indexedFace := range indexedFaces {
		if indexedFace.FaceId == "" {
			t.Errorf("indexed face is missing FaceId: %+v", indexedFace)
		}
		fmt.Printf("FaceId: %s, BoundingBox: %+v\n", indexedFace.FaceId, indexedFace.BoundingBox)
	}
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
	indexedFaces, err := faceIndexer.IndexFaceWithBucket(ctx, s3Bucket, s3Key, externalImageId, collectionId)
	if err != nil {
		log.Fatalf("error indexing face: %v", err)
	}
	if len(indexedFaces) == 0 {
		log.Fatalf("expected at least one indexed face, got none")
	}
	for _, indexedFace := range indexedFaces {
		if indexedFace.FaceId == "" {
			t.Errorf("indexed face is missing FaceId: %+v", indexedFace)
		}
		fmt.Printf("FaceId: %s, BoundingBox: %+v\n", indexedFace.FaceId, indexedFace.BoundingBox)
	}
}
