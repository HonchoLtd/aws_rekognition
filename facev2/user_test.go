package facev2

import (
	"context"
	"log"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

// --- Pure unit tests (no AWS credentials/network needed) ---

func TestExtractFaceMatches(t *testing.T) {
	faceId1 := "face-1"

	faceMatches := []types.FaceMatch{
		{
			Similarity: aws32(97.0),
			Face: &types.Face{
				FaceId: &faceId1,
				// AWS also returns ExternalImageId/BoundingBox on the Face
				// here — we deliberately don't surface them onto FaceMatch
				// (see the type's doc comment for why), so setting them here
				// or not makes no difference to the mapping.
			},
		},
		{
			// A match missing its Face should be skipped defensively.
			Similarity: aws32(96.0),
			Face:       nil,
		},
	}

	got := extractFaceMatches(faceMatches)

	want := []FaceMatch{
		{FaceId: faceId1, Similarity: 97.0},
	}

	if len(got) != len(want) {
		t.Fatalf("extractFaceMatches() returned %d matches, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("extractFaceMatches()[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// --- Integration-style tests (hit the real AWS Rekognition API) ---
//
// This mirrors engine_test.go's existing convention: it needs a populated
// .env (AWS_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY) and a collection
// that already has faces indexed (e.g. via TestIndexFace) to be meaningful.
// Replace the placeholder collectionId/faceId values with real ones from
// your own environment before running.

func TestSearchFacesByFaceId(t *testing.T) {
	rekognitionClient, _ := loadAwsRekognition()
	faceIndexer := &rekognitionFaceIndexer{client: rekognitionClient}

	collectionId := "675c4c8cf3bf5db0b14a04ce" // Replace with your collection/event id
	faceId := "replace-with-real-face-id"      // Replace with a real FaceId from IndexFace

	ctx := context.TODO()
	faceMatches, err := faceIndexer.SearchFacesByFaceId(ctx, collectionId, faceId, WithSearchFacesMatchThreshold(90), WithSearchFacesMaxResults(100))
	if err != nil {
		log.Fatalf("error searching faces by face id: %v", err)
	}
	for _, faceMatch := range faceMatches {
		log.Printf("FaceId: %s, Similarity: %f", faceMatch.FaceId, faceMatch.Similarity)
	}
}
