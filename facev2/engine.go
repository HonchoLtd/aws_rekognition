// Package facev2 is the v2 face-indexing package. It exists alongside the
// original `face` package so that existing callers of `face.Face.IndexFace`
// (which returns just an error) are left untouched, while new callers can
// opt into the richer response by depending on facev2 instead.
//
// The main difference from `face`: IndexFace / IndexFaceWithBucket here
// return the metadata AWS Rekognition's IndexFaces API hands back for every
// indexed face — specifically FaceId and BoundingBox — rather than
// discarding it.
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
package facev2

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

// FaceBoundingBox mirrors the relevant fields of the AWS Rekognition
// BoundingBox type returned as part of IndexFaces' FaceRecords[].Face.
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_BoundingBox.html
type FaceBoundingBox struct {
	Width  float32 `json:"width"`
	Height float32 `json:"height"`
	Left   float32 `json:"left"`
	Top    float32 `json:"top"`
}

// IndexedFace holds the metadata we care about from a single entry of
// IndexFaces' response FaceRecords[].Face.
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
type IndexedFace struct {
	FaceId      string          `json:"face_id"`
	BoundingBox FaceBoundingBox `json:"bounding_box"`
}

type Face interface {
	IndexFace(ctx context.Context, image []byte, imageID string, eventID string) ([]IndexedFace, error)
	IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, imageID string, eventID string) ([]IndexedFace, error)
}

type rekognitionFaceIndexer struct {
	client *rekognition.Client
}

func NewRekognitionFaceIndexer(client *rekognition.Client) Face {
	return &rekognitionFaceIndexer{client: client}
}

// createCollectionIfNotExists creates the collection when it doesn't exist yet.
func (r *rekognitionFaceIndexer) createCollectionIfNotExists(ctx context.Context, rekognitionClient *rekognition.Client, collectionId string) error {
	// Check if the collection exists
	_, err := rekognitionClient.DescribeCollection(ctx, &rekognition.DescribeCollectionInput{
		CollectionId: aws.String(collectionId),
	})

	// If the collection does not exist, create it
	if err != nil {
		fmt.Printf("Collection %s does not exist. Creating a new collection...\n", collectionId)
		_, err := rekognitionClient.CreateCollection(ctx, &rekognition.CreateCollectionInput{
			CollectionId: aws.String(collectionId),
		})
		if err != nil {
			var rae *types.ResourceAlreadyExistsException
			if errors.As(err, &rae) {
				log.Printf("Collection %s already exists, skip error while failed create it.\n", collectionId)
				return nil
			} else {
				return fmt.Errorf("eror is not ResourceAlreadyExistsException failed to create collection: %v", err)
			}
		}
		fmt.Printf("Collection %s created successfully.\n", collectionId)
	}

	return nil
}

// IndexFace indexes the face(s) found in imageBytes into collectionId, and
// returns the FaceId/BoundingBox metadata AWS Rekognition assigned to each
// indexed face.
func (r *rekognitionFaceIndexer) IndexFace(ctx context.Context, imageBytes []byte, externalImageId string, collectionId string) ([]IndexedFace, error) {

	// First, ensure the collection exists
	err := r.createCollectionIfNotExists(ctx, r.client, collectionId)
	if err != nil {
		return nil, fmt.Errorf("failed to ensure collection exists: %v", err)
	}

	// Prepare the input for the IndexFaces API
	input := &rekognition.IndexFacesInput{
		CollectionId:    aws.String(collectionId),
		Image:           &types.Image{Bytes: imageBytes},
		ExternalImageId: aws.String(externalImageId),
	}

	log.Printf("Delay before Index faces by 0.5 second")
	time.Sleep(500 * time.Millisecond)

	// Call the IndexFaces API
	resp, err := r.client.IndexFaces(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to index face: %v", err)
	}

	// Build the response metadata from FaceRecords[].Face, per the IndexFaces
	// API contract: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
	indexedFaces := extractIndexedFaces(resp.FaceRecords)

	// Output the result
	fmt.Printf("Successfully indexed face for ExternalImageId: %s\n", externalImageId)
	for _, indexedFace := range indexedFaces {
		fmt.Printf("FaceId: %s, BoundingBox: %+v\n", indexedFace.FaceId, indexedFace.BoundingBox)
	}

	return indexedFaces, nil
}

// IndexFaceWithBucket is the S3-backed counterpart of IndexFace: it indexes
// the face(s) found in the s3Bucket/s3Key object instead of raw bytes, and
// returns the same FaceId/BoundingBox metadata per indexed face.
func (r *rekognitionFaceIndexer) IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, externalImageId string, collectionId string) ([]IndexedFace, error) {
	// First, ensure the collection exists
	err := r.createCollectionIfNotExists(ctx, r.client, collectionId)
	if err != nil {
		return nil, fmt.Errorf("failed to ensure collection exists: %v", err)
	}

	// Prepare the input for the IndexFaces API using S3Object
	input := &rekognition.IndexFacesInput{
		CollectionId: aws.String(collectionId),
		Image: &types.Image{
			S3Object: &types.S3Object{
				Bucket: aws.String(s3Bucket),
				Name:   aws.String(s3Key),
			},
		},
		ExternalImageId: aws.String(externalImageId),
	}

	// Call the IndexFaces API
	resp, err := r.client.IndexFaces(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to index face: %v", err)
	}

	// Build the response metadata from FaceRecords[].Face, per the IndexFaces
	// API contract: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
	indexedFaces := extractIndexedFaces(resp.FaceRecords)

	// Output the result
	fmt.Printf("Successfully indexed face for ExternalImageId: %s\n", externalImageId)
	for _, indexedFace := range indexedFaces {
		fmt.Printf("FaceId: %s, BoundingBox: %+v\n", indexedFace.FaceId, indexedFace.BoundingBox)
	}

	return indexedFaces, nil
}

// extractIndexedFaces maps IndexFaces' FaceRecords (resp.FaceRecords[].Face)
// into the FaceId/BoundingBox metadata we expose to callers. FaceRecords with
// a nil Face or BoundingBox are skipped defensively, matching the fact that
// most Face fields are documented as optional in the API reference.
func extractIndexedFaces(faceRecords []types.FaceRecord) []IndexedFace {
	indexedFaces := make([]IndexedFace, 0, len(faceRecords))
	for _, faceRecord := range faceRecords {
		if faceRecord.Face == nil || faceRecord.Face.FaceId == nil || faceRecord.Face.BoundingBox == nil {
			continue
		}
		bbox := faceRecord.Face.BoundingBox
		indexedFaces = append(indexedFaces, IndexedFace{
			FaceId: *faceRecord.Face.FaceId,
			BoundingBox: FaceBoundingBox{
				Width:  aws.ToFloat32(bbox.Width),
				Height: aws.ToFloat32(bbox.Height),
				Left:   aws.ToFloat32(bbox.Left),
				Top:    aws.ToFloat32(bbox.Top),
			},
		})
	}
	return indexedFaces
}
