package face

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
	"github.com/google/uuid"
	"github.com/samber/lo"
)

type Face interface {
	IndexFace(ctx context.Context, image []byte, imageID string, eventID string) error
	SearchAndIndexSelfieFace(ctx context.Context, imageSelfie []byte, eventID string) (string, []string, error)
	SearchFacebyFaceId(ctx context.Context, imageSelfieId string, eventID string) ([]string, error)
	IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, imageID string, eventID string) error
	SearchFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, collectionId string) ([]string, error)
	DeleteFacebyFaceIds(ctx context.Context, faceIds []string, collectionId string) ([]string, error)
	listFace(ctx context.Context, collectionId string) ([]string, error)
}

type rekognitionFaceIndexer struct {
	client *rekognition.Client
}

func NewRekognitionFaceIndexer(client *rekognition.Client) Face {
	return &rekognitionFaceIndexer{client: client}
}

// Function to create a collection if it doesn't exist
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

// IndexFace Implementation of IndexFace method in Face interface
func (r *rekognitionFaceIndexer) IndexFace(ctx context.Context, imageBytes []byte, externalImageId string, collectionId string) error {

	// First, ensure the collection exists
	err := r.createCollectionIfNotExists(ctx, r.client, collectionId)
	if err != nil {
		return fmt.Errorf("failed to ensure collection exists: %v", err)
	}

	// Prepare the input for the IndexFaces API
	input := &rekognition.IndexFacesInput{
		CollectionId:    aws.String(collectionId),
		Image:           &types.Image{Bytes: imageBytes},
		ExternalImageId: aws.String(externalImageId),
	}

	log.Printf("Delay before Index faces by 3 second")
	time.Sleep(3 * time.Second)

	// Call the IndexFaces API
	resp, err := r.client.IndexFaces(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to index face: %v", err)
	}

	// Output the result
	fmt.Printf("Successfully indexed face for ExternalImageId: %s\n", externalImageId)
	for _, faceRecord := range resp.FaceRecords {
		fmt.Printf("FaceId: %s, Confidence: %f\n", *faceRecord.Face.FaceId, *faceRecord.Face.Confidence)
	}

	return nil
}

// SearchFace Implementation of SearchFace method in Face interface
func (r *rekognitionFaceIndexer) SearchAndIndexSelfieFace(ctx context.Context, imageSelfie []byte, collectionId string) (string, []string, error) {

	// Generate a random UUID as ExternalImageId
	externalImageId := fmt.Sprintf("%s_%s", uuid.New().String(), collectionId)

	// Index the input selfie
	inputIndexSelfie := &rekognition.IndexFacesInput{
		CollectionId:    aws.String(collectionId),
		Image:           &types.Image{Bytes: imageSelfie},
		ExternalImageId: aws.String(externalImageId),
	}
	// Call the IndexFaces API
	resp, err := r.client.IndexFaces(ctx, inputIndexSelfie)
	if err != nil {
		return "", nil, fmt.Errorf("search face failed: error when try to index selfie face: %v", err)
	}

	// Check if a face was detected and indexed
	if len(resp.FaceRecords) == 0 {
		return "", nil, fmt.Errorf("search face failed: no face detected in the image")
	}

	// Get the FaceId of the first indexed face
	faceId := *resp.FaceRecords[0].Face.FaceId
	fmt.Printf("Successfully Indexed FaceId: %s, ExternalImageId: %s\n", faceId, externalImageId)

	externalImageIdResult, err := r.SearchFacebyFaceId(ctx, faceId, collectionId)
	if err != nil {
		return faceId, nil, fmt.Errorf("search Face Failed: error when try to find selfie in collection: %v", err)
	}
	return faceId, externalImageIdResult, nil
}

// IndexFaceWithBucket Implementation of IndexFace method for S3 image input
func (r *rekognitionFaceIndexer) IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, externalImageId string, collectionId string) error {
	// First, ensure the collection exists
	err := r.createCollectionIfNotExists(ctx, r.client, collectionId)
	if err != nil {
		return fmt.Errorf("failed to ensure collection exists: %v", err)
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
		return fmt.Errorf("failed to index face: %v", err)
	}

	// Output the result
	fmt.Printf("Successfully indexed face for ExternalImageId: %s\n", externalImageId)
	for _, faceRecord := range resp.FaceRecords {
		fmt.Printf("FaceId: %s, Confidence: %f\n", *faceRecord.Face.FaceId, *faceRecord.Face.Confidence)
	}

	return nil
}

// SearchFaceWithBucket Implementation of SearchFace method for S3 image input
func (r *rekognitionFaceIndexer) SearchFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, collectionId string) ([]string, error) {
	// Prepare the input for the SearchFacesByImage API using S3Object
	input := &rekognition.SearchFacesByImageInput{
		CollectionId: aws.String(collectionId),
		Image: &types.Image{
			S3Object: &types.S3Object{
				Bucket: aws.String(s3Bucket),
				Name:   aws.String(s3Key),
			},
		},
	}

	// Call the SearchFacesByImage API
	resp, err := r.client.SearchFacesByImage(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to search face by image: %v", err)
	}

	// Use a slice to store ExternalImageIds
	var externalImageIds []string

	for _, match := range resp.FaceMatches {
		if match.Face.ExternalImageId != nil {
			externalImageIds = append(externalImageIds, *match.Face.ExternalImageId)
		}
	}

	// Use lo.Uniq to filter out duplicate ExternalImageIds
	uniqueExternalImageIds := lo.Uniq(externalImageIds)

	return uniqueExternalImageIds, nil
}

func (r *rekognitionFaceIndexer) SearchFacebyFaceId(ctx context.Context, imageSelfieId string, collectionId string) ([]string, error) {
	// Prepare the input for the SearchFacesByImage API
	input := &rekognition.SearchFacesInput{
		CollectionId: aws.String(collectionId),  // The collection where the face is stored
		FaceId:       aws.String(imageSelfieId), // The FaceId we want to search for
	}
	log.Printf("Try to find this generated face id: %s", imageSelfieId)
	log.Printf("Try to find this collection id: %s", collectionId)
	// Try to find the collection exists or not
	inputCheckCollection := &rekognition.DescribeCollectionInput{
		CollectionId: aws.String(*input.CollectionId),
	}
	resp_collection, err := r.client.DescribeCollection(ctx, inputCheckCollection)
	if err != nil {
		log.Printf("Error collection : %v", err)
	}
	json_resp_col, _ := json.Marshal(resp_collection)
	log.Printf("Try to check this collection : %s", string(json_resp_col))
	log.Printf("Input payload: %s %s", *input.CollectionId, *input.FaceId)

	// start := time.Now()
	// // Try to list all the faces in collection
	// inputListFacesCollection := &rekognition.ListFacesInput{
	// 	CollectionId: aws.String(*input.CollectionId),
	// }
	// resp_list_faces, err := r.client.ListFaces(ctx, inputListFacesCollection)
	// if err != nil {
	// 	log.Printf("Error List Faces : %v", err)
	// }
	// json_resp_list_faces, _ := json.Marshal(resp_list_faces)
	// log.Printf("Try to list all faces collection : %s", string(json_resp_list_faces))
	// elapsed := time.Since(start)
	// log.Printf("Binomial took %s", elapsed)

	log.Printf("Delay before search faces by 3 second")
	time.Sleep(3 * time.Second)

	log.Printf("Input payload: %s %s", *input.CollectionId, *input.FaceId)
	// Call the SearchFacesByImage API
	resp, err := r.client.SearchFaces(ctx, input)
	if err != nil {
		log.Printf("error line: %v", err)
		// Check if the error is an InvalidParameterException (no faces in the image)
		var invalidParamErr *types.InvalidParameterException
		if errors.As(err, &invalidParamErr) {
			// Handle the case where no faces were detected in the image
			log.Printf("Search Face Error: Invalid Parameter")
			return nil, fmt.Errorf("found this error when search face by id: %v", err)
		}
		return nil, fmt.Errorf("failed to search face by id, [Invalid, please try again]: %v", err)
	}

	// Use a map to ensure uniqueness of ExternalImageId
	var externalImageIds []string
	for _, match := range resp.FaceMatches {
		if match.Face.ExternalImageId != nil {
			externalImageIds = append(externalImageIds, *match.Face.ExternalImageId)
		}
	}

	// Use lo.Uniq to filter out duplicate ExternalImageIds
	uniqueExternalImageIds := lo.Uniq(externalImageIds)

	return uniqueExternalImageIds, nil
}

func (r *rekognitionFaceIndexer) DeleteFacebyFaceIds(ctx context.Context, faceIds []string, collectionId string) ([]string, error) {
	// Prepare the input for the DeleteFace API
	input := &rekognition.DeleteFacesInput{
		CollectionId: aws.String(collectionId), // The collection where the face is stored
		FaceIds:      faceIds,                  // The FaceId we want to search for
	}
	// Call the SearchFacesByImage API
	resp, err := r.client.DeleteFaces(ctx, input)
	if err != nil {
		log.Printf("error line: %v", err)
		return nil, fmt.Errorf("Failed to delete faces: %v", err)
	}

	unsuccessfulFaces := make([]string, 0)
	for _, unsuccessFace := range resp.UnsuccessfulFaceDeletions {
		fmt.Println("FOUND FACES", unsuccessFace.FaceId)
		unsuccessfulFaces = append(unsuccessfulFaces, *unsuccessFace.FaceId)
		log.Printf("FaceId : %s failed to be deleted because: %s", *unsuccessFace.FaceId, *&unsuccessFace.Reasons)
	}
	return unsuccessfulFaces, nil
}

func (r *rekognitionFaceIndexer) listFace(ctx context.Context, collectionId string) ([]string, error) {
	// Prepare the input for the DeleteFace API
	input := &rekognition.ListFacesInput{
		CollectionId: aws.String(collectionId), // The collection where the face is stored              // The FaceId we want to search for
	}
	// Call the SearchFacesByImage API
	resp, err := r.client.ListFaces(ctx, input)
	if err != nil {
		log.Printf("error line: %v", err)
		return nil, fmt.Errorf("Failed to delete faces: %v", err)
	}

	facesResult := make([]string, 0)
	for _, face := range resp.Faces {
		facesResult = append(facesResult, *face.FaceId)
	}
	return facesResult, nil

}
