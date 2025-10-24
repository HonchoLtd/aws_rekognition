package face

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png" // enable PNG decoding as well
	"log"
	"math"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
	"github.com/google/uuid"
	"github.com/samber/lo"
)

type Face interface {
	IndexFace(ctx context.Context, image []byte, imageID string, eventID string) error
	SearchAndIndexSelfieFace(ctx context.Context, imageSelfie []byte, eventID string) (string, []string, []byte, error)
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

	log.Printf("Delay before Index faces by 0.5 second")
	time.Sleep(500 * time.Millisecond)

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

func rotateAccordingToOrientation(src image.Image, oc types.OrientationCorrection) image.Image {
	switch string(oc) {
	case "ROTATE_90":
		b := src.Bounds()
		dst := image.NewRGBA(image.Rect(0, 0, b.Dy(), b.Dx()))
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				dst.Set(b.Max.Y-1-y, x, src.At(x, y))
			}
		}
		return dst
	case "ROTATE_180":
		b := src.Bounds()
		dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				dst.Set(b.Max.X-1-x, b.Max.Y-1-y, src.At(x, y))
			}
		}
		return dst
	case "ROTATE_270":
		b := src.Bounds()
		dst := image.NewRGBA(image.Rect(0, 0, b.Dy(), b.Dx()))
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				dst.Set(y, b.Max.X-1-x, src.At(x, y))
			}
		}
		return dst
	default: // "ROTATE_0" or empty
		return src
	}
}

// clamp clamps v between [lo, hi]
func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// cropWithBoundingBoxScaled expands the Rekognition bbox by `scale` (e.g., 1.8)
// around its center, then clamps to the image bounds.
func cropWithBoundingBoxScaled(img image.Image, bbox types.BoundingBox, scale float64) (image.Image, error) {
	if bbox.Width == nil || bbox.Height == nil || bbox.Left == nil || bbox.Top == nil {
		return nil, fmt.Errorf("incomplete bounding box")
	}
	if scale <= 0 {
		scale = 1.0
	}

	b := img.Bounds()
	W, H := b.Dx(), b.Dy()

	// Original bbox (in pixels)
	left := float64(*bbox.Left) * float64(W)
	top := float64(*bbox.Top) * float64(H)
	w := float64(*bbox.Width) * float64(W)
	h := float64(*bbox.Height) * float64(H)

	// Center of the original bbox
	cx := left + w/2.0
	cy := top + h/2.0

	// Expanded size
	newW := w * scale
	newH := h * scale

	// Proposed expanded rect (float)
	x0f := cx - newW/2.0
	y0f := cy - newH/2.0
	x1f := cx + newW/2.0
	y1f := cy + newH/2.0

	// Clamp to image bounds
	x0 := clamp(int(math.Round(x0f)), 0, W)
	y0 := clamp(int(math.Round(y0f)), 0, H)
	x1 := clamp(int(math.Round(x1f)), 0, W)
	y1 := clamp(int(math.Round(y1f)), 0, H)

	// Ensure we still have a valid rectangle; if not, fall back to original bbox clamped
	if x1 <= x0 || y1 <= y0 {
		// Fall back to original bbox but clamped
		x0o := clamp(int(math.Round(left)), 0, W)
		y0o := clamp(int(math.Round(top)), 0, H)
		x1o := clamp(int(math.Round(left+w)), 0, W)
		y1o := clamp(int(math.Round(top+h)), 0, H)
		if x1o <= x0o || y1o <= y0o {
			return nil, fmt.Errorf("invalid crop rectangle even after fallback")
		}
		srcRect := image.Rect(x0o, y0o, x1o, y1o)
		dst := image.NewRGBA(image.Rect(0, 0, srcRect.Dx(), srcRect.Dy()))
		draw.Draw(dst, dst.Bounds(), img, srcRect.Min, draw.Src)
		return dst, nil
	}

	// Crop using expanded & clamped rect
	srcRect := image.Rect(x0, y0, x1, y1)
	dst := image.NewRGBA(image.Rect(0, 0, srcRect.Dx(), srcRect.Dy()))
	draw.Draw(dst, dst.Bounds(), img, srcRect.Min, draw.Src)
	return dst, nil
}

// encodeJPEG encodes an image to JPEG bytes.
func encodeJPEG(img image.Image, quality int) ([]byte, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// SearchFace Implementation of SearchFace method in Face interface
func (r *rekognitionFaceIndexer) SearchAndIndexSelfieFace(ctx context.Context, imageSelfie []byte, collectionId string) (string, []string, []byte, error) {

	err := r.createCollectionIfNotExists(ctx, r.client, collectionId)
	if err != nil {
		return "", nil, nil, fmt.Errorf("failed to ensure collection exists: %v", err)
	}

	log.Printf("Delay before Index faces by 0.5 second")
	time.Sleep(500 * time.Millisecond)

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
		return "", nil, nil, fmt.Errorf("search face failed: error when try to index selfie face: %v", err)
	}

	// Check if a face was detected and indexed
	if len(resp.FaceRecords) == 0 {
		return "", nil, nil, fmt.Errorf("search face failed: no face detected in the image")
	}

	// Get the FaceId of the first indexed face
	first := resp.FaceRecords[0]
	faceId := *resp.FaceRecords[0].Face.FaceId
	fmt.Printf("Successfully Indexed FaceId: %s, ExternalImageId: %s\n", faceId, externalImageId)

	// Decode original selfie
	srcImg, _, decErr := image.Decode(bytes.NewReader(imageSelfie))
	if decErr != nil {
		return faceId, nil, nil, fmt.Errorf("failed to decode input image for cropping: %v", decErr)
	}

	// Apply orientation correction BEFORE cropping (if Rekognition indicated one)
	corrected := rotateAccordingToOrientation(srcImg, resp.OrientationCorrection)

	// Prefer Face.BoundingBox; if nil, try FaceDetail.BoundingBox
	var bbox types.BoundingBox
	if first.Face != nil && first.Face.BoundingBox != nil {
		bbox = *first.Face.BoundingBox
	} else if first.FaceDetail != nil && first.FaceDetail.BoundingBox != nil {
		bbox = *first.FaceDetail.BoundingBox
	} else {
		return faceId, nil, nil, fmt.Errorf("no bounding box returned for indexed face")
	}
	const scale = 1.8
	croppedImg, cropErr := cropWithBoundingBoxScaled(corrected, bbox, scale)
	if cropErr != nil {
		return faceId, nil, nil, fmt.Errorf("failed to crop face: %v", cropErr)
	}

	croppedBytes, encErr := encodeJPEG(croppedImg, 90)
	if encErr != nil {
		return faceId, nil, nil, fmt.Errorf("failed to encode cropped face: %v", encErr)
	}

	externalImageIdResult, err := r.SearchFacebyFaceId(ctx, faceId, collectionId)
	if err != nil {
		return faceId, nil, croppedBytes, fmt.Errorf("search Face Failed: error when try to find selfie in collection: %v", err)
	}
	return faceId, externalImageIdResult, croppedBytes, nil
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
