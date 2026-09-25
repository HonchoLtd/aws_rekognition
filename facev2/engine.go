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

// FaceOccluded mirrors the relevant fields of AWS Rekognition's
// FaceOccluded type, returned as part of IndexFaces'
// FaceRecords[].FaceDetail — but ONLY when the FACE_OCCLUDED (or ALL)
// attribute is explicitly requested via IndexFacesInput.DetectionAttributes.
// It is NOT part of the DEFAULT attribute set, unlike Pose/Quality.
//
// Value reports whether the face's eyes, nose, and mouth are partially
// captured or covered (masks, sunglasses, hands, phones, ...). Confidence
// (0-100) is how confident AWS is in that Value.
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_FaceOccluded.html
type FaceOccluded struct {
	Value      bool    `json:"value"`
	Confidence float32 `json:"confidence"`
}

// IndexedFace holds the metadata we care about from a single entry of
// IndexFaces' response FaceRecords[] — both FaceRecords[].Face
// (FaceId/BoundingBox) and FaceRecords[].FaceDetail (Pose/Quality, part of
// AWS's DEFAULT attribute set, always returned with no extra request
// parameters needed; plus FaceOccluded, which requires FACE_OCCLUDED to be
// explicitly requested — see IndexFaces below).
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
type IndexedFace struct {
	FaceId      string          `json:"face_id"`
	BoundingBox FaceBoundingBox `json:"bounding_box"`
	Pose        FacePose        `json:"pose"`
	Quality     FaceQuality     `json:"quality"`
	// Occluded is AWS's judgment on whether the face is occluded (mask,
	// sunglasses, hand, phone, ...) plus how confident AWS is in that
	// judgment. Populated only when IndexFaces is asked for the
	// FACE_OCCLUDED attribute — this package does so; a nil FaceOccluded in
	// the raw response is surfaced as the zero-value (not occluded, 0
	// confidence). It is a direct input to ComputeFaceScore via
	// OccludedScore — see score.go.
	Occluded FaceOccluded `json:"occluded"`
	// Score is ComputeFaceScore(Pose, Quality, Occluded) — see score.go.
	// Computed here so every Face implementation (including FakeFace) hands
	// callers a ready-to-use score without needing to know the formula
	// themselves.
	Score float32 `json:"score"`
}

type Face interface {
	IndexFace(ctx context.Context, image []byte, imageID string, eventID string) ([]IndexedFace, error)
	IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, imageID string, eventID string) ([]IndexedFace, error)

	// SearchFacesByFaceId supports grouping multiple indexed faces (photos)
	// that belong to the same real person, purely by raw face-to-face
	// similarity — no AWS Rekognition User is ever created or associated.
	// See facev2/simulation.DBGrouper and facev2/store for how the
	// resulting face_id -> user_id grouping is generated (a locally minted
	// id, not an AWS one) and persisted. See user.go.
	SearchFacesByFaceId(ctx context.Context, collectionId string, faceId string, opts ...SearchFacesOption) ([]FaceMatch, error)
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
	start := time.Now()
	_, err := rekognitionClient.DescribeCollection(ctx, &rekognition.DescribeCollectionInput{
		CollectionId: aws.String(collectionId),
	})
	if err == nil {
		getLogger().Debug("rekognition api call", "op", "DescribeCollection", "collection_id", collectionId, "duration_ms", time.Since(start).Milliseconds(), "result", "exists")
		return nil
	}

	// DescribeCollection erroring doesn't necessarily mean the collection is
	// missing (could be throttling, permissions, ...), but that's the
	// existing assumption here — log what we actually saw so a false
	// "doesn't exist" is debuggable rather than silent.
	getLogger().Info("collection not found (or DescribeCollection failed), creating it", "collection_id", collectionId, "describe_error", err)

	start = time.Now()
	_, err = rekognitionClient.CreateCollection(ctx, &rekognition.CreateCollectionInput{
		CollectionId: aws.String(collectionId),
	})
	duration := time.Since(start)
	if err != nil {
		var rae *types.ResourceAlreadyExistsException
		if errors.As(err, &rae) {
			getLogger().Info("rekognition api call", "op", "CreateCollection", "collection_id", collectionId, "duration_ms", duration.Milliseconds(), "result", "already_exists")
			return nil
		}
		getLogger().Error("rekognition api call failed", "op", "CreateCollection", "collection_id", collectionId, "duration_ms", duration.Milliseconds(), "error", err)
		return fmt.Errorf("eror is not ResourceAlreadyExistsException failed to create collection: %v", err)
	}
	getLogger().Info("rekognition api call", "op", "CreateCollection", "collection_id", collectionId, "duration_ms", duration.Milliseconds(), "result", "created")

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

	// Prepare the input for the IndexFaces API. FACE_OCCLUDED is asked for
	// explicitly because it's NOT part of AWS's DEFAULT attribute set (unlike
	// Pose/Quality, which come back for free), and we want to surface it on
	// every IndexedFace — see IndexedFace.Occluded.
	// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html#rekognition-IndexFaces-request-DetectionAttributes
	input := &rekognition.IndexFacesInput{
		CollectionId:        aws.String(collectionId),
		Image:               &types.Image{Bytes: imageBytes},
		ExternalImageId:     aws.String(externalImageId),
		DetectionAttributes: []types.Attribute{types.AttributeFaceOccluded},
	}

	getLogger().Debug("delaying before IndexFaces for collection consistency", "delay_ms", 500)
	time.Sleep(500 * time.Millisecond)

	// Call the IndexFaces API
	start := time.Now()
	resp, err := r.client.IndexFaces(ctx, input)
	duration := time.Since(start)
	if err != nil {
		getLogger().Error("rekognition api call failed", "op", "IndexFaces", "collection_id", collectionId, "external_image_id", externalImageId, "duration_ms", duration.Milliseconds(), "error", err)
		return nil, fmt.Errorf("failed to index face: %v", err)
	}

	// Build the response metadata from FaceRecords[].Face, per the IndexFaces
	// API contract: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
	indexedFaces := extractIndexedFaces(resp.FaceRecords)
	getLogger().Info("rekognition api call", "op", "IndexFaces", "collection_id", collectionId, "external_image_id", externalImageId, "duration_ms", duration.Milliseconds(), "faces_found", len(indexedFaces))
	for _, indexedFace := range indexedFaces {
		getLogger().Debug("face indexed", "collection_id", collectionId, "external_image_id", externalImageId, "face_id", indexedFace.FaceId, "bounding_box", indexedFace.BoundingBox, "pose", indexedFace.Pose, "quality", indexedFace.Quality, "occluded", indexedFace.Occluded, "score", indexedFace.Score)
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

	// Prepare the input for the IndexFaces API using S3Object. Same
	// FACE_OCCLUDED opt-in as the byte-based IndexFace above — see there for
	// why.
	input := &rekognition.IndexFacesInput{
		CollectionId: aws.String(collectionId),
		Image: &types.Image{
			S3Object: &types.S3Object{
				Bucket: aws.String(s3Bucket),
				Name:   aws.String(s3Key),
			},
		},
		ExternalImageId:     aws.String(externalImageId),
		DetectionAttributes: []types.Attribute{types.AttributeFaceOccluded},
	}

	// Call the IndexFaces API
	start := time.Now()
	resp, err := r.client.IndexFaces(ctx, input)
	duration := time.Since(start)
	if err != nil {
		getLogger().Error("rekognition api call failed", "op", "IndexFaces", "collection_id", collectionId, "external_image_id", externalImageId, "s3_bucket", s3Bucket, "s3_key", s3Key, "duration_ms", duration.Milliseconds(), "error", err)
		return nil, fmt.Errorf("failed to index face: %v", err)
	}

	// Build the response metadata from FaceRecords[].Face, per the IndexFaces
	// API contract: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_IndexFaces.html
	indexedFaces := extractIndexedFaces(resp.FaceRecords)
	getLogger().Info("rekognition api call", "op", "IndexFaces", "collection_id", collectionId, "external_image_id", externalImageId, "s3_bucket", s3Bucket, "s3_key", s3Key, "duration_ms", duration.Milliseconds(), "faces_found", len(indexedFaces))
	for _, indexedFace := range indexedFaces {
		getLogger().Debug("face indexed", "collection_id", collectionId, "external_image_id", externalImageId, "face_id", indexedFace.FaceId, "bounding_box", indexedFace.BoundingBox, "pose", indexedFace.Pose, "quality", indexedFace.Quality, "occluded", indexedFace.Occluded, "score", indexedFace.Score)
	}

	return indexedFaces, nil
}

// extractIndexedFaces maps IndexFaces' FaceRecords (resp.FaceRecords[].Face
// and resp.FaceRecords[].FaceDetail) into the metadata we expose to callers.
// FaceRecords with a nil Face or BoundingBox are skipped defensively,
// matching the fact that most Face fields are documented as optional in the
// API reference.
func extractIndexedFaces(faceRecords []types.FaceRecord) []IndexedFace {
	indexedFaces := make([]IndexedFace, 0, len(faceRecords))
	for _, faceRecord := range faceRecords {
		if faceRecord.Face == nil || faceRecord.Face.FaceId == nil || faceRecord.Face.BoundingBox == nil {
			continue
		}
		pose := toFacePose(faceRecord.FaceDetail)
		quality := toFaceQuality(faceRecord.FaceDetail)
		occluded := toFaceOccluded(faceRecord.FaceDetail)
		indexedFaces = append(indexedFaces, IndexedFace{
			FaceId:      *faceRecord.Face.FaceId,
			BoundingBox: toFaceBoundingBox(faceRecord.Face.BoundingBox),
			Pose:        pose,
			Quality:     quality,
			Occluded:    occluded,
			Score:       ComputeFaceScore(pose, quality, occluded),
		})
	}
	return indexedFaces
}

// toFacePose extracts FacePose from a FaceRecord's FaceDetail. Pose is part
// of AWS's DEFAULT attribute set, so detail (and detail.Pose) should always
// be present in practice; a nil detail/Pose is handled defensively as the
// zero-value (perfectly frontal) pose rather than panicking.
func toFacePose(detail *types.FaceDetail) FacePose {
	if detail == nil || detail.Pose == nil {
		return FacePose{}
	}
	return FacePose{
		Yaw:   aws.ToFloat32(detail.Pose.Yaw),
		Pitch: aws.ToFloat32(detail.Pose.Pitch),
		Roll:  aws.ToFloat32(detail.Pose.Roll),
	}
}

// toFaceQuality extracts FaceQuality from a FaceRecord's FaceDetail. Quality
// is part of AWS's DEFAULT attribute set, so detail (and detail.Quality)
// should always be present in practice; a nil detail/Quality is handled
// defensively as the zero-value (lowest possible) quality.
func toFaceQuality(detail *types.FaceDetail) FaceQuality {
	if detail == nil || detail.Quality == nil {
		return FaceQuality{}
	}
	return FaceQuality{
		Brightness: aws.ToFloat32(detail.Quality.Brightness),
		Sharpness:  aws.ToFloat32(detail.Quality.Sharpness),
	}
}

// toFaceOccluded extracts FaceOccluded from a FaceRecord's FaceDetail.
// Unlike Pose/Quality, FaceOccluded is NOT in the DEFAULT attribute set and
// is only present when IndexFaces was called with FACE_OCCLUDED (or ALL) in
// DetectionAttributes — the byte- and S3-backed IndexFace variants above
// both request it. A nil detail/FaceOccluded is treated as the zero-value
// (not occluded, zero confidence) rather than an error.
func toFaceOccluded(detail *types.FaceDetail) FaceOccluded {
	if detail == nil || detail.FaceOccluded == nil {
		return FaceOccluded{}
	}
	return FaceOccluded{
		Value:      detail.FaceOccluded.Value,
		Confidence: aws.ToFloat32(detail.FaceOccluded.Confidence),
	}
}

// toFaceBoundingBox converts an AWS *types.BoundingBox into our own
// FaceBoundingBox, treating a nil pointer as the zero-value box.
func toFaceBoundingBox(bbox *types.BoundingBox) FaceBoundingBox {
	if bbox == nil {
		return FaceBoundingBox{}
	}
	return FaceBoundingBox{
		Width:  aws.ToFloat32(bbox.Width),
		Height: aws.ToFloat32(bbox.Height),
		Left:   aws.ToFloat32(bbox.Left),
		Top:    aws.ToFloat32(bbox.Top),
	}
}
