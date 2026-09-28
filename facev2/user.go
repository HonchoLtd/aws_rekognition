// This file adds the AWS Rekognition "SearchFaces" API used to find
// previously indexed faces that look like a given one, by raw
// face-to-face similarity:
//   - SearchFaces: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_SearchFaces.html
//
// AWS Rekognition's own "User" APIs (CreateUser, AssociateFaces,
// SearchUsers) are deliberately NOT used here — grouping which faces belong
// to the same real person, and minting the user_id for that group, is done
// entirely outside AWS: see facev2/simulation.DBGrouper (the clustering
// algorithm, driven solely by SearchFacesByFaceId below) and facev2/store
// (where the resulting face_id -> user_id mapping is persisted).
package facev2

import (
	"context"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
	"github.com/aws/aws-sdk-go-v2/service/rekognition/types"
)

// FaceMatch is a previously indexed face that SearchFacesByFaceId found to
// be similar to the queried FaceId — i.e. very likely the same person.
//
// Only the two fields an external grouping layer actually needs to make a
// decision are exposed: FaceId (the identity of the sibling face, used to
// look up which user_id — if any — it's already been assigned to) and
// Similarity (AWS's confidence in the match, useful for the caller's own
// gating). The caller already knows the source image / external image id
// of every face it indexed, so surfacing those on the match too would just
// be redundant.
type FaceMatch struct {
	FaceId     string  `json:"face_id"`
	Similarity float32 `json:"similarity"`
}

// SearchFacesByFaceId answers "which other indexed faces in this collection
// look like the same person as faceId?", ordered by similarity (highest
// first). An external grouping layer uses each match's FaceId to look up
// (in its own datastore) whether that sibling face is already assigned to
// a user_id, and adopts the highest-similarity such user; see facev2's
// README for the full algorithm.
func (r *rekognitionFaceIndexer) SearchFacesByFaceId(ctx context.Context, collectionId string, faceId string, opts ...SearchFacesOption) ([]FaceMatch, error) {
	cfg := &searchFacesConfig{}
	for _, opt := range opts {
		opt(cfg)
	}

	input := &rekognition.SearchFacesInput{
		CollectionId:       aws.String(collectionId),
		FaceId:             aws.String(faceId),
		FaceMatchThreshold: cfg.faceMatchThreshold,
		MaxFaces:           cfg.maxFaces,
	}

	start := time.Now()
	var resp *rekognition.SearchFacesOutput
	err := withRetry(ctx, defaultRetryPolicy, "SearchFaces", func() error {
		var err error
		resp, err = r.client.SearchFaces(ctx, input)
		return err
	})
	duration := time.Since(start)
	if err != nil {
		getLogger().Error("rekognition api call failed", "op", "SearchFaces", "collection_id", collectionId, "face_id", faceId, "duration_ms", duration.Milliseconds(), "error", err)
		return nil, fmt.Errorf("failed to search faces by face id: %v", err)
	}

	matches := extractFaceMatches(resp.FaceMatches)
	getLogger().Info("rekognition api call", "op", "SearchFaces", "collection_id", collectionId, "face_id", faceId, "duration_ms", duration.Milliseconds(), "matched_faces", len(matches))
	return matches, nil
}

// extractFaceMatches maps SearchFaces' FaceMatches into FaceMatch, skipping
// entries with a nil Face or FaceId.
func extractFaceMatches(faceMatches []types.FaceMatch) []FaceMatch {
	matches := make([]FaceMatch, 0, len(faceMatches))
	for _, faceMatch := range faceMatches {
		if faceMatch.Face == nil || faceMatch.Face.FaceId == nil {
			continue
		}
		matches = append(matches, FaceMatch{
			FaceId:     *faceMatch.Face.FaceId,
			Similarity: aws.ToFloat32(faceMatch.Similarity),
		})
	}
	return matches
}
