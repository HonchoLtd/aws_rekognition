// Package simulation exercises the batching + face-clustering flow described
// for facev2 (IndexFace -> SearchFacesByFaceId, with the face_id -> user_id
// grouping generated and persisted outside AWS entirely — see DBGrouper and
// facev2/store) end-to-end, without calling real AWS Rekognition.
//
// It does this with a FakeFace: an in-memory stand-in for facev2.Face that
// implements the same interface, so the exact same Batcher/DBGrouper code
// that would run against the real rekognitionFaceIndexer can be driven here
// deterministically and asserted against in tests.
//
// FakeFace can't do actual face recognition, so it cheats: the "image bytes"
// passed to IndexFace ARE the ground-truth person id (e.g. []byte("alice")).
// Two indexed faces are considered a Rekognition match if and only if they
// were indexed with the same person id. That's enough to validate the
// orchestration logic (batching, dedup, cross-batch consistency) — which is
// the part this simulation is for — without needing real images or models.
package simulation

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/HonchoLtd/aws_rekognition/facev2"
	"github.com/google/uuid"
)

// FakeFace is an in-memory, concurrency-safe stand-in for facev2.Face.
type FakeFace struct {
	mu sync.Mutex

	// faceRecord holds what IndexFace "detected" for a FaceId.
	faces map[string]fakeFaceRecord // faceId -> record
}

type fakeFaceRecord struct {
	personId     string // ground truth identity, used as the similarity oracle
	collectionId string
}

// NewFakeFace creates an empty FakeFace backend.
func NewFakeFace() *FakeFace {
	return &FakeFace{
		faces: make(map[string]fakeFaceRecord),
	}
}

var _ facev2.Face = (*FakeFace)(nil)

// IndexFace "indexes" personId (passed as imageBytes) into collectionId,
// assigning it a brand-new FaceId — mirroring what AWS Rekognition's
// IndexFaces would return for a single detected face. The externalImageId
// argument is accepted (to match the Face interface) but ignored: FaceMatch
// no longer surfaces it, and clustering doesn't need it.
func (f *FakeFace) IndexFace(_ context.Context, imageBytes []byte, _ string, collectionId string) ([]facev2.IndexedFace, error) {
	personId := string(imageBytes)
	if personId == "" {
		return nil, fmt.Errorf("simulation: IndexFace requires non-empty imageBytes (used as the ground-truth person id)")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	faceId := uuid.New().String()
	f.faces[faceId] = fakeFaceRecord{
		personId:     personId,
		collectionId: collectionId,
	}

	return []facev2.IndexedFace{
		{
			FaceId:      faceId,
			BoundingBox: facev2.FaceBoundingBox{Width: 0.2, Height: 0.2, Left: 0.1, Top: 0.1},
		},
	}, nil
}

// IndexFaceWithBucket behaves like IndexFace, treating s3Key as the
// ground-truth person id so callers can simulate S3-backed indexing too.
func (f *FakeFace) IndexFaceWithBucket(ctx context.Context, _ string, s3Key string, externalImageId string, collectionId string) ([]facev2.IndexedFace, error) {
	return f.IndexFace(ctx, []byte(s3Key), externalImageId, collectionId)
}

// SearchFacesByFaceId returns every other indexed face sharing faceId's
// ground-truth person id, within the same collection — the simulation's
// stand-in for "faces AWS Rekognition considers the same person".
func (f *FakeFace) SearchFacesByFaceId(_ context.Context, collectionId string, faceId string, _ ...facev2.SearchFacesOption) ([]facev2.FaceMatch, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	record, ok := f.faces[faceId]
	if !ok {
		return nil, fmt.Errorf("simulation: face %q not found in collection %q", faceId, collectionId)
	}

	var matches []facev2.FaceMatch
	for otherFaceId, otherRecord := range f.faces {
		if otherFaceId == faceId {
			continue
		}
		if otherRecord.collectionId != collectionId || otherRecord.personId != record.personId {
			continue
		}
		matches = append(matches, facev2.FaceMatch{
			FaceId:     otherFaceId,
			Similarity: 99,
		})
	}

	// Deterministic ordering makes assertions in tests reproducible.
	sort.Slice(matches, func(i, j int) bool { return matches[i].FaceId < matches[j].FaceId })
	return matches, nil
}
