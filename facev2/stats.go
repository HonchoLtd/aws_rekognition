package facev2

import (
	"context"
	"fmt"
	"sync/atomic"
)

// CallStats counts how many times each Face method was called. Read it via
// StatsFace.Snapshot() — never read the fields directly while calls may
// still be in flight, since the atomic counters back concurrent Add()s.
type CallStats struct {
	IndexFace           int64 `json:"index_face"`
	IndexFaceWithBucket int64 `json:"index_face_with_bucket"`
	SearchFacesByFaceId int64 `json:"search_faces_by_face_id"`
}

// Total is the sum of every counted call — the number of AWS Rekognition
// API round-trips CallStats observed.
func (c CallStats) Total() int64 {
	return c.IndexFace + c.IndexFaceWithBucket + c.SearchFacesByFaceId
}

// String renders a one-line-per-call breakdown, suitable for dropping
// straight into a summary report.
func (c CallStats) String() string {
	return fmt.Sprintf(
		"IndexFace=%d IndexFaceWithBucket=%d SearchFacesByFaceId=%d (total=%d)",
		c.IndexFace, c.IndexFaceWithBucket, c.SearchFacesByFaceId, c.Total(),
	)
}

// atomicCallStats is CallStats' concurrent-safe backing store — one
// atomic.Int64 per counted method, incremented from StatsFace's wrapper
// methods and read out via Snapshot().
type atomicCallStats struct {
	indexFace           atomic.Int64
	indexFaceWithBucket atomic.Int64
	searchFacesByFaceId atomic.Int64
}

func (a *atomicCallStats) snapshot() CallStats {
	return CallStats{
		IndexFace:           a.indexFace.Load(),
		IndexFaceWithBucket: a.indexFaceWithBucket.Load(),
		SearchFacesByFaceId: a.searchFacesByFaceId.Load(),
	}
}

// StatsFace wraps a Face implementation and counts calls made through it,
// without changing behavior at all — every call is delegated unchanged. Use
// it to answer "how many AWS Rekognition API calls does processing this set
// of images actually make", e.g.:
//
//	base := facev2.NewRekognitionFaceIndexer(client)
//	tracked := facev2.NewStatsFace(base)
//	// ... run your indexing + grouping pipeline against `tracked` ...
//	fmt.Println(tracked.Snapshot())
//
// It works identically wrapping any Face implementation, including a fake
// one in tests.
type StatsFace struct {
	face  Face
	stats atomicCallStats
}

// NewStatsFace wraps face with call counting.
func NewStatsFace(face Face) *StatsFace {
	return &StatsFace{face: face}
}

// Snapshot returns the call counts observed so far. Safe to call at any
// time, including while other calls are in flight (it reflects whatever has
// completed incrementing up to that point).
func (s *StatsFace) Snapshot() CallStats {
	return s.stats.snapshot()
}

func (s *StatsFace) IndexFace(ctx context.Context, image []byte, imageID string, eventID string) ([]IndexedFace, error) {
	s.stats.indexFace.Add(1)
	return s.face.IndexFace(ctx, image, imageID, eventID)
}

func (s *StatsFace) IndexFaceWithBucket(ctx context.Context, s3Bucket string, s3Key string, imageID string, eventID string) ([]IndexedFace, error) {
	s.stats.indexFaceWithBucket.Add(1)
	return s.face.IndexFaceWithBucket(ctx, s3Bucket, s3Key, imageID, eventID)
}

func (s *StatsFace) SearchFacesByFaceId(ctx context.Context, collectionId string, faceId string, opts ...SearchFacesOption) ([]FaceMatch, error) {
	s.stats.searchFacesByFaceId.Add(1)
	return s.face.SearchFacesByFaceId(ctx, collectionId, faceId, opts...)
}

var _ Face = (*StatsFace)(nil)
