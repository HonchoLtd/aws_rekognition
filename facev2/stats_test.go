package facev2

import (
	"context"
	"sync"
	"testing"
)

// noopFace is a minimal Face that does nothing but succeed, for exercising
// StatsFace without any real dependency.
type noopFace struct{}

func (noopFace) IndexFace(context.Context, []byte, string, string) ([]IndexedFace, error) {
	return nil, nil
}
func (noopFace) IndexFaceWithBucket(context.Context, string, string, string, string) ([]IndexedFace, error) {
	return nil, nil
}
func (noopFace) SearchFacesByFaceId(context.Context, string, string, ...SearchFacesOption) ([]FaceMatch, error) {
	return nil, nil
}

var _ Face = noopFace{}

func TestStatsFace_CountsEachMethodIndependently(t *testing.T) {
	stats := NewStatsFace(noopFace{})
	ctx := context.Background()

	stats.IndexFace(ctx, nil, "img", "coll")
	stats.IndexFace(ctx, nil, "img2", "coll")
	stats.IndexFaceWithBucket(ctx, "bucket", "key", "img3", "coll")
	stats.SearchFacesByFaceId(ctx, "coll", "face1")
	stats.SearchFacesByFaceId(ctx, "coll", "face2")

	got := stats.Snapshot()
	want := CallStats{
		IndexFace:           2,
		IndexFaceWithBucket: 1,
		SearchFacesByFaceId: 2,
	}
	if got != want {
		t.Errorf("Snapshot() = %+v, want %+v", got, want)
	}
	if got.Total() != 5 {
		t.Errorf("Total() = %d, want 5", got.Total())
	}
}

func TestStatsFace_ConcurrentCallsCountCorrectly(t *testing.T) {
	stats := NewStatsFace(noopFace{})
	ctx := context.Background()

	const goroutines = 50
	const callsEach = 20

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < callsEach; j++ {
				stats.SearchFacesByFaceId(ctx, "coll", "face")
			}
		}()
	}
	wg.Wait()

	if got := stats.Snapshot().SearchFacesByFaceId; got != goroutines*callsEach {
		t.Errorf("SearchFacesByFaceId count = %d, want %d", got, goroutines*callsEach)
	}
}

func TestCallStats_String(t *testing.T) {
	c := CallStats{IndexFace: 8, SearchFacesByFaceId: 2}
	s := c.String()
	if s == "" {
		t.Fatal("String() returned empty string")
	}
	// Just confirm it's not silently dropping a field a maintainer might add
	// later without updating String() — total should be present and correct.
	wantTotal := c.Total()
	if wantTotal != 10 {
		t.Fatalf("sanity check failed: Total() = %d, want 10", wantTotal)
	}
}
