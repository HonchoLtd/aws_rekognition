package simulation

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/HonchoLtd/aws_rekognition/facev2"
	"github.com/HonchoLtd/aws_rekognition/facev2/store"
)

// openTestStore opens a fresh, ephemeral store.SQLiteStore for a single
// test. See facev2/store's own test helper for why ":memory:" is safe here.
func openTestStore(t *testing.T) *store.SQLiteStore {
	t.Helper()
	s, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open(:memory:) failed: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("store Close() failed: %v", err)
		}
	})
	return s
}

// countingFace wraps a FakeFace and counts SearchFacesByFaceId calls, so
// tests can assert on exactly how many AWS round-trips a DBGrouper run
// made — not just that the end result was correct. SearchFacesByFaceId is
// the only Face method DBGrouper ever calls (see db_grouper.go), so it's
// the only one worth counting here.
type countingFace struct {
	*FakeFace
	mu               sync.Mutex
	searchFacesCalls int
}

func (f *countingFace) reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.searchFacesCalls = 0
}

func (f *countingFace) SearchFacesByFaceId(ctx context.Context, collectionId string, faceId string, opts ...facev2.SearchFacesOption) ([]facev2.FaceMatch, error) {
	f.mu.Lock()
	f.searchFacesCalls++
	f.mu.Unlock()
	return f.FakeFace.SearchFacesByFaceId(ctx, collectionId, faceId, opts...)
}

// TestDBGrouper_GroupsFacesOfTheSamePersonUnderOneUser confirms the basic
// clustering outcome: every face of the same person ends up under the same
// user_id, and a single new user_id is minted for the whole group.
func TestDBGrouper_GroupsFacesOfTheSamePersonUnderOneUser(t *testing.T) {
	const collectionId = "event-db-grouper-same-person"
	ctx := context.Background()

	fake := NewFakeFace()
	grouper := &DBGrouper{Face: fake, Store: openTestStore(t)}

	const groupSize = 4
	faceIds := make([]string, 0, groupSize)
	for i := 0; i < groupSize; i++ {
		indexed, err := fake.IndexFace(ctx, []byte("ivy"), fmt.Sprintf("ivy-%d.jpg", i), collectionId)
		if err != nil || len(indexed) != 1 {
			t.Fatalf("IndexFace(ivy-%d) = %+v, %v", i, indexed, err)
		}
		faceIds = append(faceIds, indexed[0].FaceId)
	}

	result, err := grouper.ProcessBatch(ctx, collectionId, faceIds)
	if err != nil {
		t.Fatalf("ProcessBatch error: %v", err)
	}
	if len(result.CreatedUsers) != 1 {
		t.Fatalf("created %d users, want 1: %v", len(result.CreatedUsers), result.CreatedUsers)
	}

	userId := result.CreatedUsers[0]
	for _, faceId := range faceIds {
		if result.FaceToUser[faceId] != userId {
			t.Errorf("face %s resolved to %q, want %q", faceId, result.FaceToUser[faceId], userId)
		}
	}
}

// TestDBGrouper_DifferentPeopleGetDifferentUsers is the flip side: unrelated
// people must never be folded into the same user_id.
func TestDBGrouper_DifferentPeopleGetDifferentUsers(t *testing.T) {
	const collectionId = "event-db-grouper-different-people"
	ctx := context.Background()

	fake := NewFakeFace()
	grouper := &DBGrouper{Face: fake, Store: openTestStore(t)}

	var faceIds []string
	for _, person := range []string{"alice", "bob", "carol"} {
		indexed, err := fake.IndexFace(ctx, []byte(person), person+".jpg", collectionId)
		if err != nil || len(indexed) != 1 {
			t.Fatalf("IndexFace(%s) = %+v, %v", person, indexed, err)
		}
		faceIds = append(faceIds, indexed[0].FaceId)
	}

	result, err := grouper.ProcessBatch(ctx, collectionId, faceIds)
	if err != nil {
		t.Fatalf("ProcessBatch error: %v", err)
	}
	if len(result.CreatedUsers) != 3 {
		t.Fatalf("created %d users, want 3: %v", len(result.CreatedUsers), result.CreatedUsers)
	}

	seen := make(map[string]bool)
	for _, faceId := range faceIds {
		userId := result.FaceToUser[faceId]
		if userId == "" {
			t.Errorf("face %s was left unresolved", faceId)
			continue
		}
		if seen[userId] {
			t.Errorf("user %s was assigned to more than one distinct person", userId)
		}
		seen[userId] = true
	}
}

// TestDBGrouper_LaterPhotoJoinsExistingUserAcrossBatches confirms a person's
// user_id minted in one batch is correctly reused (via the Store, not any
// AWS User API) for a later photo of them in a subsequent batch.
func TestDBGrouper_LaterPhotoJoinsExistingUserAcrossBatches(t *testing.T) {
	const collectionId = "event-db-grouper-cross-batch"
	ctx := context.Background()

	fake := NewFakeFace()
	grouper := &DBGrouper{Face: fake, Store: openTestStore(t)}

	indexed1, err := fake.IndexFace(ctx, []byte("dave"), "dave-1.jpg", collectionId)
	if err != nil || len(indexed1) != 1 {
		t.Fatalf("IndexFace(dave-1) = %+v, %v", indexed1, err)
	}
	result1, err := grouper.ProcessBatch(ctx, collectionId, []string{indexed1[0].FaceId})
	if err != nil {
		t.Fatalf("ProcessBatch(batch 1) error: %v", err)
	}
	daveUserId := result1.FaceToUser[indexed1[0].FaceId]

	indexed2, err := fake.IndexFace(ctx, []byte("dave"), "dave-2.jpg", collectionId)
	if err != nil || len(indexed2) != 1 {
		t.Fatalf("IndexFace(dave-2) = %+v, %v", indexed2, err)
	}
	result2, err := grouper.ProcessBatch(ctx, collectionId, []string{indexed2[0].FaceId})
	if err != nil {
		t.Fatalf("ProcessBatch(batch 2) error: %v", err)
	}

	if len(result2.CreatedUsers) != 0 {
		t.Errorf("batch 2 created %d users, want 0 (dave's user already existed)", len(result2.CreatedUsers))
	}
	if got := result2.FaceToUser[indexed2[0].FaceId]; got != daveUserId {
		t.Errorf("batch 2 face resolved to %q, want %q (dave's existing user)", got, daveUserId)
	}
}

// TestDBGrouper_MinimizesSearchFacesCallsForANeverSeenGroup pins down the
// call-count regression this whole test file exists to catch: for N
// never-before-seen faces of the same person landing in one batch, a single
// SearchFacesByFaceId call already reveals the whole group, so the other
// N-1 faces must be persisted from that one call instead of each making
// their own redundant SearchFacesByFaceId round-trip.
func TestDBGrouper_MinimizesSearchFacesCallsForANeverSeenGroup(t *testing.T) {
	const collectionId = "event-db-grouper-call-count-same-batch"
	ctx := context.Background()

	fake := NewFakeFace()
	counting := &countingFace{FakeFace: fake}
	grouper := &DBGrouper{Face: counting, Store: openTestStore(t)}

	const groupSize = 5
	faceIds := make([]string, 0, groupSize)
	for i := 0; i < groupSize; i++ {
		indexed, err := fake.IndexFace(ctx, []byte("henry"), fmt.Sprintf("henry-%d.jpg", i), collectionId)
		if err != nil || len(indexed) != 1 {
			t.Fatalf("IndexFace(henry-%d) = %+v, %v", i, indexed, err)
		}
		faceIds = append(faceIds, indexed[0].FaceId)
	}

	result, err := grouper.ProcessBatch(ctx, collectionId, faceIds)
	if err != nil {
		t.Fatalf("ProcessBatch error: %v", err)
	}
	if len(result.CreatedUsers) != 1 {
		t.Fatalf("created %d users, want 1: %v", len(result.CreatedUsers), result.CreatedUsers)
	}
	userId := result.CreatedUsers[0]
	for _, faceId := range faceIds {
		if result.FaceToUser[faceId] != userId {
			t.Errorf("face %s resolved to %q, want %q", faceId, result.FaceToUser[faceId], userId)
		}
	}

	counting.mu.Lock()
	defer counting.mu.Unlock()
	if counting.searchFacesCalls != 1 {
		t.Errorf("SearchFacesByFaceId called %d times, want 1 (one call should discover and persist the whole %d-face group)", counting.searchFacesCalls, groupSize)
	}
}

// TestDBGrouper_PersistsGroupAcrossBatchesFromOneSearchFacesCall covers the
// scenario the real facesmoketest demo actually hits: every image is
// indexed up front, so by the time grouping runs, a person's later photo is
// already sitting in the collection even if it lands in an *earlier* batch
// than the one it happens to be processed in. The very first member of the
// group to be resolved should discover (and persist) every other member,
// including ones that haven't been handed to ProcessBatch yet — so the
// second batch here should cost 0 further SearchFacesByFaceId calls and
// mint 0 further users.
func TestDBGrouper_PersistsGroupAcrossBatchesFromOneSearchFacesCall(t *testing.T) {
	const collectionId = "event-db-grouper-call-count-cross-batch"
	ctx := context.Background()

	fake := NewFakeFace()
	counting := &countingFace{FakeFace: fake}
	grouper := &DBGrouper{Face: counting, Store: openTestStore(t)}

	// All 3 photos of "henry" are indexed before any batch is processed,
	// mirroring facesmoketest's index-everything-then-group pipeline.
	faceIds := make([]string, 0, 3)
	for i := 0; i < 3; i++ {
		indexed, err := fake.IndexFace(ctx, []byte("henry"), fmt.Sprintf("henry-%d.jpg", i), collectionId)
		if err != nil || len(indexed) != 1 {
			t.Fatalf("IndexFace(henry-%d) = %+v, %v", i, indexed, err)
		}
		faceIds = append(faceIds, indexed[0].FaceId)
	}

	// Split the group across two batches: [0,1] then [2].
	result1, err := grouper.ProcessBatch(ctx, collectionId, faceIds[:2])
	if err != nil {
		t.Fatalf("ProcessBatch(batch 1) error: %v", err)
	}
	if len(result1.CreatedUsers) != 1 {
		t.Fatalf("batch 1 created %d users, want 1: %v", len(result1.CreatedUsers), result1.CreatedUsers)
	}
	userId := result1.CreatedUsers[0]
	counting.reset() // only care about calls made processing batch 2

	result2, err := grouper.ProcessBatch(ctx, collectionId, faceIds[2:])
	if err != nil {
		t.Fatalf("ProcessBatch(batch 2) error: %v", err)
	}
	if len(result2.CreatedUsers) != 0 {
		t.Errorf("batch 2 created %d users, want 0 (henry's group was already fully discovered in batch 1)", len(result2.CreatedUsers))
	}
	if got := result2.FaceToUser[faceIds[2]]; got != userId {
		t.Errorf("batch 2 face resolved to %q, want %q", got, userId)
	}

	counting.mu.Lock()
	defer counting.mu.Unlock()
	if counting.searchFacesCalls != 0 {
		t.Errorf("SearchFacesByFaceId called %d times in batch 2, want 0 (batch 1's single call already discovered and persisted this face)", counting.searchFacesCalls)
	}
}

// TestDBGrouper_DoesNotOverwriteASiblingAlreadyOwnedByADifferentUser
// confirms the pre-assignment optimization never clobbers a face that
// already belongs to a different, pre-existing user — that assignment must
// stay authoritative over a mere similarity edge from an unrelated search.
func TestDBGrouper_DoesNotOverwriteASiblingAlreadyOwnedByADifferentUser(t *testing.T) {
	const collectionId = "event-db-grouper-no-clobber"
	ctx := context.Background()

	fake := NewFakeFace()
	grouper := &DBGrouper{Face: fake, Store: openTestStore(t)}

	// Establish ivy's own user first.
	indexedIvy, err := fake.IndexFace(ctx, []byte("ivy"), "ivy-1.jpg", collectionId)
	if err != nil || len(indexedIvy) != 1 {
		t.Fatalf("IndexFace(ivy) = %+v, %v", indexedIvy, err)
	}
	result1, err := grouper.ProcessBatch(ctx, collectionId, []string{indexedIvy[0].FaceId})
	if err != nil {
		t.Fatalf("ProcessBatch(ivy) error: %v", err)
	}
	ivyUserId := result1.FaceToUser[indexedIvy[0].FaceId]

	// A second, unrelated group forms separately.
	faceIds := make([]string, 0, 2)
	for i := 0; i < 2; i++ {
		indexed, err := fake.IndexFace(ctx, []byte("jack"), fmt.Sprintf("jack-%d.jpg", i), collectionId)
		if err != nil || len(indexed) != 1 {
			t.Fatalf("IndexFace(jack-%d) = %+v, %v", i, indexed, err)
		}
		faceIds = append(faceIds, indexed[0].FaceId)
	}
	if _, err := grouper.ProcessBatch(ctx, collectionId, faceIds); err != nil {
		t.Fatalf("ProcessBatch(jack) error: %v", err)
	}

	// ivy's assignment must be untouched by any of the above.
	userId, found, err := grouper.Store.UserForFace(ctx, collectionId, indexedIvy[0].FaceId)
	if err != nil {
		t.Fatalf("UserForFace(ivy) error: %v", err)
	}
	if !found || userId != ivyUserId {
		t.Errorf("ivy's assignment changed to (%q, found=%v), want (%q, true)", userId, found, ivyUserId)
	}
}

// TestDBGrouper_ResolvingIsIdempotent confirms re-processing a FaceId that's
// already assigned (e.g. redelivered into a later batch) doesn't touch
// SearchFacesByFaceId again and doesn't change its assignment.
func TestDBGrouper_ResolvingIsIdempotent(t *testing.T) {
	const collectionId = "event-db-grouper-idempotent"
	ctx := context.Background()

	fake := NewFakeFace()
	counting := &countingFace{FakeFace: fake}
	grouper := &DBGrouper{Face: counting, Store: openTestStore(t)}

	indexed, err := fake.IndexFace(ctx, []byte("erin"), "erin-1.jpg", collectionId)
	if err != nil || len(indexed) != 1 {
		t.Fatalf("IndexFace(erin-1) = %+v, %v", indexed, err)
	}
	faceId := indexed[0].FaceId

	result1, err := grouper.ProcessBatch(ctx, collectionId, []string{faceId})
	if err != nil {
		t.Fatalf("ProcessBatch(1st) error: %v", err)
	}
	counting.reset()

	result2, err := grouper.ProcessBatch(ctx, collectionId, []string{faceId})
	if err != nil {
		t.Fatalf("ProcessBatch(2nd, same face) error: %v", err)
	}

	if result2.FaceToUser[faceId] != result1.FaceToUser[faceId] {
		t.Errorf("re-processing changed the assignment: %q -> %q", result1.FaceToUser[faceId], result2.FaceToUser[faceId])
	}
	if len(result2.CreatedUsers) != 0 {
		t.Errorf("re-processing created %d users, want 0", len(result2.CreatedUsers))
	}

	counting.mu.Lock()
	defer counting.mu.Unlock()
	if counting.searchFacesCalls != 0 {
		t.Errorf("SearchFacesByFaceId called %d times on the re-process, want 0 (already assigned)", counting.searchFacesCalls)
	}
}

// --- Representative-thumbnail selection ---
//
// These tests drive DBGrouper's representative selection directly by
// calling Store.RecordFaceMetrics themselves (exactly what a real caller
// does right after facev2.Face.IndexFace, using its returned
// facev2.IndexedFace.Score) — FakeFace itself has no notion of the
// scoring inputs, so this is the seam these tests use to control the
// score deterministically. The raw FaceOccluded fields on FaceMetrics are
// left at zero here because Score already reflects their contribution;
// they're informational only from the store's point of view.

// indexWithScore indexes a face for personId and records its combined
// score, in one step, mirroring what a real indexing pipeline would do
// right after IndexFace using its returned facev2.IndexedFace.Score
// (which already folds pose/quality/occlusion together).
func indexWithScore(t *testing.T, fake *FakeFace, s *store.SQLiteStore, collectionId string, personId string, externalImageId string, score float64) string {
	t.Helper()
	ctx := context.Background()
	indexed, err := fake.IndexFace(ctx, []byte(personId), externalImageId, collectionId)
	if err != nil || len(indexed) != 1 {
		t.Fatalf("IndexFace(%s) = %+v, %v", externalImageId, indexed, err)
	}
	faceId := indexed[0].FaceId
	if err := s.RecordFaceMetrics(ctx, collectionId, faceId, store.FaceMetrics{Score: score}); err != nil {
		t.Fatalf("RecordFaceMetrics(%s) error: %v", faceId, err)
	}
	return faceId
}

// TestDBGrouper_LocksInFirstFaceThatPassesThreshold confirms the core
// selection rule: once a face's score passes the threshold, it's locked in
// as the representative, and a LATER, even better-scoring face of the same
// person does not replace it.
func TestDBGrouper_LocksInFirstFaceThatPassesThreshold(t *testing.T) {
	const collectionId = "event-representative-locks-in"
	ctx := context.Background()

	fake := NewFakeFace()
	s := openTestStore(t)
	grouper := &DBGrouper{Face: fake, Store: s, RepresentativeThreshold: 80}

	// Indexed (and processed) one at a time, each right before its own
	// ProcessBatch call — mirroring photos arriving over time, rather than
	// all being indexed up front — so each SearchFacesByFaceId only ever
	// discovers faces indexed so far and the locking behavior is exercised
	// incrementally instead of being resolved in one shot as a group.
	faceLow := indexWithScore(t, fake, s, collectionId, "mia", "mia-low.jpg", 50)
	if _, err := grouper.ProcessBatch(ctx, collectionId, []string{faceLow}); err != nil {
		t.Fatalf("ProcessBatch(low) error: %v", err)
	}
	userId := mustUserFor(t, s, ctx, collectionId, faceLow)

	rep, found, err := s.Representative(ctx, collectionId, userId)
	if err != nil || !found || rep.FaceId != faceLow || rep.Locked {
		t.Fatalf("after low-score face: Representative() = %+v, found=%v, err=%v, want unlocked fallback on %s", rep, found, err, faceLow)
	}

	facePass := indexWithScore(t, fake, s, collectionId, "mia", "mia-pass.jpg", 85)
	if _, err := grouper.ProcessBatch(ctx, collectionId, []string{facePass}); err != nil {
		t.Fatalf("ProcessBatch(pass) error: %v", err)
	}
	rep, found, err = s.Representative(ctx, collectionId, userId)
	if err != nil || !found || rep.FaceId != facePass || !rep.Locked {
		t.Fatalf("after passing-score face: Representative() = %+v, found=%v, err=%v, want locked on %s", rep, found, err, facePass)
	}

	faceBetter := indexWithScore(t, fake, s, collectionId, "mia", "mia-better.jpg", 99)
	if _, err := grouper.ProcessBatch(ctx, collectionId, []string{faceBetter}); err != nil {
		t.Fatalf("ProcessBatch(better) error: %v", err)
	}
	rep, found, err = s.Representative(ctx, collectionId, userId)
	if err != nil || !found || rep.FaceId != facePass || !rep.Locked {
		t.Errorf("after an even-better-scoring face arrived: Representative() = %+v, want it to STILL be %s (locked search must not reopen)", rep, facePass)
	}
}

// TestDBGrouper_FallsBackToBestScoreSeenWhenNonePass confirms that when no
// face ever reaches the threshold, the representative tracks whichever
// scored highest so far — not the first or last one processed.
func TestDBGrouper_FallsBackToBestScoreSeenWhenNonePass(t *testing.T) {
	const collectionId = "event-representative-fallback"
	ctx := context.Background()

	fake := NewFakeFace()
	s := openTestStore(t)
	grouper := &DBGrouper{Face: fake, Store: s, RepresentativeThreshold: 80}

	faceMed := indexWithScore(t, fake, s, collectionId, "noah", "noah-med.jpg", 60)
	faceLow := indexWithScore(t, fake, s, collectionId, "noah", "noah-low.jpg", 40)
	faceBest := indexWithScore(t, fake, s, collectionId, "noah", "noah-best.jpg", 75) // still below 80

	for _, faceId := range []string{faceMed, faceLow, faceBest} {
		if _, err := grouper.ProcessBatch(ctx, collectionId, []string{faceId}); err != nil {
			t.Fatalf("ProcessBatch(%s) error: %v", faceId, err)
		}
	}

	userId := mustUserFor(t, s, ctx, collectionId, faceMed)
	rep, found, err := s.Representative(ctx, collectionId, userId)
	if err != nil {
		t.Fatalf("Representative() error: %v", err)
	}
	if !found || rep.Locked {
		t.Fatalf("Representative() = %+v, found=%v, want an unlocked fallback (nothing passed 80)", rep, found)
	}
	if rep.FaceId != faceBest {
		t.Errorf("Representative().FaceId = %s, want %s (the highest-scoring of the three, even though it wasn't first or last)", rep.FaceId, faceBest)
	}
}

// TestDBGrouper_SkipsRepresentativeWhenScoreNeverRecorded confirms
// representative selection is opt-in: a face resolved without ever having
// its score recorded must not error, and must leave the user without a
// representative rather than fabricating one.
func TestDBGrouper_SkipsRepresentativeWhenScoreNeverRecorded(t *testing.T) {
	const collectionId = "event-representative-no-score"
	ctx := context.Background()

	fake := NewFakeFace()
	s := openTestStore(t)
	grouper := &DBGrouper{Face: fake, Store: s}

	indexed, err := fake.IndexFace(ctx, []byte("olivia"), "olivia-1.jpg", collectionId)
	if err != nil || len(indexed) != 1 {
		t.Fatalf("IndexFace = %+v, %v", indexed, err)
	}
	faceId := indexed[0].FaceId

	result, err := grouper.ProcessBatch(ctx, collectionId, []string{faceId})
	if err != nil {
		t.Fatalf("ProcessBatch error: %v", err)
	}
	userId := result.FaceToUser[faceId]

	_, found, err := s.Representative(ctx, collectionId, userId)
	if err != nil {
		t.Fatalf("Representative() error: %v", err)
	}
	if found {
		t.Errorf("Representative() found a candidate, want none (score was never recorded)")
	}
}

// TestDBGrouper_DefaultRepresentativeThreshold confirms a zero-value
// RepresentativeThreshold field falls back to DefaultRepresentativeThreshold
// rather than locking in on the very first face regardless of score.
func TestDBGrouper_DefaultRepresentativeThreshold(t *testing.T) {
	const collectionId = "event-representative-default-threshold"
	ctx := context.Background()

	fake := NewFakeFace()
	s := openTestStore(t)
	grouper := &DBGrouper{Face: fake, Store: s} // RepresentativeThreshold left unset

	faceId := indexWithScore(t, fake, s, collectionId, "pete", "pete-1.jpg", DefaultRepresentativeThreshold-1)
	if _, err := grouper.ProcessBatch(ctx, collectionId, []string{faceId}); err != nil {
		t.Fatalf("ProcessBatch error: %v", err)
	}
	userId := mustUserFor(t, s, ctx, collectionId, faceId)

	rep, found, err := s.Representative(ctx, collectionId, userId)
	if err != nil || !found || rep.Locked {
		t.Fatalf("Representative() = %+v, found=%v, err=%v, want an unlocked fallback (score is below the default threshold)", rep, found, err)
	}
}

// mustUserFor looks up which userId faceId was resolved to, failing the test
// if it wasn't resolved at all.
func mustUserFor(t *testing.T, s *store.SQLiteStore, ctx context.Context, collectionId string, faceId string) string {
	t.Helper()
	userId, found, err := s.UserForFace(ctx, collectionId, faceId)
	if err != nil || !found {
		t.Fatalf("UserForFace(%s) = %q, found=%v, err=%v", faceId, userId, found, err)
	}
	return userId
}

// Occlusion is no longer handled at this layer — it's already folded into
// each face's Score via facev2.ComputeFaceScore (see facev2.OccludedScore
// for the mapping from AWS's FaceOccluded {Value, Confidence} pair). The
// dedicated occlusion-tier tests that lived here previously have been
// removed; the behavior they used to cover (heavily-occluded faces losing
// to lightly-occluded ones, etc.) is exercised by score_test.go against
// the score formula directly, and by the score-based tests above against
// the streaming selection.
