package store

import (
	"context"
	"testing"
)

// openTestStore opens a fresh, ephemeral SQLiteStore for a single test.
// ":memory:" plus SQLiteStore's own single-connection pool keeps this both
// isolated per test and consistent for the test's whole lifetime (SQLite's
// ":memory:" database lives only as long as its one connection stays open).
func openTestStore(t *testing.T) *SQLiteStore {
	t.Helper()
	s, err := Open(":memory:")
	if err != nil {
		t.Fatalf("Open(:memory:) failed: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Close() failed: %v", err)
		}
	})
	return s
}

func TestSQLiteStore_UserForFace_NotAssignedYet(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	userId, found, err := s.UserForFace(ctx, "col-1", "face-unknown")
	if err != nil {
		t.Fatalf("UserForFace() error: %v", err)
	}
	if found {
		t.Errorf("UserForFace() found = true, userId = %q, want not found", userId)
	}
}

func TestSQLiteStore_AssignFaceThenLookUp(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.CreateUser(ctx, "col-1", "user-1"); err != nil {
		t.Fatalf("CreateUser() error: %v", err)
	}
	if err := s.AssignFace(ctx, "col-1", "face-1", "user-1"); err != nil {
		t.Fatalf("AssignFace() error: %v", err)
	}

	userId, found, err := s.UserForFace(ctx, "col-1", "face-1")
	if err != nil {
		t.Fatalf("UserForFace() error: %v", err)
	}
	if !found || userId != "user-1" {
		t.Errorf("UserForFace() = (%q, %v), want (%q, true)", userId, found, "user-1")
	}
}

// TestSQLiteStore_ScopedByCollection confirms the same face_id/user_id
// values in two different collections are tracked independently, mirroring
// how Rekognition itself scopes FaceIds/UserIds to a collection.
func TestSQLiteStore_ScopedByCollection(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.CreateUser(ctx, "col-a", "user-1"); err != nil {
		t.Fatalf("CreateUser(col-a) error: %v", err)
	}
	if err := s.AssignFace(ctx, "col-a", "face-1", "user-1"); err != nil {
		t.Fatalf("AssignFace(col-a) error: %v", err)
	}

	if _, found, err := s.UserForFace(ctx, "col-b", "face-1"); err != nil {
		t.Fatalf("UserForFace(col-b) error: %v", err)
	} else if found {
		t.Errorf("UserForFace(col-b, face-1) found an assignment that only exists in col-a")
	}
}

// TestSQLiteStore_CreateUserIsIdempotent confirms creating the same userId
// twice is a no-op rather than an error.
func TestSQLiteStore_CreateUserIsIdempotent(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.CreateUser(ctx, "col-1", "user-1"); err != nil {
		t.Fatalf("CreateUser() (1st) error: %v", err)
	}
	if err := s.CreateUser(ctx, "col-1", "user-1"); err != nil {
		t.Fatalf("CreateUser() (2nd, same id) error: %v", err)
	}
}

// TestSQLiteStore_AssignFaceOverwritesPreviousUser confirms re-assigning a
// FaceId to a different UserId is "last write wins" rather than an error or
// a silently ignored write.
func TestSQLiteStore_AssignFaceOverwritesPreviousUser(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	for _, userId := range []string{"user-1", "user-2"} {
		if err := s.CreateUser(ctx, "col-1", userId); err != nil {
			t.Fatalf("CreateUser(%s) error: %v", userId, err)
		}
	}
	if err := s.AssignFace(ctx, "col-1", "face-1", "user-1"); err != nil {
		t.Fatalf("AssignFace(user-1) error: %v", err)
	}
	if err := s.AssignFace(ctx, "col-1", "face-1", "user-2"); err != nil {
		t.Fatalf("AssignFace(user-2) error: %v", err)
	}

	userId, found, err := s.UserForFace(ctx, "col-1", "face-1")
	if err != nil {
		t.Fatalf("UserForFace() error: %v", err)
	}
	if !found || userId != "user-2" {
		t.Errorf("UserForFace() = (%q, %v), want (%q, true) after reassignment", userId, found, "user-2")
	}
}

func TestSQLiteStore_UsersForFaces(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.CreateUser(ctx, "col-1", "user-1"); err != nil {
		t.Fatalf("CreateUser() error: %v", err)
	}
	if err := s.AssignFace(ctx, "col-1", "face-1", "user-1"); err != nil {
		t.Fatalf("AssignFace(face-1) error: %v", err)
	}
	if err := s.AssignFace(ctx, "col-1", "face-2", "user-1"); err != nil {
		t.Fatalf("AssignFace(face-2) error: %v", err)
	}

	got, err := s.UsersForFaces(ctx, "col-1", []string{"face-1", "face-2", "face-unassigned"})
	if err != nil {
		t.Fatalf("UsersForFaces() error: %v", err)
	}

	want := map[string]string{"face-1": "user-1", "face-2": "user-1"}
	if len(got) != len(want) {
		t.Fatalf("UsersForFaces() = %v, want %v", got, want)
	}
	for faceId, userId := range want {
		if got[faceId] != userId {
			t.Errorf("UsersForFaces()[%s] = %q, want %q", faceId, got[faceId], userId)
		}
	}
	if _, ok := got["face-unassigned"]; ok {
		t.Errorf("UsersForFaces() included an unassigned face_id in the result")
	}
}

func TestSQLiteStore_UsersForFaces_EmptyInput(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	got, err := s.UsersForFaces(ctx, "col-1", nil)
	if err != nil {
		t.Fatalf("UsersForFaces(nil) error: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("UsersForFaces(nil) = %v, want empty", got)
	}
}

func TestSQLiteStore_FaceMetrics_NotRecordedYet(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	metrics, found, err := s.FaceMetrics(ctx, "col-1", "face-unknown")
	if err != nil {
		t.Fatalf("FaceMetrics() error: %v", err)
	}
	if found {
		t.Errorf("FaceMetrics() found = true, metrics = %+v, want not found", metrics)
	}
}

// TestSQLiteStore_RecordFaceMetrics_BeforeAnyUserAssignment confirms a
// face's metrics can be recorded (and read back) before it's ever assigned
// to a user — the whole point of keeping face_metrics independent of
// faces/users, since indexing happens before grouping.
func TestSQLiteStore_RecordFaceMetrics_BeforeAnyUserAssignment(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	want := FaceMetrics{Score: 72.5}
	if err := s.RecordFaceMetrics(ctx, "col-1", "face-1", want); err != nil {
		t.Fatalf("RecordFaceMetrics() error: %v", err)
	}

	got, found, err := s.FaceMetrics(ctx, "col-1", "face-1")
	if err != nil {
		t.Fatalf("FaceMetrics() error: %v", err)
	}
	if !found || got != want {
		t.Errorf("FaceMetrics() = (%+v, %v), want (%+v, true)", got, found, want)
	}
}

func TestSQLiteStore_RecordFaceMetrics_OverwritesPreviousMetrics(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.RecordFaceMetrics(ctx, "col-1", "face-1", FaceMetrics{Score: 50}); err != nil {
		t.Fatalf("RecordFaceMetrics() (1st) error: %v", err)
	}
	want := FaceMetrics{Score: 90, Occluded: true, OccludedConfidence: 88.4}
	if err := s.RecordFaceMetrics(ctx, "col-1", "face-1", want); err != nil {
		t.Fatalf("RecordFaceMetrics() (2nd) error: %v", err)
	}

	got, found, err := s.FaceMetrics(ctx, "col-1", "face-1")
	if err != nil || !found || got != want {
		t.Errorf("FaceMetrics() = (%+v, %v, %v), want (%+v, true, nil)", got, found, err, want)
	}
}

func TestSQLiteStore_Representative_NotSetYet(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.CreateUser(ctx, "col-1", "user-1"); err != nil {
		t.Fatalf("CreateUser() error: %v", err)
	}

	rep, found, err := s.Representative(ctx, "col-1", "user-1")
	if err != nil {
		t.Fatalf("Representative() error: %v", err)
	}
	if found {
		t.Errorf("Representative() found = true, rep = %+v, want not found", rep)
	}
}

func TestSQLiteStore_SetAndGetRepresentative(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	if err := s.CreateUser(ctx, "col-1", "user-1"); err != nil {
		t.Fatalf("CreateUser() error: %v", err)
	}
	want := Representative{FaceId: "face-1", Score: 83.25, Locked: true}
	if err := s.SetRepresentative(ctx, "col-1", "user-1", want); err != nil {
		t.Fatalf("SetRepresentative() error: %v", err)
	}

	got, found, err := s.Representative(ctx, "col-1", "user-1")
	if err != nil {
		t.Fatalf("Representative() error: %v", err)
	}
	if !found || got != want {
		t.Errorf("Representative() = (%+v, %v), want (%+v, true)", got, found, want)
	}
}

// TestSQLiteStore_SetRepresentative_UnknownUserFails confirms
// SetRepresentative refuses to silently no-op against a user_id that was
// never created — a caller bug (calling it before CreateUser) should
// surface as an error, not vanish.
func TestSQLiteStore_SetRepresentative_UnknownUserFails(t *testing.T) {
	s := openTestStore(t)
	ctx := context.Background()

	err := s.SetRepresentative(ctx, "col-1", "no-such-user", Representative{FaceId: "face-1", Score: 90, Locked: true})
	if err == nil {
		t.Fatal("SetRepresentative() for a nonexistent user succeeded, want an error")
	}
}

var _ Store = (*SQLiteStore)(nil)
