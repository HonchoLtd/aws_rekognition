// Package store is a reference-implementation datastore for the grouping /
// representative-selection layer built on top of the facev2 SDK — it is
// NOT part of the SDK itself. External services are expected to write
// their own datastore (Postgres, DynamoDB, etc.) that satisfies the same
// contract; see facev2/README.md for the schema and the operations an
// external store needs to support.
//
// It persists the face_id -> user_id grouping that
// facev2/simulation's DBGrouper uses in place of AWS Rekognition's own User
// associations (AssociateFaces to build a group, SearchUsersByFaceId to
// query it). SearchFacesByFaceId's raw face-to-face similarity still comes
// from AWS Rekognition; whose "user" that similar face belongs to comes from
// here instead.
//
// It also persists two things DBGrouper's representative-thumbnail
// selection needs: each face's metrics (FaceMetrics — its
// facev2.ComputeFaceScore Score, plus the raw FaceOccluded signal that
// score already folds in, recorded at index time before the face is
// necessarily resolved to any user) and each user's current
// representative-thumbnail candidate.
//
// The demo/reference implementation, SQLiteStore, is backed by
// modernc.org/sqlite — a pure-Go SQLite driver, so no C toolchain/cgo is
// needed to build or run it. Nothing about the Store interface is
// SQLite-specific, so a production deployment can swap in a different
// backing store (Postgres, DynamoDB, ...) without touching
// facev2/simulation.
package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// FaceMetrics is the per-face metadata recorded at index time — before the
// face is necessarily resolved to a user_id — that downstream steps read
// back later.
type FaceMetrics struct {
	// Score is facev2.ComputeFaceScore's 0-100 "how good a representative
	// thumbnail would this be" judgment; it drives DBGrouper's
	// representative selection (see Representative below). It already folds
	// in pose, quality, and occlusion — the Occluded/OccludedConfidence
	// fields below are the raw AWS FaceOccluded signal Score consumed,
	// carried through purely for callers that want to surface it
	// themselves (e.g. show it in a crop's filename for debuggability).
	Score float64
	// Occluded is AWS's judgment (facev2.IndexedFace.Occluded.Value) on
	// whether the face is occluded — eyes/nose/mouth partially captured or
	// covered by a mask, sunglasses, a hand, a phone, etc. Populated only
	// when IndexFaces is asked for the FACE_OCCLUDED attribute (facev2's
	// IndexFace variants do). Purely informational; the real influence of
	// occlusion on selection is already inside Score via
	// facev2.OccludedScore.
	Occluded bool
	// OccludedConfidence is how confident AWS is in Occluded (0-100) —
	// facev2.IndexedFace.Occluded.Confidence. See Occluded's doc comment.
	OccludedConfidence float64
}

// Representative is a user's current representative-thumbnail candidate:
// the best-scoring face seen so far, and whether the search for an even
// better one has stopped. See facev2/simulation.DBGrouper's doc comment
// for the selection rule that produces this.
type Representative struct {
	FaceId string
	Score  float64
	// Locked reports whether FaceId's score already passed the selection
	// threshold — once true, it's final: a later, even better-scoring face
	// deliberately does NOT replace it.
	Locked bool
}

// Store is the persistence boundary DBGrouper depends on. Every method is
// scoped to a collectionId, mirroring how Rekognition itself scopes FaceIds
// and UserIds to a single collection — the same face_id/user_id values in a
// different collection are unrelated.
type Store interface {
	// CreateUser registers userId (a locally-minted id — no AWS API call
	// involved) within collectionId. Creating a userId that already exists
	// is a no-op.
	CreateUser(ctx context.Context, collectionId string, userId string) error

	// AssignFace records that faceId belongs to userId within collectionId.
	// Re-assigning the same faceId to the same userId is a no-op; assigning
	// it to a different userId overwrites the previous assignment (last
	// write wins) — call UserForFace first if that distinction matters to
	// the caller.
	AssignFace(ctx context.Context, collectionId string, faceId string, userId string) error

	// UserForFace returns the userId faceId is already assigned to, if any.
	// found is false (with a nil error) when faceId has no assignment yet.
	UserForFace(ctx context.Context, collectionId string, faceId string) (userId string, found bool, err error)

	// UsersForFaces is UserForFace batched over multiple faceIds in a single
	// round-trip. The returned map contains only the faceIds that already
	// have an assignment — a missing key means "not yet assigned", not an
	// error. An empty/nil faceIds returns an empty map without touching the
	// store.
	UsersForFaces(ctx context.Context, collectionId string, faceIds []string) (map[string]string, error)

	// RecordFaceMetrics stores faceId's metrics, independent of any user
	// assignment — they're known at index time, before the face is
	// necessarily resolved to a user. Overwrites any previously recorded
	// metrics for the same faceId.
	RecordFaceMetrics(ctx context.Context, collectionId string, faceId string, metrics FaceMetrics) error

	// FaceMetrics returns faceId's previously recorded metrics. found is
	// false when none were ever recorded for it (e.g. the caller never
	// wired metrics recording into its indexing step) — callers should treat
	// that as "this feature wasn't used for this face", not an error.
	FaceMetrics(ctx context.Context, collectionId string, faceId string) (metrics FaceMetrics, found bool, err error)

	// Representative returns userId's current representative-thumbnail
	// candidate. found is false when userId has no candidate yet (e.g. none
	// of its faces so far had a recorded score).
	Representative(ctx context.Context, collectionId string, userId string) (rep Representative, found bool, err error)

	// SetRepresentative sets userId's representative-thumbnail candidate.
	// userId must already exist (via CreateUser).
	SetRepresentative(ctx context.Context, collectionId string, userId string, rep Representative) error

	// Close releases the store's underlying resources (e.g. the database
	// connection). Safe to call once when the store is no longer needed.
	Close() error
}

// SQLiteStore is a Store backed by a local SQLite database file (or
// ":memory:" for an ephemeral, process-local one — handy for tests).
type SQLiteStore struct {
	db *sql.DB
}

// Open opens (creating if necessary) a SQLite database at path and ensures
// its schema exists. Callers must Close the returned SQLiteStore when done.
//
// SQLite only tolerates one writer at a time; rather than let concurrent
// flushes across different collections fight over SQLITE_BUSY, the
// connection pool is capped at one connection, so all access is naturally
// serialized. That's the right trade-off for this store's write volume (a
// handful of small writes per indexed face) — it is not meant to be a
// high-throughput store.
func Open(path string) (*SQLiteStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("store: open sqlite db %q: %w", path, err)
	}
	db.SetMaxOpenConns(1)

	if _, err := db.Exec(`PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000; PRAGMA foreign_keys = ON;`); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: configure sqlite pragmas: %w", err)
	}

	s := &SQLiteStore{db: db}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// migrate creates the store's schema if it doesn't exist yet. There's
// deliberately no migration framework here — this is a demo-scoped store
// with one, stable schema. (SQLite's ALTER TABLE has no ADD COLUMN IF NOT
// EXISTS, and this schema evolves in place — most recently, the
// representative_acceptable and face_metrics.confidence columns were
// dropped when occlusion was folded directly into the score. Any .db file
// from a previous shape needs to be deleted and recreated rather than
// upgraded.)
func (s *SQLiteStore) migrate() error {
	const schema = `
CREATE TABLE IF NOT EXISTS users (
	collection_id          TEXT NOT NULL,
	user_id                TEXT NOT NULL,
	representative_face_id TEXT,
	representative_score   REAL,
	representative_locked  INTEGER NOT NULL DEFAULT 0,
	created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (collection_id, user_id)
);

CREATE TABLE IF NOT EXISTS faces (
	collection_id TEXT NOT NULL,
	face_id       TEXT NOT NULL,
	user_id       TEXT NOT NULL,
	created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (collection_id, face_id),
	FOREIGN KEY (collection_id, user_id) REFERENCES users (collection_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_faces_user ON faces (collection_id, user_id);

-- Independent of user assignment: a face's metrics are known at index
-- time, before it's necessarily resolved to a user_id. The raw
-- FaceOccluded fields are informational (already folded into score); the
-- score column is what DBGrouper actually compares.
CREATE TABLE IF NOT EXISTS face_metrics (
	collection_id       TEXT NOT NULL,
	face_id             TEXT NOT NULL,
	score               REAL NOT NULL,
	occluded            INTEGER NOT NULL DEFAULT 0,
	occluded_confidence REAL NOT NULL DEFAULT 0,
	created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (collection_id, face_id)
);
`
	if _, err := s.db.Exec(schema); err != nil {
		return fmt.Errorf("store: migrate schema: %w", err)
	}
	return nil
}

// CreateUser implements Store.
func (s *SQLiteStore) CreateUser(ctx context.Context, collectionId string, userId string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO users (collection_id, user_id) VALUES (?, ?)
		 ON CONFLICT (collection_id, user_id) DO NOTHING`,
		collectionId, userId,
	)
	if err != nil {
		return fmt.Errorf("store: create user %s/%s: %w", collectionId, userId, err)
	}
	return nil
}

// AssignFace implements Store.
func (s *SQLiteStore) AssignFace(ctx context.Context, collectionId string, faceId string, userId string) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO faces (collection_id, face_id, user_id) VALUES (?, ?, ?)
		 ON CONFLICT (collection_id, face_id) DO UPDATE SET user_id = excluded.user_id`,
		collectionId, faceId, userId,
	)
	if err != nil {
		return fmt.Errorf("store: assign face %s/%s to user %s: %w", collectionId, faceId, userId, err)
	}
	return nil
}

// UserForFace implements Store.
func (s *SQLiteStore) UserForFace(ctx context.Context, collectionId string, faceId string) (string, bool, error) {
	var userId string
	err := s.db.QueryRowContext(ctx,
		`SELECT user_id FROM faces WHERE collection_id = ? AND face_id = ?`,
		collectionId, faceId,
	).Scan(&userId)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("store: lookup user for face %s/%s: %w", collectionId, faceId, err)
	}
	return userId, true, nil
}

// UsersForFaces implements Store.
func (s *SQLiteStore) UsersForFaces(ctx context.Context, collectionId string, faceIds []string) (map[string]string, error) {
	result := make(map[string]string, len(faceIds))
	if len(faceIds) == 0 {
		return result, nil
	}

	placeholders := make([]string, len(faceIds))
	args := make([]any, 0, len(faceIds)+1)
	args = append(args, collectionId)
	for i, faceId := range faceIds {
		placeholders[i] = "?"
		args = append(args, faceId)
	}

	query := fmt.Sprintf(
		`SELECT face_id, user_id FROM faces WHERE collection_id = ? AND face_id IN (%s)`,
		strings.Join(placeholders, ","),
	)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: batch lookup users for %d face(s): %w", len(faceIds), err)
	}
	defer rows.Close()

	for rows.Next() {
		var faceId, userId string
		if err := rows.Scan(&faceId, &userId); err != nil {
			return nil, fmt.Errorf("store: scan batch lookup row: %w", err)
		}
		result[faceId] = userId
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: iterate batch lookup rows: %w", err)
	}
	return result, nil
}

// RecordFaceMetrics implements Store.
func (s *SQLiteStore) RecordFaceMetrics(ctx context.Context, collectionId string, faceId string, metrics FaceMetrics) error {
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO face_metrics (collection_id, face_id, score, occluded, occluded_confidence)
		 VALUES (?, ?, ?, ?, ?)
		 ON CONFLICT (collection_id, face_id) DO UPDATE SET
			 score = excluded.score,
			 occluded = excluded.occluded,
			 occluded_confidence = excluded.occluded_confidence`,
		collectionId, faceId, metrics.Score, metrics.Occluded, metrics.OccludedConfidence,
	)
	if err != nil {
		return fmt.Errorf("store: record metrics for face %s/%s: %w", collectionId, faceId, err)
	}
	return nil
}

// FaceMetrics implements Store.
func (s *SQLiteStore) FaceMetrics(ctx context.Context, collectionId string, faceId string) (FaceMetrics, bool, error) {
	var metrics FaceMetrics
	err := s.db.QueryRowContext(ctx,
		`SELECT score, occluded, occluded_confidence
		 FROM face_metrics WHERE collection_id = ? AND face_id = ?`,
		collectionId, faceId,
	).Scan(&metrics.Score, &metrics.Occluded, &metrics.OccludedConfidence)
	if errors.Is(err, sql.ErrNoRows) {
		return FaceMetrics{}, false, nil
	}
	if err != nil {
		return FaceMetrics{}, false, fmt.Errorf("store: lookup metrics for face %s/%s: %w", collectionId, faceId, err)
	}
	return metrics, true, nil
}

// Representative implements Store.
func (s *SQLiteStore) Representative(ctx context.Context, collectionId string, userId string) (Representative, bool, error) {
	var (
		faceId sql.NullString
		score  sql.NullFloat64
		locked bool
	)
	err := s.db.QueryRowContext(ctx,
		`SELECT representative_face_id, representative_score, representative_locked
		 FROM users WHERE collection_id = ? AND user_id = ?`,
		collectionId, userId,
	).Scan(&faceId, &score, &locked)
	if errors.Is(err, sql.ErrNoRows) {
		return Representative{}, false, nil
	}
	if err != nil {
		return Representative{}, false, fmt.Errorf("store: lookup representative for user %s/%s: %w", collectionId, userId, err)
	}
	if !faceId.Valid {
		return Representative{}, false, nil
	}
	return Representative{FaceId: faceId.String, Score: score.Float64, Locked: locked}, true, nil
}

// SetRepresentative implements Store.
func (s *SQLiteStore) SetRepresentative(ctx context.Context, collectionId string, userId string, rep Representative) error {
	res, err := s.db.ExecContext(ctx,
		`UPDATE users SET representative_face_id = ?, representative_score = ?, representative_locked = ?
		 WHERE collection_id = ? AND user_id = ?`,
		rep.FaceId, rep.Score, rep.Locked, collectionId, userId,
	)
	if err != nil {
		return fmt.Errorf("store: set representative for user %s/%s: %w", collectionId, userId, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: set representative for user %s/%s: %w", collectionId, userId, err)
	}
	if n == 0 {
		return fmt.Errorf("store: set representative for user %s/%s: user does not exist (call CreateUser first)", collectionId, userId)
	}
	return nil
}

// Close implements Store.
func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

var _ Store = (*SQLiteStore)(nil)
