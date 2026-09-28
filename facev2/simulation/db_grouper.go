package simulation

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/HonchoLtd/aws_rekognition/facev2"
	"github.com/HonchoLtd/aws_rekognition/facev2/store"
	"github.com/google/uuid"
)

// This package is a reference implementation of the grouping /
// representative-selection layer that the facev2 SDK is designed to sit
// under — it is NOT part of the SDK itself. External services are expected
// to build their own equivalent on top of facev2.Face + their own
// datastore; see facev2/README.md for the algorithm this reference
// implementation follows, complete with sequence diagrams and integration
// guidance.

// GroupResult is the outcome of processing one batch of FaceIds: which
// UserId each face ended up under, and which UserIds were newly minted to
// get there.
type GroupResult struct {
	// FaceToUser maps every successfully processed FaceId in the batch to
	// the UserId it is now associated with.
	FaceToUser map[string]string
	// CreatedUsers lists the UserIds minted while processing this batch (as
	// opposed to ones that already existed from a previous batch).
	CreatedUsers []string
}

// DefaultRepresentativeThreshold is the combined score (see
// facev2.ComputeFaceScore) a face must reach to be locked in immediately as
// its user's representative thumbnail — used whenever
// DBGrouper.RepresentativeThreshold is left at its zero value.
//
// Note that ComputeFaceScore now folds pose, quality, AND occlusion into
// a single 0-100 number (see facev2.OccludedScore for how the FaceOccluded
// signal contributes), so a face fully occluded but otherwise perfect
// tops out around 60 — well below the default 70 — and will not lock the
// representative on its own. Faces that pass 70 have to look good on all
// three axes, not just one.
const DefaultRepresentativeThreshold = 70.0

// DBGrouper is the DB-backed face-clustering algorithm for facev2: instead
// of tracking which faces belong to the same person via AWS Rekognition's
// own User bookkeeping (CreateUser+AssociateFaces to build the group,
// SearchUsersByFaceId to query it), it mints the user_id itself — a plain
// locally-generated id, no AWS API involved — and keeps the resulting
// face_id -> user_id mapping in an external Store (facev2/store).
//
// The only Face method it ever calls is SearchFacesByFaceId, for raw
// face-to-face similarity ("which other indexed faces look like this one").
// CreateUser, AssociateFaces and SearchUsersByFaceId are never called —
// AWS Rekognition Users are not used at all in this flow.
//
// A single SearchFacesByFaceId call returns every look-alike of the queried
// face already in the collection, not just the queried face itself — so
// resolveFace persists that whole discovery to the Store in one go (see its
// doc comment), not just the one face it was asked to resolve. That's what
// keeps a group of N never-before-seen faces down to one SearchFacesByFaceId
// call for the whole group, instead of one per face.
//
// It also picks each user's representative thumbnail as faces are assigned
// (see considerRepresentative): the first face whose score passes
// RepresentativeThreshold is locked in and the search stops there, even if a
// later face would have scored higher; if none ever passes, the
// best-scoring face seen so far is kept as a fallback, always up to date.
// Because ComputeFaceScore folds pose, quality, AND occlusion into that
// one score, no separate occlusion tiering is needed here — a face's
// occlusion signal already influences whether it can pass the threshold or
// beat the current fallback. This only works for faces whose metrics were
// recorded via Store.RecordFaceMetrics before DBGrouper sees them
// (typically done by the caller right after facev2.Face.IndexFace, using
// its returned facev2.IndexedFace.Score) — a face with no recorded score
// is simply never considered, so the feature is opt-in.
type DBGrouper struct {
	Face  facev2.Face
	Store store.Store

	// NewUserId generates a UserId for a newly discovered person. Defaults
	// to uuid.New().String() when nil; tests can override it for
	// deterministic, readable UserIds.
	NewUserId func() string

	// RepresentativeThreshold overrides DefaultRepresentativeThreshold. Zero
	// (the field's default) means "use DefaultRepresentativeThreshold" — a
	// real caller would essentially never want an actual threshold of 0,
	// since that locks in whatever face is resolved first regardless of
	// quality.
	RepresentativeThreshold float64
}

// representativeThreshold returns the effective threshold: g's override, or
// DefaultRepresentativeThreshold if it wasn't set.
func (g *DBGrouper) representativeThreshold() float64 {
	if g.RepresentativeThreshold > 0 {
		return g.RepresentativeThreshold
	}
	return DefaultRepresentativeThreshold
}

// ProcessBatch runs the DB-backed clustering algorithm over faceIds, all
// assumed to belong to collectionId. It returns partial results (whatever
// succeeded) alongside a joined error of everything that failed, so a
// batcher can keep going rather than lose an entire batch to one bad face.
func (g *DBGrouper) ProcessBatch(ctx context.Context, collectionId string, faceIds []string) (GroupResult, error) {
	start := time.Now()
	slog.Info("db_grouper: processing batch", "collection_id", collectionId, "face_count", len(faceIds))

	result := GroupResult{FaceToUser: make(map[string]string, len(faceIds))}
	var errs []error

	for _, faceId := range faceIds {
		if _, done := result.FaceToUser[faceId]; done {
			// Already resolved earlier in this same batch — skip the
			// redundant round-trips.
			slog.Debug("db_grouper: face already resolved earlier in this batch, skipping", "collection_id", collectionId, "face_id", faceId, "user_id", result.FaceToUser[faceId])
			continue
		}
		if err := g.resolveFace(ctx, collectionId, faceId, &result); err != nil {
			errs = append(errs, fmt.Errorf("face %s: %w", faceId, err))
			slog.Error("db_grouper: failed to resolve face", "collection_id", collectionId, "face_id", faceId, "error", err)
		}
	}

	err := errors.Join(errs...)
	slog.Info("db_grouper: batch processed", "collection_id", collectionId,
		"face_count", len(faceIds), "resolved", len(result.FaceToUser),
		"users_created", len(result.CreatedUsers), "errors", len(errs), "duration_ms", time.Since(start).Milliseconds())

	return result, err
}

// resolveFace handles a single FaceId that hasn't been resolved yet this
// batch, mutating result in place with whatever it discovers/creates. The
// Store is written synchronously before resolveFace returns, so any later
// FaceId in this same batch (or a future batch) that turns out to be a
// look-alike of faceId will see this assignment via UsersForFaces.
func (g *DBGrouper) resolveFace(ctx context.Context, collectionId string, faceId string, result *GroupResult) error {
	slog.Debug("db_grouper: resolving face", "collection_id", collectionId, "face_id", faceId)

	// Idempotency: a face already assigned (from an earlier batch, or a
	// retried one) needs no more work.
	if userId, found, err := g.Store.UserForFace(ctx, collectionId, faceId); err != nil {
		return fmt.Errorf("lookup existing assignment: %w", err)
	} else if found {
		result.FaceToUser[faceId] = userId
		slog.Debug("db_grouper: face already assigned", "collection_id", collectionId, "face_id", faceId, "user_id", userId)
		return nil
	}

	similarFaces, err := g.Face.SearchFacesByFaceId(ctx, collectionId, faceId)
	if err != nil {
		return fmt.Errorf("search similar faces: %w", err)
	}
	slog.Debug("db_grouper: searched for similar faces", "collection_id", collectionId, "face_id", faceId, "similar_faces_found", len(similarFaces))

	matchedFaceIds := make([]string, 0, len(similarFaces))
	for _, match := range similarFaces {
		matchedFaceIds = append(matchedFaceIds, match.FaceId)
	}

	usersByFace, err := g.Store.UsersForFaces(ctx, collectionId, matchedFaceIds)
	if err != nil {
		return fmt.Errorf("batch lookup matched faces: %w", err)
	}

	// similarFaces is ordered by similarity, highest first (SearchFacesByFaceId's
	// contract), so the first matched face that already has an assigned
	// user gives us the highest-similarity owner when more than one
	// candidate user exists.
	userId := ""
	var matchedOn facev2.FaceMatch
	for _, match := range similarFaces {
		if uid, ok := usersByFace[match.FaceId]; ok {
			userId = uid
			matchedOn = match
			break
		}
	}

	createdNewUser := false
	if userId == "" {
		// No AWS call here: the user_id is minted locally and only ever
		// recorded in our own Store.
		userId = newUserIdOrDefault(g.NewUserId)
		slog.Info("db_grouper: no known person matched, minting new user", "collection_id", collectionId, "face_id", faceId, "user_id", userId, "similar_faces_found", len(similarFaces))
		if err := g.Store.CreateUser(ctx, collectionId, userId); err != nil {
			return fmt.Errorf("create user record: %w", err)
		}
		createdNewUser = true
	} else {
		slog.Info("db_grouper: face matched an already-known person", "collection_id", collectionId, "face_id", faceId, "user_id", userId, "matched_face_id", matchedOn.FaceId, "similarity", matchedOn.Similarity)
	}

	if err := g.Store.AssignFace(ctx, collectionId, faceId, userId); err != nil {
		return fmt.Errorf("assign face to user %s: %w", userId, err)
	}

	result.FaceToUser[faceId] = userId
	if createdNewUser {
		result.CreatedUsers = append(result.CreatedUsers, userId)
	}

	var softErrs []error
	if err := g.considerRepresentative(ctx, collectionId, userId, faceId); err != nil {
		softErrs = append(softErrs, fmt.Errorf("consider representative for %s: %w", faceId, err))
		slog.Error("db_grouper: failed to update representative candidate", "collection_id", collectionId, "face_id", faceId, "user_id", userId, "error", err)
	}

	// SearchFacesByFaceId already revealed every look-alike of faceId in
	// this one round-trip — persist that discovery for whichever of them
	// don't already have an assignment, so THEIR turn through resolveFace
	// (whether later in this same batch, or in some future one) short-
	// circuits at the UserForFace check above instead of repeating this
	// exact same SearchFacesByFaceId call. This is what keeps a group of N
	// never-before-seen faces to a single SearchFacesByFaceId call overall,
	// instead of N. A sibling that's already assigned to a DIFFERENT user is
	// left untouched — that pre-existing assignment is authoritative, not
	// this similarity edge.
	for _, match := range similarFaces {
		if _, alreadyAssigned := usersByFace[match.FaceId]; alreadyAssigned {
			continue
		}
		if err := g.Store.AssignFace(ctx, collectionId, match.FaceId, userId); err != nil {
			softErrs = append(softErrs, fmt.Errorf("assign sibling face %s: %w", match.FaceId, err))
			slog.Error("db_grouper: failed to pre-assign sibling face discovered via similarity search", "collection_id", collectionId, "face_id", match.FaceId, "user_id", userId, "error", err)
			continue
		}
		result.FaceToUser[match.FaceId] = userId
		if err := g.considerRepresentative(ctx, collectionId, userId, match.FaceId); err != nil {
			softErrs = append(softErrs, fmt.Errorf("consider representative for sibling %s: %w", match.FaceId, err))
			slog.Error("db_grouper: failed to update representative candidate for sibling face", "collection_id", collectionId, "face_id", match.FaceId, "user_id", userId, "error", err)
		}
	}

	return errors.Join(softErrs...)
}

// considerRepresentative updates userId's representative-thumbnail
// candidate with faceId, if warranted:
//
//   - if userId's representative is already locked, faceId is ignored — the
//     search already stopped.
//   - else if faceId's score passes representativeThreshold(), it's set as
//     the representative and locked — search over, even if some other,
//     not-yet-seen face would have scored higher.
//   - else if faceId scores higher than the current fallback (or there is
//     no fallback yet), it becomes the new (still unlocked) fallback.
//   - else nothing changes.
//
// The score compared here (FaceMetrics.Score) already folds pose, quality,
// AND occlusion into a single number via facev2.ComputeFaceScore, so no
// separate occlusion check is needed at this layer — a confidently-occluded
// face is naturally penalized by its lower score.
//
// A faceId with no metrics ever recorded via Store.RecordFaceMetrics (found
// == false) is silently skipped: representative selection is opt-in per
// face.
func (g *DBGrouper) considerRepresentative(ctx context.Context, collectionId string, userId string, faceId string) error {
	metrics, found, err := g.Store.FaceMetrics(ctx, collectionId, faceId)
	if err != nil {
		return fmt.Errorf("lookup face metrics: %w", err)
	}
	if !found {
		return nil
	}
	score := metrics.Score

	current, hasCurrent, err := g.Store.Representative(ctx, collectionId, userId)
	if err != nil {
		return fmt.Errorf("lookup current representative: %w", err)
	}
	if hasCurrent && current.Locked {
		return nil
	}

	passes := score >= g.representativeThreshold()
	if !passes && hasCurrent && score <= current.Score {
		return nil
	}

	if err := g.Store.SetRepresentative(ctx, collectionId, userId, store.Representative{
		FaceId: faceId,
		Score:  score,
		Locked: passes,
	}); err != nil {
		return fmt.Errorf("set representative: %w", err)
	}
	if passes {
		slog.Info("db_grouper: representative locked in", "collection_id", collectionId, "user_id", userId, "face_id", faceId, "score", score, "threshold", g.representativeThreshold())
	} else {
		slog.Debug("db_grouper: representative fallback updated", "collection_id", collectionId, "user_id", userId, "face_id", faceId, "score", score)
	}
	return nil
}

// newUserIdOrDefault generates a UserId via newUserId if provided, or a
// random UUID otherwise.
func newUserIdOrDefault(newUserId func() string) string {
	if newUserId != nil {
		return newUserId()
	}
	return uuid.New().String()
}
