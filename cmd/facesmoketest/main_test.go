package main

import (
	"context"
	"testing"

	"github.com/HonchoLtd/aws_rekognition/facev2/store"
)

// TestCropFileName_WithMetrics confirms the score/occluded prefix format
// (each numeric field zero-padded to a fixed width, so filenames sort by
// score) matches what a human browsing the output folder is meant to read
// at a glance.
func TestCropFileName_WithMetrics(t *testing.T) {
	s, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open(:memory:) failed: %v", err)
	}
	defer s.Close()
	ctx := context.Background()

	const collectionId = "col-1"
	const faceId = "face-1234567890"
	if err := s.RecordFaceMetrics(ctx, collectionId, faceId, store.FaceMetrics{Score: 85.3, Occluded: false, OccludedConfidence: 5.2}); err != nil {
		t.Fatalf("RecordFaceMetrics() error: %v", err)
	}

	got := cropFileName(ctx, s, collectionId, faceId, "img001")
	want := "score085.3_occlN005.2_img001_face-123.jpg" // shortId truncates the face_id to 8 chars
	if got != want {
		t.Errorf("cropFileName() = %q, want %q", got, want)
	}
}

// TestCropFileName_OccludedTrue confirms an occluded face gets an "occlY"
// flag in its filename (as opposed to "occlN") so it's visually distinct
// from a non-occluded one at a glance.
func TestCropFileName_OccludedTrue(t *testing.T) {
	s, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open(:memory:) failed: %v", err)
	}
	defer s.Close()
	ctx := context.Background()

	const collectionId = "col-1"
	const faceId = "face-abcdef1234"
	if err := s.RecordFaceMetrics(ctx, collectionId, faceId, store.FaceMetrics{Score: 70.1, Occluded: true, OccludedConfidence: 88.4}); err != nil {
		t.Fatalf("RecordFaceMetrics() error: %v", err)
	}

	got := cropFileName(ctx, s, collectionId, faceId, "img003")
	want := "score070.1_occlY088.4_img003_face-abc.jpg"
	if got != want {
		t.Errorf("cropFileName() = %q, want %q", got, want)
	}
}

// TestCropFileName_NoMetricsRecorded confirms a face with no recorded
// metrics (e.g. scoring wasn't wired up, or a stale .db predating this
// feature) falls back to the plain filename instead of a malformed prefix.
func TestCropFileName_NoMetricsRecorded(t *testing.T) {
	s, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open(:memory:) failed: %v", err)
	}
	defer s.Close()
	ctx := context.Background()

	got := cropFileName(ctx, s, "col-1", "face-unscored", "img002")
	want := "img002_face-uns.jpg"
	if got != want {
		t.Errorf("cropFileName() = %q, want %q", got, want)
	}
}

// TestCropFileName_ScoresSortAlphabeticallyLikeNumbers confirms the
// zero-padding actually achieves its purpose: a lower score's filename must
// sort before a higher score's, lexicographically, matching numeric order.
func TestCropFileName_ScoresSortAlphabeticallyLikeNumbers(t *testing.T) {
	s, err := store.Open(":memory:")
	if err != nil {
		t.Fatalf("store.Open(:memory:) failed: %v", err)
	}
	defer s.Close()
	ctx := context.Background()

	const collectionId = "col-1"
	if err := s.RecordFaceMetrics(ctx, collectionId, "face-low", store.FaceMetrics{Score: 7.5}); err != nil {
		t.Fatalf("RecordFaceMetrics(low) error: %v", err)
	}
	if err := s.RecordFaceMetrics(ctx, collectionId, "face-high", store.FaceMetrics{Score: 100}); err != nil {
		t.Fatalf("RecordFaceMetrics(high) error: %v", err)
	}

	low := cropFileName(ctx, s, collectionId, "face-low", "img")
	high := cropFileName(ctx, s, collectionId, "face-high", "img")
	if !(low < high) {
		t.Errorf("low-score filename %q does not sort before high-score filename %q", low, high)
	}
}
