package facev2

// This file scores how good a candidate a single indexed face is for use as
// its person's representative thumbnail, from three signals AWS Rekognition's
// IndexFaces API returns per face: Pose and Quality (part of the DEFAULT
// attribute set — no extra request parameters needed) and FaceOccluded
// (returned only when FACE_OCCLUDED is asked for explicitly; facev2's
// IndexFace does so). See facev2/simulation.DBGrouper for how the resulting
// score is used to actually pick a representative once faces are grouped
// into users.

// Weights and axis limits used by ComputeFaceScore. Exported so the exact
// derivation of the 0-100 result is visible (and tunable) to callers. The
// three sub-score weights are chosen to sum to 1, so the final score stays
// in [0, 100] without any post-hoc clamping.
const (
	// PoseScoreWeight, OccludedScoreWeight, and QualityScoreWeight split
	// ComputeFaceScore's final result three ways: 40% frontality, 40%
	// occlusion, 20% raw image quality. Pose and occlusion are the two
	// things a viewer notices first about a thumbnail; sharpness/exposure
	// matter, but a slightly soft photo is a much better thumbnail than a
	// sharp side-profile or a sharp masked face.
	PoseScoreWeight     = 0.4
	OccludedScoreWeight = 0.4
	QualityScoreWeight  = 0.2

	// MaxYawDegrees and MaxPitchDegrees are the out-of-plane rotation (i.e.
	// turning away from the camera) beyond which a face scores 0 on that
	// axis; within them, the score falls off linearly to 0. Roll (in-plane
	// head tilt) is deliberately excluded from frontality: tilting your head
	// doesn't hide your face or turn it away from the camera.
	MaxYawDegrees   = 45.0
	MaxPitchDegrees = 45.0

	// BrightnessWeight and SharpnessWeight split the quality half of the
	// score. Weighted toward sharpness: a blurry photo makes an unusable
	// thumbnail in a way mild under/over-exposure usually doesn't.
	BrightnessWeight = 0.4
	SharpnessWeight  = 0.6
)

// FacePose mirrors the relevant fields of AWS Rekognition's Pose type,
// returned as part of IndexFaces' FaceRecords[].FaceDetail.
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_Pose.html
type FacePose struct {
	Yaw   float32 `json:"yaw"`
	Pitch float32 `json:"pitch"`
	Roll  float32 `json:"roll"`
}

// FaceQuality mirrors the relevant fields of AWS Rekognition's ImageQuality
// type, returned as part of IndexFaces' FaceRecords[].FaceDetail.
// See: https://docs.aws.amazon.com/rekognition/latest/APIReference/API_ImageQuality.html
type FaceQuality struct {
	Brightness float32 `json:"brightness"`
	Sharpness  float32 `json:"sharpness"`
}

// ComputeFaceScore combines pose, occlusion, and quality into a single
// 0-100 "how good a representative thumbnail would this face make" score:
//
//   - Frontality (pose): each of Yaw/Pitch is scored on how close to 0 it
//     is, falling off linearly to 0 at Max{Yaw,Pitch}Degrees, and the WORSE
//     of the two axes is used — a face isn't frontal unless both are in
//     range, so a perfect yaw can't compensate for an extreme pitch. Roll
//     is ignored entirely (see MaxYawDegrees' doc comment).
//   - Occlusion: see OccludedScore — folds AWS's FaceOccluded {Value,
//     Confidence} pair into a 0-100 signal that rewards confidently-
//     not-occluded faces, penalizes confidently-occluded ones, and treats
//     low-confidence flags as near-neutral so an uncertain AWS call doesn't
//     tank the score.
//   - Quality: a weighted average of AWS's Brightness and Sharpness, both
//     already 0-100 with higher meaning better.
//
// The three parts are then blended via
// PoseScoreWeight/OccludedScoreWeight/QualityScoreWeight (currently
// 40/40/20). All three sub-scores are clamped to [0, 100] before the
// weighted sum, so the final result is guaranteed to be in [0, 100] too.
func ComputeFaceScore(pose FacePose, quality FaceQuality, occluded FaceOccluded) float32 {
	poseScore := min(poseAxisScore(pose.Yaw, MaxYawDegrees), poseAxisScore(pose.Pitch, MaxPitchDegrees))
	qualityScore := max(0, min(100, BrightnessWeight*quality.Brightness+SharpnessWeight*quality.Sharpness))
	occludedScore := OccludedScore(occluded)
	return PoseScoreWeight*poseScore + OccludedScoreWeight*occludedScore + QualityScoreWeight*qualityScore
}

// OccludedScore turns AWS's FaceOccluded {Value, Confidence} pair into a
// 0-100 signal (higher = better for a thumbnail) using a quadratic offset
// from a neutral 50 midpoint:
//
//	c = Confidence / 100           // 0..1
//	offset = 50 * c*c              // 0..50
//	Value == false: 50 + offset    // -> [50..100], "AWS thinks it's not occluded"
//	Value == true:  50 - offset    // -> [0..50],   "AWS thinks it is occluded"
//
// The quadratic curve makes low-confidence flags in either direction barely
// move the needle (Value=true Conf=25 -> 46.9, Value=false Conf=25 -> 53.1),
// while high-confidence flags move the full ±50 (Value=true Conf=100 -> 0,
// Value=false Conf=100 -> 100). This deliberately treats a low-confidence
// occluded=true not as "occluded", but as "AWS isn't sure" — since AWS
// itself isn't confident, we shouldn't be either.
//
// A zero-value FaceOccluded (which is also what arrives when AWS didn't
// return one at all, e.g. FACE_OCCLUDED wasn't in DetectionAttributes) maps
// to exactly 50 — a fully neutral contribution to ComputeFaceScore.
func OccludedScore(occluded FaceOccluded) float32 {
	c := occluded.Confidence / 100.0
	if c < 0 {
		c = 0
	}
	if c > 1 {
		c = 1
	}
	offset := 50.0 * c * c
	if occluded.Value {
		return 50.0 - offset
	}
	return 50.0 + offset
}

// poseAxisScore scores a single pose axis: 100 at angle=0, falling off
// linearly to 0 at |angle|=maxDegrees, and staying 0 beyond it.
func poseAxisScore(angleDegrees float32, maxDegrees float32) float32 {
	if angleDegrees < 0 {
		angleDegrees = -angleDegrees
	}
	if maxDegrees <= 0 {
		return 0
	}
	return max(0, min(100, 100*(1-angleDegrees/maxDegrees)))
}
