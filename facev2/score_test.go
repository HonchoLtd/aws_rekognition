package facev2

import "testing"

// scoreEpsilon accounts for float32 rounding in the weight multiplications
// (e.g. 0.4*100 comes back as 39.999996, not exactly 40) — irrelevant for
// anything that actually consumes a 0-100 score, but exact equality in a
// test would be testing float32 precision, not the formula.
const scoreEpsilon = 0.01

func approxEqual(a, b float32) bool {
	d := a - b
	return d > -scoreEpsilon && d < scoreEpsilon
}

// TestOccludedScore pins down the quadratic mapping from AWS's
// {Value, Confidence} occlusion signal to a 0-100 sub-score. See
// OccludedScore's doc comment for the formula.
func TestOccludedScore(t *testing.T) {
	tests := []struct {
		name     string
		occluded FaceOccluded
		want     float32
	}{
		{
			name:     "zero value: no signal from AWS -> neutral 50",
			occluded: FaceOccluded{},
			want:     50,
		},
		{
			name:     "confidently not occluded -> full 100",
			occluded: FaceOccluded{Value: false, Confidence: 100},
			want:     100,
		},
		{
			name:     "confidently occluded -> full 0",
			occluded: FaceOccluded{Value: true, Confidence: 100},
			want:     0,
		},
		{
			name:     "half-confident not occluded -> quadratic pull, well short of 75 (linear)",
			occluded: FaceOccluded{Value: false, Confidence: 50},
			want:     62.5, // 50 + 50*(0.5)^2
		},
		{
			name:     "half-confident occluded -> quadratic pull, well short of 25 (linear)",
			occluded: FaceOccluded{Value: true, Confidence: 50},
			want:     37.5, // 50 - 50*(0.5)^2
		},
		{
			name:     "low-confidence occluded barely moves the needle",
			occluded: FaceOccluded{Value: true, Confidence: 25},
			want:     46.875, // 50 - 50*(0.25)^2
		},
		{
			name:     "low-confidence not occluded also barely moves the needle",
			occluded: FaceOccluded{Value: false, Confidence: 25},
			want:     53.125, // 50 + 50*(0.25)^2
		},
		{
			name:     "confidence above 100 clamps to 100 (defensive)",
			occluded: FaceOccluded{Value: true, Confidence: 150},
			want:     0,
		},
		{
			name:     "negative confidence clamps to 0 (defensive)",
			occluded: FaceOccluded{Value: true, Confidence: -20},
			want:     50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := OccludedScore(tt.occluded)
			if !approxEqual(got, tt.want) {
				t.Errorf("OccludedScore(%+v) = %v, want %v", tt.occluded, got, tt.want)
			}
		})
	}
}

func TestComputeFaceScore(t *testing.T) {
	tests := []struct {
		name     string
		pose     FacePose
		quality  FaceQuality
		occluded FaceOccluded
		want     float32
	}{
		{
			name:     "perfectly frontal, perfect quality, confidently not occluded",
			pose:     FacePose{Yaw: 0, Pitch: 0, Roll: 0},
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: false, Confidence: 100},
			want:     100, // 0.4*100 + 0.4*100 + 0.2*100
		},
		{
			name:     "perfectly frontal, perfect quality, confidently OCCLUDED loses the occlusion half",
			pose:     FacePose{Yaw: 0, Pitch: 0},
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: true, Confidence: 100},
			want:     60, // 0.4*100 + 0.4*0 + 0.2*100
		},
		{
			name:     "all zero-value inputs: perfectly frontal but no quality/occlusion signal",
			pose:     FacePose{},
			quality:  FaceQuality{},
			occluded: FaceOccluded{},
			want:     60, // 0.4*100 (pose) + 0.4*50 (occl neutral) + 0.2*0 (quality)
		},
		{
			name:     "yaw at the max cutoff zeroes the pose third",
			pose:     FacePose{Yaw: MaxYawDegrees, Pitch: 0},
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: false, Confidence: 100},
			want:     60, // 0.4*0 + 0.4*100 + 0.2*100
		},
		{
			name:     "yaw beyond the max cutoff still clamps to 0, not negative",
			pose:     FacePose{Yaw: MaxYawDegrees * 2, Pitch: 0},
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: false, Confidence: 100},
			want:     60,
		},
		{
			name:     "negative angles are treated the same as positive (absolute value)",
			pose:     FacePose{Yaw: -MaxYawDegrees / 2, Pitch: 0},
			quality:  FaceQuality{},
			occluded: FaceOccluded{},
			want:     40, // poseAxisScore(22.5,45)=50; 0.4*50 + 0.4*50 (occl neutral) + 0.2*0
		},
		{
			name:     "the worse of yaw/pitch drives the pose third, not the average",
			pose:     FacePose{Yaw: 0, Pitch: MaxPitchDegrees}, // perfect yaw can't rescue an extreme pitch
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: false, Confidence: 100},
			want:     60, // 0.4*min(100,0) + 0.4*100 + 0.2*100
		},
		{
			name:     "roll is ignored entirely, however extreme",
			pose:     FacePose{Yaw: 0, Pitch: 0, Roll: 179},
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: false, Confidence: 100},
			want:     100,
		},
		{
			name:     "quality favors sharpness over brightness",
			pose:     FacePose{},
			quality:  FaceQuality{Brightness: 100, Sharpness: 0},
			occluded: FaceOccluded{},
			want:     68, // 0.4*100 + 0.4*50 + 0.2*(0.4*100 + 0.6*0)=40+20+8
		},
		{
			name:     "low-confidence occluded=true is close to neutral, so the score barely dips",
			pose:     FacePose{},
			quality:  FaceQuality{Brightness: 100, Sharpness: 100},
			occluded: FaceOccluded{Value: true, Confidence: 25},
			want:     78.75, // 0.4*100 + 0.4*46.875 + 0.2*100
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeFaceScore(tt.pose, tt.quality, tt.occluded)
			if !approxEqual(got, tt.want) {
				t.Errorf("ComputeFaceScore(%+v, %+v, %+v) = %v, want %v", tt.pose, tt.quality, tt.occluded, got, tt.want)
			}
		})
	}
}
