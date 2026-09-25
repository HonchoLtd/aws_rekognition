package main

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png" // enable PNG decoding
	"math"

	"github.com/HonchoLtd/aws_rekognition/facev2"
)

// clamp clamps v between [lo, hi].
func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// cropFace crops img to facev2's normalized BoundingBox (expanded by scale
// around its center, then clamped to the image bounds), and encodes the
// result as JPEG.
//
// Ported from face.cropWithBoundingBoxScaled/encodeJPEG — this tool
// deliberately doesn't apply AWS's OrientationCorrection the way the `face`
// package does (see indexAndCrop in main.go for why): input images are
// assumed to already be right-side up. If crops come out rotated for your
// sample set, that's the reason — see the `face` package's
// rotateAccordingToOrientation for how to add it back.
func cropFace(img image.Image, bbox facev2.FaceBoundingBox, scale float64) ([]byte, error) {
	if bbox.Width <= 0 || bbox.Height <= 0 {
		return nil, fmt.Errorf("incomplete bounding box: %+v", bbox)
	}
	if scale <= 0 {
		scale = 1.0
	}

	b := img.Bounds()
	W, H := b.Dx(), b.Dy()

	left := float64(bbox.Left) * float64(W)
	top := float64(bbox.Top) * float64(H)
	w := float64(bbox.Width) * float64(W)
	h := float64(bbox.Height) * float64(H)

	cx := left + w/2.0
	cy := top + h/2.0

	newW := w * scale
	newH := h * scale

	x0f := cx - newW/2.0
	y0f := cy - newH/2.0
	x1f := cx + newW/2.0
	y1f := cy + newH/2.0

	x0 := clamp(int(math.Round(x0f)), 0, W)
	y0 := clamp(int(math.Round(y0f)), 0, H)
	x1 := clamp(int(math.Round(x1f)), 0, W)
	y1 := clamp(int(math.Round(y1f)), 0, H)

	if x1 <= x0 || y1 <= y0 {
		// Fall back to the original (unscaled) bbox, clamped.
		x0 = clamp(int(math.Round(left)), 0, W)
		y0 = clamp(int(math.Round(top)), 0, H)
		x1 = clamp(int(math.Round(left+w)), 0, W)
		y1 = clamp(int(math.Round(top+h)), 0, H)
		if x1 <= x0 || y1 <= y0 {
			return nil, fmt.Errorf("invalid crop rectangle even after fallback: bbox=%+v", bbox)
		}
	}

	srcRect := image.Rect(x0, y0, x1, y1)
	dst := image.NewRGBA(image.Rect(0, 0, srcRect.Dx(), srcRect.Dy()))
	draw.Draw(dst, dst.Bounds(), img, srcRect.Min, draw.Src)

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: 90}); err != nil {
		return nil, fmt.Errorf("encode cropped face: %w", err)
	}
	return buf.Bytes(), nil
}
