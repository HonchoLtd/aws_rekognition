package modelembedding

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"testing"
)

type byteOrder interface {
	binary.ByteOrder
	binary.AppendByteOrder
}

// withExifOrientation returns jpegData with an APP1 EXIF segment carrying
// the given Orientation inserted right after the SOI marker.
func withExifOrientation(t *testing.T, jpegData []byte, orientation uint16, order byteOrder) []byte {
	t.Helper()
	tiff := make([]byte, 0, 26)
	if order == binary.LittleEndian {
		tiff = append(tiff, 'I', 'I')
	} else {
		tiff = append(tiff, 'M', 'M')
	}
	tiff = order.AppendUint16(tiff, 42)
	tiff = order.AppendUint32(tiff, 8)      // IFD0 offset
	tiff = order.AppendUint16(tiff, 1)      // one entry
	tiff = order.AppendUint16(tiff, 0x0112) // Orientation
	tiff = order.AppendUint16(tiff, 3)      // SHORT
	tiff = order.AppendUint32(tiff, 1)      // count
	tiff = order.AppendUint16(tiff, orientation)
	tiff = append(tiff, 0, 0)          // value padding
	tiff = order.AppendUint32(tiff, 0) // no next IFD

	payload := append([]byte("Exif\x00\x00"), tiff...)
	segment := []byte{0xFF, 0xE1}
	segment = binary.BigEndian.AppendUint16(segment, uint16(len(payload)+2))
	segment = append(segment, payload...)

	out := append([]byte{}, jpegData[:2]...)
	out = append(out, segment...)
	return append(out, jpegData[2:]...)
}

// leftRedRightBlueJPEG is a width x height JPEG whose left half is red and
// right half blue: big solid blocks survive JPEG compression, so tests can
// tell which way the image was rotated.
func leftRedRightBlueJPEG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if x < width/2 {
				img.Set(x, y, color.RGBA{R: 255, A: 255})
			} else {
				img.Set(x, y, color.RGBA{B: 255, A: 255})
			}
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func isRed(c color.Color) bool {
	r, g, b, _ := c.RGBA()
	return r > 0xC000 && g < 0x4000 && b < 0x4000
}

func isBlue(c color.Color) bool {
	r, g, b, _ := c.RGBA()
	return b > 0xC000 && r < 0x4000 && g < 0x4000
}

func TestJpegOrientation(t *testing.T) {
	plain := leftRedRightBlueJPEG(t, 8, 8)
	if got := jpegOrientation(plain); got != 1 {
		t.Errorf("no EXIF: orientation = %d, want 1", got)
	}
	for _, order := range []byteOrder{binary.BigEndian, binary.LittleEndian} {
		for o := uint16(1); o <= 8; o++ {
			if got := jpegOrientation(withExifOrientation(t, plain, o, order)); got != int(o) {
				t.Errorf("%v orientation %d: got %d", order, o, got)
			}
		}
	}
	if got := jpegOrientation(withExifOrientation(t, plain, 9, binary.BigEndian)); got != 1 {
		t.Errorf("out-of-range orientation: got %d, want 1", got)
	}
	if got := jpegOrientation([]byte{0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF}); got != 1 {
		t.Errorf("truncated segment: got %d, want 1", got)
	}
	if got := jpegOrientation([]byte("not a jpeg")); got != 1 {
		t.Errorf("non-JPEG: got %d, want 1", got)
	}
}

func TestApplyOrientation(t *testing.T) {
	// 3x2 source with a distinct value per pixel:
	//   0 1 2
	//   3 4 5
	src := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for i := 0; i < 6; i++ {
		src.Pix[i*4] = uint8(i)
	}
	tests := map[int][]uint8{
		1: {0, 1, 2, 3, 4, 5},
		2: {2, 1, 0, 5, 4, 3},
		3: {5, 4, 3, 2, 1, 0},
		4: {3, 4, 5, 0, 1, 2},
		5: {0, 3, 1, 4, 2, 5},
		6: {3, 0, 4, 1, 5, 2},
		7: {5, 2, 4, 1, 3, 0},
		8: {2, 5, 1, 4, 0, 3},
	}
	for orientation, want := range tests {
		dst := applyOrientation(src, orientation)
		wantW, wantH := 3, 2
		if orientation >= 5 {
			wantW, wantH = 2, 3
		}
		if dst.Bounds().Dx() != wantW || dst.Bounds().Dy() != wantH {
			t.Errorf("orientation %d: size %v, want %dx%d", orientation, dst.Bounds().Size(), wantW, wantH)
			continue
		}
		got := make([]uint8, 6)
		for i := range got {
			got[i] = dst.Pix[i*4]
		}
		if !bytes.Equal(got, want) {
			t.Errorf("orientation %d: pixels %v, want %v", orientation, got, want)
		}
	}
}

func TestPreprocessImage_AppliesExifRotation(t *testing.T) {
	rotated := withExifOrientation(t, leftRedRightBlueJPEG(t, 40, 20), 6, binary.BigEndian)

	out, err := PreprocessImage(rotated, 1024)
	if err != nil {
		t.Fatalf("PreprocessImage() error = %v", err)
	}
	img, err := jpeg.Decode(bytes.NewReader(out))
	if err != nil {
		t.Fatalf("output isn't a JPEG: %v", err)
	}
	if img.Bounds().Dx() != 20 || img.Bounds().Dy() != 40 {
		t.Fatalf("size = %v, want 20x40 after a 90 degree rotation", img.Bounds().Size())
	}
	if !isRed(img.At(10, 5)) || !isBlue(img.At(10, 35)) {
		t.Errorf("rotated 90 clockwise, the red left half should be on top and blue at the bottom; got top=%v bottom=%v", img.At(10, 5), img.At(10, 35))
	}
}

func TestPreprocessImage_ShrinksAndFlattensTransparency(t *testing.T) {
	transparent := pngBytes(t, 300, 150, color.NRGBA{})

	out, err := PreprocessImage(transparent, 100)
	if err != nil {
		t.Fatalf("PreprocessImage() error = %v", err)
	}
	img, format, err := image.Decode(bytes.NewReader(out))
	if err != nil || format != "jpeg" {
		t.Fatalf("output decode = %q, %v; want jpeg", format, err)
	}
	if img.Bounds().Dx() != 100 || img.Bounds().Dy() != 50 {
		t.Errorf("size = %v, want 100x50", img.Bounds().Size())
	}
	if r, g, b, _ := img.At(50, 25).RGBA(); r < 0xF000 || g < 0xF000 || b < 0xF000 {
		t.Errorf("transparent pixel became (%d,%d,%d), want white", r>>8, g>>8, b>>8)
	}
}

func TestPreprocessImage_DoesNotEnlarge(t *testing.T) {
	out, err := PreprocessImage(pngBytes(t, 30, 20, color.White), 1024)
	if err != nil {
		t.Fatal(err)
	}
	cfg, _, _ := image.DecodeConfig(bytes.NewReader(out))
	if cfg.Width != 30 || cfg.Height != 20 {
		t.Errorf("size = %dx%d, want 30x20 (unchanged)", cfg.Width, cfg.Height)
	}
}

func TestPreprocessImage_Errors(t *testing.T) {
	if _, err := PreprocessImage([]byte("nope"), 100); !errors.Is(err, ErrUnsupportedImageFormat) {
		t.Errorf("undecodable error = %v, want ErrUnsupportedImageFormat", err)
	}
	if _, err := PreprocessImage(pngBytes(t, 2, 2, color.White), 0); err == nil {
		t.Error("maxSide 0: want an error")
	}
}

func TestScaledSize(t *testing.T) {
	tests := []struct{ w, h, max, wantW, wantH int }{
		{2000, 1000, 1024, 1024, 512},
		{1000, 2000, 1024, 512, 1024},
		{800, 600, 1024, 800, 600},
		{5000, 1, 100, 100, 1},
	}
	for _, tt := range tests {
		if w, h := scaledSize(tt.w, tt.h, tt.max); w != tt.wantW || h != tt.wantH {
			t.Errorf("scaledSize(%d, %d, %d) = %dx%d, want %dx%d", tt.w, tt.h, tt.max, w, h, tt.wantW, tt.wantH)
		}
	}
}

func TestDetectImageFormat(t *testing.T) {
	if f, err := DetectImageFormat(pngBytes(t, 1, 1, color.White)); err != nil || f != ImageFormatPNG {
		t.Errorf("PNG = %q, %v", f, err)
	}
	if f, err := DetectImageFormat(leftRedRightBlueJPEG(t, 2, 2)); err != nil || f != ImageFormatJPEG {
		t.Errorf("JPEG = %q, %v", f, err)
	}
	webp := append([]byte("RIFF\x00\x00\x00\x00WEBPVP8 "), make([]byte, 16)...)
	if f, err := DetectImageFormat(webp); err != nil || f != ImageFormatWEBP {
		t.Errorf("WEBP = %q, %v", f, err)
	}
	if _, err := DetectImageFormat([]byte("BM\x00\x00")); !errors.Is(err, ErrUnsupportedImageFormat) {
		t.Errorf("BMP error = %v, want ErrUnsupportedImageFormat", err)
	}
}
