package modelembedding

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"net/http"

	_ "image/gif"
	_ "image/png"

	"golang.org/x/image/draw"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

const (
	// DefaultMaxImageSide is the longest side, in pixels, PreprocessImage
	// shrinks images to when WithPreprocess is used.
	DefaultMaxImageSide = 1024

	preprocessJPEGQuality = 90
)

// DetectImageFormat sniffs image's magic bytes and returns the matching
// Nova format, or ErrUnsupportedImageFormat for anything Nova can't take
// directly.
func DetectImageFormat(image []byte) (ImageFormat, error) {
	switch contentType := http.DetectContentType(image); contentType {
	case "image/jpeg":
		return ImageFormatJPEG, nil
	case "image/png":
		return ImageFormatPNG, nil
	case "image/gif":
		return ImageFormatGIF, nil
	case "image/webp":
		return ImageFormatWEBP, nil
	default:
		return "", fmt.Errorf("%w: detected %q (Nova takes jpeg, png, gif, webp; WithPreprocess also converts bmp and tiff)", ErrUnsupportedImageFormat, contentType)
	}
}

// PreprocessImage normalizes an image before embedding: it applies the
// JPEG EXIF orientation (Go's decoder ignores it, so re-encoding would
// otherwise bake in a sideways photo), flattens transparency onto white,
// shrinks the image so its longest side is at most maxSide pixels (never
// enlarges), and re-encodes it as a JPEG. GIFs keep only their first frame.
func PreprocessImage(data []byte, maxSide int) ([]byte, error) {
	if maxSide <= 0 {
		return nil, fmt.Errorf("modelembedding: max image side must be positive, got %d", maxSide)
	}
	src, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("%w: decode image: %v", ErrUnsupportedImageFormat, err)
	}

	bounds := src.Bounds()
	width, height := scaledSize(bounds.Dx(), bounds.Dy(), maxSide)

	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(dst, dst.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	if width == bounds.Dx() && height == bounds.Dy() {
		draw.Draw(dst, dst.Bounds(), src, bounds.Min, draw.Over)
	} else {
		draw.CatmullRom.Scale(dst, dst.Bounds(), src, bounds, draw.Over, nil)
	}

	oriented := dst
	if format == "jpeg" {
		oriented = applyOrientation(dst, jpegOrientation(data))
	}

	var out bytes.Buffer
	if err := jpeg.Encode(&out, oriented, &jpeg.Options{Quality: preprocessJPEGQuality}); err != nil {
		return nil, fmt.Errorf("modelembedding: encode jpeg: %w", err)
	}
	return out.Bytes(), nil
}

// scaledSize returns width/height shrunk proportionally so the longest side
// is at most maxSide, keeping each side at least 1 pixel.
func scaledSize(width, height, maxSide int) (int, int) {
	longest := max(width, height)
	if longest <= maxSide {
		return width, height
	}
	scale := float64(maxSide) / float64(longest)
	return max(1, int(float64(width)*scale+0.5)), max(1, int(float64(height)*scale+0.5))
}

// applyOrientation returns src transformed according to an EXIF
// orientation value (1-8), so it displays upright. Values outside 2-8
// return src unchanged.
func applyOrientation(src *image.RGBA, orientation int) *image.RGBA {
	if orientation < 2 || orientation > 8 {
		return src
	}
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	dstW, dstH := w, h
	if orientation >= 5 {
		dstW, dstH = h, w
	}
	dst := image.NewRGBA(image.Rect(0, 0, dstW, dstH))

	for sy := 0; sy < h; sy++ {
		for sx := 0; sx < w; sx++ {
			var dx, dy int
			switch orientation {
			case 2: // mirror horizontally
				dx, dy = w-1-sx, sy
			case 3: // rotate 180
				dx, dy = w-1-sx, h-1-sy
			case 4: // mirror vertically
				dx, dy = sx, h-1-sy
			case 5: // transpose
				dx, dy = sy, sx
			case 6: // rotate 90 clockwise
				dx, dy = h-1-sy, sx
			case 7: // transverse
				dx, dy = h-1-sy, w-1-sx
			case 8: // rotate 90 counter-clockwise
				dx, dy = sy, w-1-sx
			}
			si := src.PixOffset(sx+src.Rect.Min.X, sy+src.Rect.Min.Y)
			di := dst.PixOffset(dx, dy)
			copy(dst.Pix[di:di+4], src.Pix[si:si+4])
		}
	}
	return dst
}

// jpegOrientation reads the EXIF Orientation tag (0x0112) from a JPEG's
// APP1 segment. It returns 1 (upright) when the data has no EXIF block or
// the block can't be parsed.
func jpegOrientation(data []byte) int {
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1
	}
	pos := 2
	for pos+4 <= len(data) {
		if data[pos] != 0xFF {
			return 1
		}
		marker := data[pos+1]
		if marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01 || marker == 0xFF {
			pos++
			continue
		}
		if marker == 0xDA || marker == 0xD9 { // start of scan / end of image
			return 1
		}
		segmentLen := int(binary.BigEndian.Uint16(data[pos+2 : pos+4]))
		if segmentLen < 2 || pos+2+segmentLen > len(data) {
			return 1
		}
		segment := data[pos+4 : pos+2+segmentLen]
		if marker == 0xE1 && len(segment) >= 6 && string(segment[:6]) == "Exif\x00\x00" {
			return exifOrientation(segment[6:])
		}
		pos += 2 + segmentLen
	}
	return 1
}

// exifOrientation parses a TIFF-structured EXIF payload and returns IFD0's
// Orientation value, or 1 if absent or malformed.
func exifOrientation(tiff []byte) int {
	if len(tiff) < 8 {
		return 1
	}
	var order binary.ByteOrder
	switch string(tiff[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 1
	}
	if order.Uint16(tiff[2:4]) != 42 {
		return 1
	}
	ifd := int(order.Uint32(tiff[4:8]))
	if ifd < 8 || ifd+2 > len(tiff) {
		return 1
	}
	entries := int(order.Uint16(tiff[ifd : ifd+2]))
	for i := 0; i < entries; i++ {
		entry := ifd + 2 + i*12
		if entry+12 > len(tiff) {
			return 1
		}
		if order.Uint16(tiff[entry:entry+2]) != 0x0112 {
			continue
		}
		const typeShort = 3
		if order.Uint16(tiff[entry+2:entry+4]) != typeShort {
			return 1
		}
		value := int(order.Uint16(tiff[entry+8 : entry+10]))
		if value < 1 || value > 8 {
			return 1
		}
		return value
	}
	return 1
}
