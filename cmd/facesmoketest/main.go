// Command facesmoketest is a real-AWS smoke test for the facev2
// face-grouping flow (IndexFace -> batch -> SearchFacesByFaceId, with the
// face_id -> user_id grouping generated and kept entirely outside AWS, in a
// local SQLite database).
//
// It reads every image in an input folder, indexes it into a fresh
// Rekognition collection, runs the same Batcher+DBGrouper used by
// facev2/simulation (against the REAL AWS client this time, not the fake),
// and writes one folder per resulting UserId containing the cropped face(s)
// of that person — each crop's filename prefixed with its combined
// representative-thumbnail score (which folds pose, quality, AND AWS's
// face-occlusion judgment together) plus the raw FaceOccluded flag +
// confidence for a human eyeballing why a face got that score (see
// cropFileName) — plus a representative.jpg copy of whichever crop
// DBGrouper picked as that person's best photo, e.g.:
//
//	output/
//	  3fae7e21-.../   (person A)
//	    score091.2_occlN099.9_img001_3fae7e21.jpg
//	    score054.7_occlY088.4_img004_3fae7e21.jpg
//	    representative.jpg   (== score091.2_occlN099.9_img001_3fae7e21.jpg)
//	  9c1b0a44-.../   (person B)
//	    score078.4_occlN099.8_img002_9c1b0a44.jpg
//	  _unresolved/     (faces that indexed fine but never got grouped)
//
// Usage:
//
//	go run ./cmd/facesmoketest -input ./sample_images -output ./output
//
// Requires a populated .env (AWS_REGION/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
// — see -env to point at a different one. This hits real Rekognition APIs
// (IndexFaces, SearchFaces) and creates a real collection, so it incurs real
// usage against your AWS account/quota. AWS Rekognition Users (CreateUser,
// AssociateFaces, SearchUsers) are not used at all — see
// facev2/simulation.DBGrouper and facev2/store.
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	"log"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/HonchoLtd/aws_rekognition/facev2"
	"github.com/HonchoLtd/aws_rekognition/facev2/simulation"
	"github.com/HonchoLtd/aws_rekognition/facev2/store"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/rekognition"
)

var imageExtensions = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
}

func main() {
	var (
		inputDir      = flag.String("input", "./images", "folder of images to index")
		outputDir     = flag.String("output", "./output", "folder to write <user_id>/<cropped_face>.jpg into")
		envPath       = flag.String("env", ".env", "path to the .env file with AWS credentials")
		collectionId  = flag.String("collection", "", "Rekognition collection id to use (default: a fresh timestamped one)")
		dbPath        = flag.String("db", "./facesmoketest.db", "path to the SQLite database used to store the face_id -> user_id grouping (created if missing)")
		batchSize     = flag.Int("batch-size", 100, "number of faces per batch before flushing (count trigger)")
		batchInterval = flag.Duration("batch-interval", 100*time.Second, "max time to wait before flushing a partial batch (time trigger)")
		cropScale     = flag.Float64("crop-scale", 1.8, "how much to expand each face's bounding box before cropping, relative to its own size")
		cleanup       = flag.Bool("cleanup", false, "delete the collection when done (irreversible; off by default)")
		logLevel      = flag.String("log-level", "info", "log verbosity: debug, info, warn, or error. debug traces every batcher Add + grouper decision; info (default) shows every AWS call and batch flush; warn/error quiet it down to problems only")
		logFormat     = flag.String("log-format", "text", "log output format: text or json")
	)
	flag.Parse()

	if *collectionId == "" {
		*collectionId = "facev2-smoketest-" + time.Now().Format("20060102-150405")
	}

	if err := configureLogging(*logLevel, *logFormat); err != nil {
		log.Fatalf("facesmoketest: %v", err)
	}

	if err := run(*inputDir, *outputDir, *envPath, *collectionId, *dbPath, *batchSize, *batchInterval, *cropScale, *cleanup); err != nil {
		log.Fatalf("facesmoketest: %v", err)
	}
}

// configureLogging routes facev2's and this tool's structured logs (via
// slog.SetDefault) at the requested level/format. facev2 and
// facev2/simulation both resolve slog.Default() dynamically (see
// facev2.SetLogger's doc comment), so this one call is enough to control
// logging everywhere in the pipeline — no per-package wiring needed.
func configureLogging(level string, format string) error {
	var lvl slog.Level
	switch strings.ToLower(level) {
	case "debug":
		lvl = slog.LevelDebug
	case "info":
		lvl = slog.LevelInfo
	case "warn", "warning":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		return fmt.Errorf("invalid -log-level %q (want debug, info, warn, or error)", level)
	}

	opts := &slog.HandlerOptions{Level: lvl}
	var handler slog.Handler
	switch strings.ToLower(format) {
	case "text":
		handler = slog.NewTextHandler(os.Stderr, opts)
	case "json":
		handler = slog.NewJSONHandler(os.Stderr, opts)
	default:
		return fmt.Errorf("invalid -log-format %q (want text or json)", format)
	}

	slog.SetDefault(slog.New(handler))
	return nil
}

func run(inputDir, outputDir, envPath, collectionId, dbPath string, batchSize int, batchInterval time.Duration, cropScale float64, cleanup bool) error {
	ctx := context.Background()

	client, err := loadRekognitionClient(envPath)
	if err != nil {
		return fmt.Errorf("load AWS client: %w", err)
	}
	// Wrap with StatsFace so every IndexFace/SearchFacesByFaceId call — from
	// both indexing and grouping below — is counted for the final "how many
	// AWS calls did this run make" breakdown.
	stats := facev2.NewStatsFace(facev2.NewRekognitionFaceIndexer(client))
	var faceClient facev2.Face = stats

	db, err := store.Open(dbPath)
	if err != nil {
		return fmt.Errorf("open grouping database %s: %w", dbPath, err)
	}
	defer db.Close()
	log.Printf("using grouping database %s", dbPath)

	imagePaths, err := listImages(inputDir)
	if err != nil {
		return fmt.Errorf("list images in %s: %w", inputDir, err)
	}
	if len(imagePaths) == 0 {
		return fmt.Errorf("no .jpg/.jpeg/.png files found in %s", inputDir)
	}
	log.Printf("found %d images in %s, indexing into collection %q", len(imagePaths), inputDir, collectionId)

	acc := newRunAccumulator()
	grouper := &simulation.DBGrouper{Face: faceClient, Store: db}
	batcher := simulation.NewBatcher(batchSize, batchInterval, acc.onFlush(ctx, grouper))

	var (
		imagesIndexed int
		facesIndexed  int
	)
	for _, path := range imagePaths {
		faceIds, err := indexAndCrop(ctx, faceClient, db, path, collectionId, cropScale, acc)
		if err != nil {
			log.Printf("WARN: skipping %s: %v", path, err)
			continue
		}
		imagesIndexed++
		facesIndexed += len(faceIds)
		for _, faceId := range faceIds {
			batcher.Add(collectionId, faceId)
		}
	}
	batcher.Stop() // flush whatever's left below the count trigger

	if cleanup {
		log.Printf("cleanup: deleting collection %q", collectionId)
		if _, err := client.DeleteCollection(ctx, &rekognition.DeleteCollectionInput{CollectionId: aws.String(collectionId)}); err != nil {
			log.Printf("WARN: failed to delete collection %q: %v", collectionId, err)
		}
	}

	return acc.writeOutput(ctx, db, collectionId, outputDir, imagesIndexed, facesIndexed, stats.Snapshot())
}

// listImages returns every .jpg/.jpeg/.png file under root, sorted, so runs
// are reproducible.
func listImages(root string) ([]string, error) {
	var paths []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if imageExtensions[strings.ToLower(filepath.Ext(path))] {
			paths = append(paths, path)
		}
		return nil
	})
	sort.Strings(paths)
	return paths, err
}

// indexAndCrop reads and indexes one image, crops every detected face right
// away (while the decoded image is still in hand), records each crop
// against its FaceId in acc, and records each face's metrics — its
// combined representative-thumbnail score (facev2.ComputeFaceScore, which
// already folds pose, quality, AND AWS's face-occlusion signal into a
// single 0-100 number) plus the raw FaceOccluded {Value, Confidence} pair
// for informational surfacing in filenames — into db. This has to happen
// now, at index time, since that's the only point this data is ever
// available for the face (facev2.Face.SearchFacesByFaceId, used later
// during grouping, doesn't return it). It returns the FaceIds detected in
// this image.
func indexAndCrop(ctx context.Context, faceClient facev2.Face, db store.Store, path string, collectionId string, cropScale float64, acc *runAccumulator) ([]string, error) {
	imageBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	externalImageId := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	indexedFaces, err := faceClient.IndexFace(ctx, imageBytes, externalImageId, collectionId)
	if err != nil {
		return nil, fmt.Errorf("IndexFace: %w", err)
	}
	if len(indexedFaces) == 0 {
		log.Printf("no face detected in %s", path)
		return nil, nil
	}

	// Decoded lazily and only if we actually have faces to crop. Input
	// images are assumed to already be right-side up: unlike the `face`
	// package, this tool doesn't apply AWS's OrientationCorrection before
	// cropping (facev2.IndexFace doesn't currently surface it). If your
	// sample set has rotated phone photos, crops may come out sideways —
	// see face.rotateAccordingToOrientation for how that's handled there.
	decoded, _, decodeErr := image.Decode(bytes.NewReader(imageBytes))
	faceIds := make([]string, 0, len(indexedFaces))
	for i, indexedFace := range indexedFaces {
		faceIds = append(faceIds, indexedFace.FaceId)
		acc.recordSource(indexedFace.FaceId, externalImageId)

		metrics := store.FaceMetrics{
			Score:              float64(indexedFace.Score),
			Occluded:           indexedFace.Occluded.Value,
			OccludedConfidence: float64(indexedFace.Occluded.Confidence),
		}
		if err := db.RecordFaceMetrics(ctx, collectionId, indexedFace.FaceId, metrics); err != nil {
			log.Printf("WARN: %s: failed to record metrics for face %s: %v", path, indexedFace.FaceId, err)
		}

		if decodeErr != nil {
			log.Printf("WARN: %s: can't decode for cropping (%v), face %s will have no crop", path, decodeErr, indexedFace.FaceId)
			continue
		}
		cropped, cropErr := cropFace(decoded, indexedFace.BoundingBox, cropScale)
		if cropErr != nil {
			log.Printf("WARN: %s: failed to crop face %d (%s): %v", path, i, indexedFace.FaceId, cropErr)
			continue
		}
		acc.recordCrop(indexedFace.FaceId, cropped)
	}

	return faceIds, nil
}

// runAccumulator collects everything discovered across the whole run —
// per-face crops/source names from indexing, and per-face UserId
// assignments from each batch's grouping — so writeOutput can lay it all
// out at the end.
type runAccumulator struct {
	mu           sync.Mutex
	crops        map[string][]byte // faceId -> cropped JPEG bytes
	sourceImages map[string]string // faceId -> originating file's base name (no ext)
	faceToUser   map[string]string // faceId -> userId, filled in by grouping
	createdUsers []string
	batchErrs    []error
}

func newRunAccumulator() *runAccumulator {
	return &runAccumulator{
		crops:        make(map[string][]byte),
		sourceImages: make(map[string]string),
		faceToUser:   make(map[string]string),
	}
}

func (a *runAccumulator) recordSource(faceId, sourceName string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sourceImages[faceId] = sourceName
}

func (a *runAccumulator) recordCrop(faceId string, cropped []byte) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.crops[faceId] = cropped
}

// onFlush wires a Batcher directly to a DBGrouper against the real AWS
// client.
func (a *runAccumulator) onFlush(ctx context.Context, grouper *simulation.DBGrouper) func(collectionId string, faceIds []string) {
	return func(collectionId string, faceIds []string) {
		log.Printf("flushing batch of %d face(s) for grouping...", len(faceIds))
		result, err := grouper.ProcessBatch(ctx, collectionId, faceIds)

		a.mu.Lock()
		defer a.mu.Unlock()
		for faceId, userId := range result.FaceToUser {
			a.faceToUser[faceId] = userId
		}
		a.createdUsers = append(a.createdUsers, result.CreatedUsers...)
		if err != nil {
			a.batchErrs = append(a.batchErrs, err)
			log.Printf("WARN: batch had errors: %v", err)
		}
	}
}

// writeOutput lays out output/<userId>/<sourceName>_face_<faceId prefix>.jpg
// for every face that both got a crop and got resolved to a UserId, plus an
// output/_unresolved/ folder for anything that fell through and an
// output/<userId>/representative.jpg copy of whichever face DBGrouper chose
// as that user's representative thumbnail (see writeRepresentatives), then
// prints a summary — including the AWS Rekognition API call breakdown from
// callStats.
func (a *runAccumulator) writeOutput(ctx context.Context, db store.Store, collectionId string, outputDir string, imagesIndexed, facesIndexed int, callStats facev2.CallStats) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	usersSeen := make(map[string]int) // userId -> face count, for the summary
	unresolved := 0
	uncropped := 0

	for faceId, cropped := range a.crops {
		userId, resolved := a.faceToUser[faceId]
		targetDir := filepath.Join(outputDir, userId)
		if !resolved {
			targetDir = filepath.Join(outputDir, "_unresolved")
			unresolved++
		} else {
			usersSeen[userId]++
		}

		if err := os.MkdirAll(targetDir, 0o755); err != nil {
			return fmt.Errorf("create dir %s: %w", targetDir, err)
		}

		fileName := cropFileName(ctx, db, collectionId, faceId, a.sourceImages[faceId])
		if err := os.WriteFile(filepath.Join(targetDir, fileName), cropped, 0o644); err != nil {
			return fmt.Errorf("write %s: %w", fileName, err)
		}
	}

	// Faces that indexed but never got a crop (decode/crop failure) still
	// count toward "resolved but no image" for an honest summary.
	for faceId := range a.faceToUser {
		if _, hasCrop := a.crops[faceId]; !hasCrop {
			uncropped++
		}
	}

	repsWritten, repsMissing, err := a.writeRepresentatives(ctx, db, collectionId, outputDir, usersSeen)
	if err != nil {
		return err
	}

	fmt.Println()
	fmt.Println("=== facesmoketest summary ===")
	fmt.Printf("images indexed:      %d\n", imagesIndexed)
	fmt.Printf("faces indexed:       %d\n", facesIndexed)
	fmt.Printf("users created:       %d\n", len(a.createdUsers))
	fmt.Printf("distinct users seen: %d\n", len(usersSeen))
	fmt.Printf("faces unresolved:    %d\n", unresolved)
	if uncropped > 0 {
		fmt.Printf("faces resolved but not cropped (decode/crop failed): %d\n", uncropped)
	}
	if len(a.batchErrs) > 0 {
		fmt.Printf("batches with errors: %d (see warnings above)\n", len(a.batchErrs))
	}
	fmt.Printf("representative thumbnails written: %d\n", repsWritten)
	if repsMissing > 0 {
		fmt.Printf("users missing a representative:    %d (see warnings above)\n", repsMissing)
	}

	fmt.Println()
	fmt.Println("=== AWS Rekognition API calls ===")
	fmt.Printf("IndexFace:            %d\n", callStats.IndexFace)
	fmt.Printf("IndexFaceWithBucket:  %d\n", callStats.IndexFaceWithBucket)
	fmt.Printf("SearchFacesByFaceId:  %d\n", callStats.SearchFacesByFaceId)
	fmt.Printf("total API calls:     %d\n", callStats.Total())
	if imagesIndexed > 0 {
		fmt.Printf("avg calls per image:  %.2f\n", float64(callStats.Total())/float64(imagesIndexed))
	}
	if facesIndexed > 0 {
		fmt.Printf("avg calls per face:   %.2f\n", float64(callStats.Total())/float64(facesIndexed))
	}

	fmt.Println()
	fmt.Println("faces per user:")
	userIds := make([]string, 0, len(usersSeen))
	for userId := range usersSeen {
		userIds = append(userIds, userId)
	}
	sort.Strings(userIds)
	for _, userId := range userIds {
		fmt.Printf("  %s: %d face(s)\n", userId, usersSeen[userId])
	}
	fmt.Printf("\noutput written to %s\n", outputDir)

	return nil
}

// writeRepresentatives writes output/<userId>/representative.jpg for every
// userId in usersSeen that has a representative-thumbnail candidate
// recorded (see facev2/simulation.DBGrouper's doc comment on
// considerRepresentative), copying whichever crop that candidate's FaceId
// already has on disk. It returns how many were written, and how many
// userIds in usersSeen had none (logged as warnings, not fatal — a run
// without scoring wired up, or where every face fell below the fallback
// threshold before a crop existed, simply won't have any).
//
// Must be called with a.mu already held (only writeOutput calls this).
func (a *runAccumulator) writeRepresentatives(ctx context.Context, db store.Store, collectionId string, outputDir string, usersSeen map[string]int) (written int, missing int, err error) {
	for userId := range usersSeen {
		rep, found, err := db.Representative(ctx, collectionId, userId)
		if err != nil {
			return written, missing, fmt.Errorf("look up representative for user %s: %w", userId, err)
		}
		if !found {
			missing++
			log.Printf("WARN: user %s has no representative candidate (no face score was ever recorded for it)", userId)
			continue
		}

		cropped, hasCrop := a.crops[rep.FaceId]
		if !hasCrop {
			missing++
			log.Printf("WARN: representative face %s for user %s has no crop on disk", rep.FaceId, userId)
			continue
		}

		repPath := filepath.Join(outputDir, userId, "representative.jpg")
		if err := os.WriteFile(repPath, cropped, 0o644); err != nil {
			return written, missing, fmt.Errorf("write representative %s: %w", repPath, err)
		}
		written++
		log.Printf("representative for user %s: face %s (score %.1f, locked=%v)", userId, shortId(rep.FaceId), rep.Score, rep.Locked)
	}
	return written, missing, nil
}

// cropFileName builds a crop's on-disk filename, prefixed with:
//   - its combined representative-thumbnail score
//     (facev2.ComputeFaceScore — pose, quality, and occlusion together),
//   - AWS's raw face-occlusion judgment (facev2.IndexedFace.Occluded) — a
//     Y/N flag plus AWS's confidence in it,
//
// both recorded at index time (see indexAndCrop), so they're visible at a
// glance in the output folder, e.g.
// "score085.3_occlN005.2_img001_3fae7e21.jpg". Numeric fields are
// zero-padded to a fixed width so filenames also SORT by score
// (alphabetically == numerically) within a folder — the occlusion segment
// is placed after score so it doesn't disturb that ordering. Falls back to
// no prefix at all if faceId has no recorded metrics (e.g. that wasn't
// wired up, or this is a stale .db file from before the feature existed).
func cropFileName(ctx context.Context, db store.Store, collectionId string, faceId string, sourceName string) string {
	metrics, found, err := db.FaceMetrics(ctx, collectionId, faceId)
	if err != nil {
		log.Printf("WARN: failed to look up metrics for face %s: %v", faceId, err)
		found = false
	}
	if !found {
		return fmt.Sprintf("%s_%s.jpg", sourceName, shortId(faceId))
	}
	occludedFlag := "N"
	if metrics.Occluded {
		occludedFlag = "Y"
	}
	return fmt.Sprintf("score%05.1f_occl%s%05.1f_%s_%s.jpg", metrics.Score, occludedFlag, metrics.OccludedConfidence, sourceName, shortId(faceId))
}

func shortId(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}
