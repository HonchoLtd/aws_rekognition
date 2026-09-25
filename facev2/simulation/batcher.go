package simulation

import (
	"log/slog"
	"sync"
	"time"
)

// Batcher accumulates FaceIds and flushes them — via onFlush — whenever
// either trigger fires first:
//   - count: the batch reaches `capacity` items, or
//   - time: `interval` has elapsed since the first item landed in the
//     current (non-empty) batch.
//
// This is the "process every 100 faces, or after a certain period, whichever
// comes first" trigger described for the face-grouping pipeline. It has no
// AWS/DB knowledge of its own — onFlush is handed the accumulated FaceIds
// and collectionId, and typically wraps a Grouper.ProcessBatch call plus
// whatever persistence the caller needs.
//
// Correctness guarantees, on top of the count/time triggers:
//   - Two flushes for the SAME collectionId never call onFlush concurrently
//     — a second trigger for a collection whose previous flush is still
//     running (most commonly: the time trigger fires on its own goroutine
//     while the last few Adds are still landing) blocks until that earlier
//     flush finishes. This is what prevents two overlapping ProcessBatch
//     calls from independently discovering and clustering the same person
//     into two different Users. Flushes for DIFFERENT collections are not
//     serialized against each other.
//   - Stop (and Wait) block until every flush that has started — including
//     one the internal timer already kicked off before Stop was called —
//     has fully completed, so a caller can never read final results (or
//     exit the process) while a batch is still being processed.
//
// Every Add and Flush logs through log/slog (Debug for Add, Info for
// Flush — including *why* it flushed: "count", "time", "manual", or
// "stop") so the batching flow is visible without instrumenting the caller.
type Batcher struct {
	capacity int
	interval time.Duration
	onFlush  func(collectionId string, faceIds []string)

	mu           sync.Mutex
	pending      map[string][]string    // collectionId -> pending faceIds
	timers       map[string]*time.Timer // collectionId -> armed time-trigger timer
	flushLocks   map[string]*sync.Mutex // collectionId -> lock serializing onFlush calls for it
	stopped      bool
	flushCount   int // number of Flush calls made so far; exposed via Stats for tests
	itemsFlushed int
	nextBatchSeq int // monotonic id assigned to each flush, for correlating log lines

	flushWG sync.WaitGroup // tracks every onFlush call currently in flight, across all collections
}

// NewBatcher creates a Batcher with the given count and time triggers.
// capacity <= 0 disables the count trigger (time-only); interval <= 0
// disables the time trigger (count-only). onFlush is called from whichever
// goroutine causes the flush (Add, Flush, the internal timer, or Stop) —
// keep it fast, or hand work off asynchronously inside it. See the type
// doc comment for the concurrency guarantees around overlapping flushes.
func NewBatcher(capacity int, interval time.Duration, onFlush func(collectionId string, faceIds []string)) *Batcher {
	return &Batcher{
		capacity:   capacity,
		interval:   interval,
		onFlush:    onFlush,
		pending:    make(map[string][]string),
		timers:     make(map[string]*time.Timer),
		flushLocks: make(map[string]*sync.Mutex),
	}
}

// Add appends faceId to collectionId's pending batch. If this is the first
// pending item for collectionId, the time trigger starts counting down now.
// If capacity is reached, Add flushes immediately — which can block if a
// previous flush for this same collectionId (e.g. a time-triggered one) is
// still in progress; see the type doc comment.
func (b *Batcher) Add(collectionId string, faceId string) {
	b.mu.Lock()

	if b.stopped {
		b.mu.Unlock()
		slog.Warn("batcher: Add called after Stop, ignoring", "collection_id", collectionId, "face_id", faceId)
		return
	}

	b.pending[collectionId] = append(b.pending[collectionId], faceId)
	pendingCount := len(b.pending[collectionId])

	if b.interval > 0 {
		if _, running := b.timers[collectionId]; !running {
			b.timers[collectionId] = time.AfterFunc(b.interval, func() { b.flush(collectionId, "time") })
			slog.Debug("batcher: time trigger armed", "collection_id", collectionId, "interval_ms", b.interval.Milliseconds())
		}
	}

	shouldFlush := b.capacity > 0 && pendingCount >= b.capacity
	b.mu.Unlock()

	slog.Debug("batcher: face added", "collection_id", collectionId, "face_id", faceId, "pending_count", pendingCount, "capacity", b.capacity)

	if shouldFlush {
		b.flush(collectionId, "count")
	}
}

// Flush immediately flushes collectionId's pending batch, if any, regardless
// of whether either trigger has fired yet. Safe to call concurrently and
// redundantly (a no-op when there's nothing pending). Can block if a
// previous flush for this same collectionId is still in progress.
func (b *Batcher) Flush(collectionId string) {
	b.flush(collectionId, "manual")
}

// flush is Flush's implementation, plus the trigger reason for logging —
// "count" (capacity reached in Add), "time" (interval elapsed), "manual"
// (an explicit Flush call), or "stop" (Stop draining pending batches).
//
// It grabs whatever's pending for collectionId (fast, under b.mu), then —
// only if there's actually something to flush — registers the flush with
// flushWG (so Stop/Wait can find it) and serializes against any other flush
// for the same collectionId via flushLocks before calling onFlush. Getting
// the per-collection lock happens outside b.mu, so collections never
// contend with each other, only same-collection flushes do.
func (b *Batcher) flush(collectionId string, reason string) {
	b.mu.Lock()
	faceIds := b.pending[collectionId]
	delete(b.pending, collectionId)
	if timer, ok := b.timers[collectionId]; ok {
		timer.Stop()
		delete(b.timers, collectionId)
	}
	if len(faceIds) == 0 {
		b.mu.Unlock()
		slog.Debug("batcher: flush requested but nothing pending", "collection_id", collectionId, "reason", reason)
		return
	}
	b.flushCount++
	b.itemsFlushed += len(faceIds)
	b.nextBatchSeq++
	batchSeq := b.nextBatchSeq
	flushLock := b.flushLockFor(collectionId)
	b.flushWG.Add(1)
	b.mu.Unlock()

	defer b.flushWG.Done()

	flushLock.Lock()
	defer flushLock.Unlock()

	slog.Info("batcher: flushing batch", "collection_id", collectionId, "batch_seq", batchSeq, "reason", reason, "face_count", len(faceIds))
	start := time.Now()
	b.onFlush(collectionId, faceIds)
	slog.Info("batcher: batch processed", "collection_id", collectionId, "batch_seq", batchSeq, "reason", reason, "face_count", len(faceIds), "duration_ms", time.Since(start).Milliseconds())
}

// flushLockFor returns the mutex serializing onFlush calls for collectionId,
// creating it on first use. Must be called with b.mu held.
func (b *Batcher) flushLockFor(collectionId string) *sync.Mutex {
	l, ok := b.flushLocks[collectionId]
	if !ok {
		l = &sync.Mutex{}
		b.flushLocks[collectionId] = l
	}
	return l
}

// Stop flushes every collection with pending items, stops all timers, and
// then blocks until every flush that had already started — including one
// the internal timer kicked off before Stop was called — has fully
// completed. The Batcher rejects further Adds after Stop. Safe to call
// exactly once; call Wait instead if you need to wait for in-flight
// flushes without also shutting the Batcher down.
func (b *Batcher) Stop() {
	b.mu.Lock()
	b.stopped = true
	collectionIds := make([]string, 0, len(b.pending))
	for collectionId := range b.pending {
		collectionIds = append(collectionIds, collectionId)
	}
	b.mu.Unlock()

	slog.Info("batcher: stopping, draining pending batches", "collections_pending", len(collectionIds))
	for _, collectionId := range collectionIds {
		b.flush(collectionId, "stop")
	}

	b.Wait()
}

// Wait blocks until every flush that has started so far — from any trigger,
// on any goroutine — has fully completed. Unlike Stop, it doesn't drain
// pending batches or reject further Adds, so it's safe to call mid-run (e.g.
// as a checkpoint) as well as during shutdown.
func (b *Batcher) Wait() {
	b.flushWG.Wait()
}

// Stats returns how many flushes have happened and how many FaceIds have
// been flushed in total, for test/observability purposes.
func (b *Batcher) Stats() (flushCount int, itemsFlushed int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.flushCount, b.itemsFlushed
}
