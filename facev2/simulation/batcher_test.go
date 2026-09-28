package simulation

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestBatcher_StopWaitsForInFlightTimerTriggeredFlush reproduces the exact
// bug a real facesmoketest run caught: the time trigger fires (on its own
// goroutine) and starts a slow onFlush, while — concurrently — the last
// image finishes indexing, gets Add()ed, and the caller calls Stop(). Stop
// must not return (and a caller must not be able to read "final" results)
// until that earlier, still-running flush has actually finished.
func TestBatcher_StopWaitsForInFlightTimerTriggeredFlush(t *testing.T) {
	const collectionId = "event-stop-waits"

	var mu sync.Mutex
	var flushedBatches [][]string
	flushStarted := make(chan struct{})
	releaseFlush := make(chan struct{})

	onFlush := func(_ string, faceIds []string) {
		close(flushStarted)
		<-releaseFlush // held open until the test says it's OK to finish

		mu.Lock()
		flushedBatches = append(flushedBatches, faceIds)
		mu.Unlock()
	}

	batcher := NewBatcher(100 /* capacity never reached */, 20*time.Millisecond, onFlush)

	batcher.Add(collectionId, "face-1") // arms the 20ms time trigger
	<-flushStarted                      // the timer fired and onFlush started (and is now blocked)

	// Simulate "the last image finishes and Stop() is called" while that
	// timer-triggered flush is still in progress.
	stopDone := make(chan struct{})
	go func() {
		batcher.Stop()
		close(stopDone)
	}()

	select {
	case <-stopDone:
		t.Fatal("Stop() returned before the in-flight timer-triggered flush finished")
	case <-time.After(100 * time.Millisecond):
		// Good: Stop() is correctly still blocked.
	}

	close(releaseFlush) // let the flush finish

	select {
	case <-stopDone:
	case <-time.After(2 * time.Second):
		t.Fatal("Stop() never returned after the in-flight flush was released")
	}

	mu.Lock()
	defer mu.Unlock()
	if len(flushedBatches) != 1 || len(flushedBatches[0]) != 1 || flushedBatches[0][0] != "face-1" {
		t.Errorf("flushedBatches = %+v, want exactly one batch containing face-1", flushedBatches)
	}
}

// TestBatcher_SerializesFlushesForSameCollection fires many concurrent Adds
// for one collection (capacity=1, so every Add triggers its own flush) and
// asserts onFlush is never called concurrently with itself for that
// collection — the guard against two ProcessBatch calls independently
// clustering the same person into two different Users.
func TestBatcher_SerializesFlushesForSameCollection(t *testing.T) {
	const collectionId = "event-serialize"

	var mu sync.Mutex
	var concurrent, maxConcurrent int

	onFlush := func(_ string, _ []string) {
		mu.Lock()
		concurrent++
		if concurrent > maxConcurrent {
			maxConcurrent = concurrent
		}
		mu.Unlock()

		time.Sleep(20 * time.Millisecond) // simulate slow AWS calls

		mu.Lock()
		concurrent--
		mu.Unlock()
	}

	batcher := NewBatcher(1 /* flush after every single Add */, time.Hour, onFlush)

	const n = 10
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			batcher.Add(collectionId, fmt.Sprintf("face-%d", i))
		}(i)
	}
	wg.Wait()
	batcher.Stop()

	mu.Lock()
	defer mu.Unlock()
	if maxConcurrent > 1 {
		t.Errorf("max concurrent onFlush calls for one collection = %d, want at most 1", maxConcurrent)
	}

	// With capacity=1 but 10 *concurrent* Adds racing in, more than one face
	// can legitimately land in b.pending before any of them gets to flush —
	// that's fine (it's only ever fewer, never more, round-trips); what
	// must hold is that every item still gets flushed exactly once, and
	// never two batches' worth concurrently (checked above).
	if flushCount, itemsFlushed := batcher.Stats(); itemsFlushed != n || flushCount > n || flushCount < 1 {
		t.Errorf("Stats() = (flushCount=%d, itemsFlushed=%d), want itemsFlushed=%d and 1<=flushCount<=%d", flushCount, itemsFlushed, n, n)
	}
}

// TestBatcher_DoesNotSerializeAcrossDifferentCollections is the flip side:
// flushes for DIFFERENT collections must still run concurrently — the
// same-collection guard shouldn't accidentally become a global one and
// throttle unrelated events/collections against each other.
func TestBatcher_DoesNotSerializeAcrossDifferentCollections(t *testing.T) {
	var mu sync.Mutex
	var concurrent, maxConcurrent int
	release := make(chan struct{})

	onFlush := func(_ string, _ []string) {
		mu.Lock()
		concurrent++
		if concurrent > maxConcurrent {
			maxConcurrent = concurrent
		}
		mu.Unlock()

		<-release

		mu.Lock()
		concurrent--
		mu.Unlock()
	}

	batcher := NewBatcher(1, time.Hour, onFlush)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); batcher.Add("collection-A", "face-a") }()
	go func() { defer wg.Done(); batcher.Add("collection-B", "face-b") }()

	// Give both a moment to reach onFlush and block on release.
	deadline := time.After(2 * time.Second)
	for {
		mu.Lock()
		c := concurrent
		mu.Unlock()
		if c == 2 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("both collections' flushes never became concurrent within the deadline")
		case <-time.After(5 * time.Millisecond):
		}
	}

	close(release)
	wg.Wait()
	batcher.Stop()

	mu.Lock()
	defer mu.Unlock()
	if maxConcurrent < 2 {
		t.Errorf("maxConcurrent = %d, want 2 (different collections should flush concurrently)", maxConcurrent)
	}
}

// TestBatcher_WaitBlocksUntilInFlightFlushesFinish exercises Wait directly
// (as opposed to Stop, which also drains and disables the Batcher).
func TestBatcher_WaitBlocksUntilInFlightFlushesFinish(t *testing.T) {
	const collectionId = "event-wait"

	release := make(chan struct{})
	started := make(chan struct{})
	onFlush := func(_ string, _ []string) {
		close(started)
		<-release
	}

	batcher := NewBatcher(1, time.Hour, onFlush)
	go batcher.Add(collectionId, "face-1")
	<-started

	waitDone := make(chan struct{})
	go func() {
		batcher.Wait()
		close(waitDone)
	}()

	select {
	case <-waitDone:
		t.Fatal("Wait() returned before the in-flight flush finished")
	case <-time.After(50 * time.Millisecond):
	}

	close(release)

	select {
	case <-waitDone:
	case <-time.After(2 * time.Second):
		t.Fatal("Wait() never returned after the flush was released")
	}
}
