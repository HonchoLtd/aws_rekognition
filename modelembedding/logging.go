package modelembedding

import (
	"log/slog"
	"sync/atomic"
)

// loggerOverride, when set via SetLogger, is the logger every
// modelembedding call uses. Left unset (the default), slog.Default() is
// resolved fresh on every log line, so a process-wide slog.SetDefault(...)
// reaches this package without a separate SetLogger call.
var loggerOverride atomic.Pointer[slog.Logger]

// SetLogger makes modelembedding log through l instead of slog.Default().
// Safe to call concurrently with in-flight calls; nil is ignored.
func SetLogger(l *slog.Logger) {
	if l == nil {
		return
	}
	loggerOverride.Store(l)
}

func getLogger() *slog.Logger {
	if l := loggerOverride.Load(); l != nil {
		return l
	}
	return slog.Default()
}
