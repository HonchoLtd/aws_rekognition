package suggestaws

import (
	"log/slog"
	"sync/atomic"
)

// loggerOverride, when set via SetLogger, is the logger every suggestaws
// call uses. Left unset, slog.Default() is resolved on every log line, so
// a process-wide slog.SetDefault reaches this package too.
var loggerOverride atomic.Pointer[slog.Logger]

// SetLogger makes suggestaws log through l instead of slog.Default().
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
