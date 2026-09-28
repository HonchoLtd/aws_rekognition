package facev2

import (
	"log/slog"
	"sync/atomic"
)

// loggerOverride, when set via SetLogger, is the logger every facev2 call
// uses. Left unset (the default), facev2 resolves slog.Default() fresh on
// every log line — same as the simulation package's plain slog.Debug/Info
// calls — so a process-wide slog.SetDefault(...) reaches both without
// requiring a separate facev2.SetLogger call. Call SetLogger only when
// facev2 specifically needs to log somewhere different from the process
// default (a different level, JSON output, a file, discarded entirely via
// slog.New(slog.DiscardHandler)).
var loggerOverride atomic.Pointer[slog.Logger]

// SetLogger makes facev2 log through l instead of slog.Default(). Safe to
// call concurrently with in-flight calls (they'll pick it up on their next
// log line); nil is ignored (there's no way to go back to "track
// slog.Default() dynamically" once overridden — pass slog.Default() itself
// if that's what you want frozen in).
func SetLogger(l *slog.Logger) {
	if l == nil {
		return
	}
	loggerOverride.Store(l)
}

// getLogger returns the logger to use for this call: the SetLogger override
// if one was set, otherwise the current slog.Default(). Internal call sites
// use this instead of slog.* directly so overriding actually takes effect.
// (Named to avoid colliding with the stdlib "log" package, which other
// files in this package still use for non-structured messages.)
func getLogger() *slog.Logger {
	if l := loggerOverride.Load(); l != nil {
		return l
	}
	return slog.Default()
}
