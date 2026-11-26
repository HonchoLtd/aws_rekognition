package suggestaws

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// MaxBatchChars keeps each ApplyGuardrail call inside one 1,000-character
// billed text unit.
const MaxBatchChars = 900

// Screen implements Client. Terms are de-duplicated (exact match, input
// order kept) and blank terms are dropped. Each batch is one billed call;
// a batch the guardrail intervenes on is split in half until every
// blocked term is isolated.
func (c *AWSClient) Screen(ctx context.Context, terms []string) (ScreenResult, error) {
	if c.cfg.GuardrailID == "" {
		return ScreenResult{}, ErrGuardrailNotConfigured
	}
	unique := dedupeTerms(terms)
	result := ScreenResult{Allowed: []string{}, Blocked: map[string]string{}}
	if len(unique) == 0 {
		return result, nil
	}

	s := &screener{client: c}
	start := time.Now()
	err := runParallel(ctx, packBatches(unique, MaxBatchChars), s.screen)
	attrs := []any{"op", "Screen", "guardrail_id", c.cfg.GuardrailID, "guardrail_version", c.cfg.GuardrailVersion,
		"terms", len(unique), "calls", s.calls, "blocked", len(s.blocked), "duration_ms", time.Since(start).Milliseconds()}
	if err != nil {
		getLogger().Error("guardrail screening failed", append(attrs, "error", err)...)
		return ScreenResult{}, fmt.Errorf("suggestaws: Screen: %w", err)
	}
	getLogger().Info("guardrail screening", attrs...)

	result.Calls = s.calls
	for _, t := range unique {
		if reason, blocked := s.blocked[t]; blocked {
			result.Blocked[t] = reason
		} else {
			result.Allowed = append(result.Allowed, t)
		}
	}
	return result, nil
}

// screener collects the outcome of one Screen call across goroutines.
type screener struct {
	client  *AWSClient
	mu      sync.Mutex
	calls   int
	blocked map[string]string
}

// screen checks one batch; on intervention it isolates the blocked terms
// by splitting the batch in half and screening both halves in parallel.
func (s *screener) screen(ctx context.Context, batch []string) error {
	out, err := s.client.applyGuardrail(ctx, strings.Join(batch, "\n"))
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.calls++
	s.mu.Unlock()

	// Anything other than NONE is treated as an intervention (fail closed).
	if out.Action == types.GuardrailActionNone {
		return nil
	}
	if len(batch) == 1 {
		s.mu.Lock()
		if s.blocked == nil {
			s.blocked = map[string]string{}
		}
		s.blocked[batch[0]] = blockReasons(out.Assessments)
		s.mu.Unlock()
		return nil
	}
	mid := len(batch) / 2
	return runParallel(ctx, [][]string{batch[:mid], batch[mid:]}, s.screen)
}

func (c *AWSClient) applyGuardrail(ctx context.Context, text string) (*bedrockruntime.ApplyGuardrailOutput, error) {
	input := &bedrockruntime.ApplyGuardrailInput{
		GuardrailIdentifier: aws.String(c.cfg.GuardrailID),
		GuardrailVersion:    aws.String(c.cfg.GuardrailVersion),
		Source:              types.GuardrailContentSourceOutput,
		Content: []types.GuardrailContentBlock{
			&types.GuardrailContentBlockMemberText{Value: types.GuardrailTextBlock{Text: aws.String(text)}},
		},
	}
	var out *bedrockruntime.ApplyGuardrailOutput
	err := withRetry(ctx, c.retry, "ApplyGuardrail", func() error {
		var callErr error
		out, callErr = c.rt.ApplyGuardrail(ctx, input)
		return callErr
	})
	if err != nil {
		return nil, err
	}
	if out == nil {
		return nil, fmt.Errorf("%w: nil ApplyGuardrail output", ErrUnexpectedResponse)
	}
	return out, nil
}

// blockReasons lists the policies that blocked, e.g.
// "topic:PersonalAttributes,word:women". De-duplicated, order kept.
func blockReasons(assessments []types.GuardrailAssessment) string {
	var reasons []string
	seen := map[string]bool{}
	add := func(r string) {
		if !seen[r] {
			seen[r] = true
			reasons = append(reasons, r)
		}
	}
	for _, a := range assessments {
		if a.TopicPolicy != nil {
			for _, t := range a.TopicPolicy.Topics {
				if t.Action == types.GuardrailTopicPolicyActionBlocked {
					add("topic:" + aws.ToString(t.Name))
				}
			}
		}
		if a.ContentPolicy != nil {
			for _, f := range a.ContentPolicy.Filters {
				if f.Action == types.GuardrailContentPolicyActionBlocked {
					add("content:" + string(f.Type))
				}
			}
		}
		if a.WordPolicy != nil {
			for _, w := range a.WordPolicy.CustomWords {
				if w.Action == types.GuardrailWordPolicyActionBlocked {
					add("word:" + strings.ToLower(aws.ToString(w.Match)))
				}
			}
			for _, w := range a.WordPolicy.ManagedWordLists {
				if w.Action == types.GuardrailWordPolicyActionBlocked {
					add("profanity:" + strings.ToLower(aws.ToString(w.Match)))
				}
			}
		}
	}
	if len(reasons) == 0 {
		return "unknown"
	}
	return strings.Join(reasons, ",")
}

// dedupeTerms drops blank terms and exact duplicates, keeping input order.
func dedupeTerms(terms []string) []string {
	seen := make(map[string]bool, len(terms))
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		if strings.TrimSpace(t) == "" || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	return out
}

// packBatches greedily packs terms, in order, into batches whose
// newline-joined length is at most maxChars. A single term longer than
// maxChars gets a batch of its own.
func packBatches(terms []string, maxChars int) [][]string {
	var batches [][]string
	var cur []string
	curLen := 0
	for _, t := range terms {
		n := utf8.RuneCountInString(t)
		if len(cur) > 0 && curLen+1+n > maxChars {
			batches = append(batches, cur)
			cur, curLen = nil, 0
		}
		if len(cur) > 0 {
			curLen++ // the newline
		}
		cur = append(cur, t)
		curLen += n
	}
	if len(cur) > 0 {
		batches = append(batches, cur)
	}
	return batches
}

// runParallel runs fn on every item concurrently and returns the first
// error, cancelling the others when one fails.
func runParallel[T any](ctx context.Context, items []T, fn func(context.Context, T) error) error {
	if len(items) == 1 {
		return fn(ctx, items[0])
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)
	for _, item := range items {
		wg.Add(1)
		go func(item T) {
			defer wg.Done()
			if err := fn(ctx, item); err != nil {
				once.Do(func() {
					firstErr = err
					cancel()
				})
			}
		}(item)
	}
	wg.Wait()
	return firstErr
}
