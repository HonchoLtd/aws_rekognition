package suggestaws

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// fakeGuardrail intervenes on any batch containing a line in blocked, and
// reports that term's assessment.
func fakeGuardrail(blocked map[string]types.GuardrailAssessment) func(*bedrockruntime.ApplyGuardrailInput) (*bedrockruntime.ApplyGuardrailOutput, error) {
	return func(in *bedrockruntime.ApplyGuardrailInput) (*bedrockruntime.ApplyGuardrailOutput, error) {
		text := aws.ToString(in.Content[0].(*types.GuardrailContentBlockMemberText).Value.Text)
		var hits []types.GuardrailAssessment
		for _, line := range strings.Split(text, "\n") {
			if a, ok := blocked[line]; ok {
				hits = append(hits, a)
			}
		}
		if len(hits) == 0 {
			return &bedrockruntime.ApplyGuardrailOutput{Action: types.GuardrailActionNone}, nil
		}
		return &bedrockruntime.ApplyGuardrailOutput{Action: types.GuardrailActionGuardrailIntervened, Assessments: hits}, nil
	}
}

func screenClient(rt *fakeRuntime) *AWSClient {
	return newTestClient(rt, Config{GuardrailID: "gr-123", GuardrailVersion: "3"})
}

func TestScreen_IsolatesBlockedTermsWithReasons(t *testing.T) {
	rt := &fakeRuntime{applyGuardrail: fakeGuardrail(map[string]types.GuardrailAssessment{
		"women networking": {
			TopicPolicy: &types.GuardrailTopicPolicyAssessment{Topics: []types.GuardrailTopic{
				{Name: aws.String("PersonalAttributes"), Type: types.GuardrailTopicTypeDeny, Action: types.GuardrailTopicPolicyActionBlocked},
			}},
			WordPolicy: &types.GuardrailWordPolicyAssessment{CustomWords: []types.GuardrailCustomWord{
				{Match: aws.String("women"), Action: types.GuardrailWordPolicyActionBlocked},
			}},
		},
		"drunk guests": {
			ContentPolicy: &types.GuardrailContentPolicyAssessment{Filters: []types.GuardrailContentFilter{
				{Type: types.GuardrailContentFilterTypeMisconduct, Action: types.GuardrailContentPolicyActionBlocked},
				{Type: types.GuardrailContentFilterTypeHate, Action: types.GuardrailContentPolicyAction("NONE")},
			}},
			WordPolicy: &types.GuardrailWordPolicyAssessment{ManagedWordLists: []types.GuardrailManagedWord{
				{Match: aws.String("Drunk"), Type: types.GuardrailManagedWordTypeProfanity, Action: types.GuardrailWordPolicyActionBlocked},
			}},
		},
	})}
	c := screenClient(rt)
	terms := []string{"keynote speech", "women networking", "drum circle", "drunk guests", "registration desk"}

	res, err := c.Screen(context.Background(), terms)
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"keynote speech", "drum circle", "registration desk"}; !reflect.DeepEqual(res.Allowed, want) {
		t.Errorf("Allowed = %q, want %q", res.Allowed, want)
	}
	wantBlocked := map[string]string{
		"women networking": "topic:PersonalAttributes,word:women",
		"drunk guests":     "content:MISCONDUCT,profanity:drunk",
	}
	if !reflect.DeepEqual(res.Blocked, wantBlocked) {
		t.Errorf("Blocked = %q, want %q", res.Blocked, wantBlocked)
	}
	if res.Calls <= 1 {
		t.Errorf("Calls = %d, want > 1 (batch had to be split)", res.Calls)
	}
	if res.Calls != len(rt.guardrailInputs) {
		t.Errorf("Calls = %d but ApplyGuardrail was called %d times", res.Calls, len(rt.guardrailInputs))
	}
	for _, in := range rt.guardrailInputs {
		if in.Source != types.GuardrailContentSourceOutput {
			t.Errorf("source = %q, want OUTPUT", in.Source)
		}
		if aws.ToString(in.GuardrailIdentifier) != "gr-123" || aws.ToString(in.GuardrailVersion) != "3" {
			t.Errorf("guardrail = %s@%s, want gr-123@3", aws.ToString(in.GuardrailIdentifier), aws.ToString(in.GuardrailVersion))
		}
	}
}

func TestScreen_AllAllowedIsOneCall(t *testing.T) {
	rt := &fakeRuntime{applyGuardrail: fakeGuardrail(nil)}
	res, err := screenClient(rt).Screen(context.Background(), []string{"a b", "c d", "a b", "", "  ", "e f"})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"a b", "c d", "e f"}; !reflect.DeepEqual(res.Allowed, want) {
		t.Errorf("Allowed = %q, want de-duplicated %q", res.Allowed, want)
	}
	if res.Calls != 1 || len(res.Blocked) != 0 {
		t.Errorf("Calls = %d, Blocked = %v; want 1 call and none blocked", res.Calls, res.Blocked)
	}
	if got := aws.ToString(rt.guardrailInputs[0].Content[0].(*types.GuardrailContentBlockMemberText).Value.Text); got != "a b\nc d\ne f" {
		t.Errorf("batch text = %q, want newline-joined terms", got)
	}
}

func TestScreen_BatchesStayUnder900Chars(t *testing.T) {
	rt := &fakeRuntime{applyGuardrail: fakeGuardrail(nil)}
	var terms []string
	for i := 0; i < 200; i++ {
		terms = append(terms, fmt.Sprintf("term number %04d", i)) // 16 chars
	}
	res, err := screenClient(rt).Screen(context.Background(), terms)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Allowed) != 200 {
		t.Errorf("Allowed = %d, want 200", len(res.Allowed))
	}
	seen := 0
	for _, in := range rt.guardrailInputs {
		text := aws.ToString(in.Content[0].(*types.GuardrailContentBlockMemberText).Value.Text)
		if n := utf8.RuneCountInString(text); n > MaxBatchChars {
			t.Errorf("batch of %d chars exceeds %d", n, MaxBatchChars)
		}
		seen += len(strings.Split(text, "\n"))
	}
	// 16-char terms + newline: 53 per batch (53*17-1 = 900).
	if res.Calls != 4 || seen != 200 {
		t.Errorf("Calls = %d covering %d terms, want 4 calls covering 200", res.Calls, seen)
	}
}

func TestScreen_NotConfiguredFailsClosed(t *testing.T) {
	rt := &fakeRuntime{}
	c := newTestClient(rt, Config{})
	if _, err := c.Screen(context.Background(), []string{"keynote speech"}); !errors.Is(err, ErrGuardrailNotConfigured) {
		t.Fatalf("err = %v, want ErrGuardrailNotConfigured", err)
	}
	if len(rt.guardrailInputs) != 0 {
		t.Error("no ApplyGuardrail call may be made without a guardrail")
	}
}

func TestScreen_EmptyInputMakesNoCall(t *testing.T) {
	rt := &fakeRuntime{}
	res, err := screenClient(rt).Screen(context.Background(), nil)
	if err != nil || res.Calls != 0 || len(res.Allowed) != 0 {
		t.Errorf("Screen(nil) = %+v, %v; want empty result, no calls", res, err)
	}
}

func TestScreen_ErrorFromAnyHalfFailsTheCall(t *testing.T) {
	var calls atomic.Int32
	rt := &fakeRuntime{applyGuardrail: func(in *bedrockruntime.ApplyGuardrailInput) (*bedrockruntime.ApplyGuardrailOutput, error) {
		if calls.Add(1) == 1 {
			return &bedrockruntime.ApplyGuardrailOutput{Action: types.GuardrailActionGuardrailIntervened}, nil
		}
		return nil, &types.ValidationException{Message: aws.String("bad")}
	}}
	_, err := screenClient(rt).Screen(context.Background(), []string{"a b", "c d"})
	var ve *types.ValidationException
	if !errors.As(err, &ve) {
		t.Errorf("err = %v, want the ValidationException", err)
	}
}

func TestPackBatches(t *testing.T) {
	long := strings.Repeat("x", 950)
	got := packBatches([]string{"aaaa", "bbbb", long, "cccc"}, 9)
	want := [][]string{{"aaaa", "bbbb"}, {long}, {"cccc"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("packBatches = %q, want %q", got, want)
	}
}

func TestBlockReasonsUnknownWhenNoPolicyNamed(t *testing.T) {
	if got := blockReasons(nil); got != "unknown" {
		t.Errorf("blockReasons(nil) = %q, want unknown", got)
	}
}
