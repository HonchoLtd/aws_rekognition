package suggestaws

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func toolAnswer(input any, in, out int32) *bedrockruntime.ConverseOutput {
	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				Name:      aws.String(ToolName),
				ToolUseId: aws.String("t1"),
				Input:     document.NewLazyDocument(input),
			}}},
		}},
		StopReason: types.StopReasonToolUse,
		Usage:      &types.TokenUsage{InputTokens: aws.Int32(in), OutputTokens: aws.Int32(out)},
	}
}

func okConverse(input any) func(*bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	return func(*bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
		return toolAnswer(input, 1234, 56), nil
	}
}

var (
	jpegA = []byte{0xff, 0xd8, 'a'}
	jpegB = []byte{0xff, 0xd8, 'b'}
	jpegC = []byte{0xff, 0xd8, 'c'}
	jpegD = []byte{0xff, 0xd8, 'd'}
)

// describe flattens content blocks into "text:<s>" / "image:<bytes>" so the
// order can be compared in one assertion.
func describe(t *testing.T, blocks []types.ContentBlock) []string {
	t.Helper()
	var out []string
	for _, b := range blocks {
		switch v := b.(type) {
		case *types.ContentBlockMemberText:
			out = append(out, "text:"+v.Value)
		case *types.ContentBlockMemberImage:
			if v.Value.Format != types.ImageFormatJpeg {
				t.Errorf("image format = %q, want jpeg", v.Value.Format)
			}
			src, ok := v.Value.Source.(*types.ImageSourceMemberBytes)
			if !ok {
				t.Fatalf("image source is %T, want bytes", v.Value.Source)
			}
			out = append(out, "image:"+string(src.Value))
		default:
			t.Fatalf("unexpected content block %T", b)
		}
	}
	return out
}

func TestProposeTerms_ForcedToolAndInferenceConfig(t *testing.T) {
	rt := &fakeRuntime{converse: okConverse(map[string]any{"groups": []any{}})}
	c := newTestClient(rt, Config{})
	_, err := c.ProposeTerms(context.Background(), ProposeRequest{
		Groups:      []Group{{Number: 1, Size: 40, JPEGs: [][]byte{jpegA}}},
		Temperature: 0.7,
	})
	if err != nil {
		t.Fatal(err)
	}
	in := rt.converseInputs[0]
	if got := aws.ToString(in.ModelId); got != DefaultLiteModelID {
		t.Errorf("ModelId = %q, want %q", got, DefaultLiteModelID)
	}
	if got := aws.ToInt32(in.InferenceConfig.MaxTokens); got != 800 {
		t.Errorf("maxTokens = %d, want 800", got)
	}
	if got := aws.ToFloat32(in.InferenceConfig.Temperature); got != 0.7 {
		t.Errorf("temperature = %v, want 0.7", got)
	}
	choice, ok := in.ToolConfig.ToolChoice.(*types.ToolChoiceMemberTool)
	if !ok || aws.ToString(choice.Value.Name) != "submit_terms" {
		t.Fatalf("toolChoice = %#v, want forced submit_terms", in.ToolConfig.ToolChoice)
	}
	if len(in.ToolConfig.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(in.ToolConfig.Tools))
	}
	spec := in.ToolConfig.Tools[0].(*types.ToolMemberToolSpec).Value
	if aws.ToString(spec.Name) != "submit_terms" || aws.ToString(spec.Description) != "Submit candidate search terms for each photo group." {
		t.Errorf("tool spec name/description = %q / %q", aws.ToString(spec.Name), aws.ToString(spec.Description))
	}
	raw, err := spec.InputSchema.(*types.ToolInputSchemaMemberJson).Value.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	var got, want any
	_ = json.Unmarshal(raw, &got)
	_ = json.Unmarshal([]byte(`{"type":"object","required":["groups"],"properties":{"groups":{"type":"array","items":{"type":"object","required":["group","terms"],"properties":{"group":{"type":"integer"},"terms":{"type":"array","items":{"type":"string"}}}}}}}`), &want)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tool schema = %s", raw)
	}
	if len(in.Messages) != 1 || in.Messages[0].Role != types.ConversationRoleUser {
		t.Errorf("want exactly one user message, got %d", len(in.Messages))
	}
}

func TestProposeTerms_ContentOrderWithContextAndAvoid(t *testing.T) {
	rt := &fakeRuntime{converse: okConverse(map[string]any{"groups": []any{}})}
	c := newTestClient(rt, Config{})
	_, err := c.ProposeTerms(context.Background(), ProposeRequest{
		Context: []NamedPhoto{{Term: "drum circle", JPEG: jpegA}},
		Groups: []Group{
			{Number: 1, Size: 91, JPEGs: [][]byte{jpegB, jpegC}, Avoid: []string{"women networking", "panel discussion"}},
			{Number: 2, Size: 33, JPEGs: [][]byte{jpegD}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := describe(t, rt.converseInputs[0].Messages[0].Content)
	want := []string{
		"text:" + PromptIntro,
		"text:These parts of the album already have suggestions; they are shown for context only, do not name them again:",
		`text:Already named "drum circle":`,
		"image:" + string(jpegA),
		"text:Group 1 (91 photos): do not suggest these rejected terms or close variants: women networking, panel discussion.",
		"image:" + string(jpegB),
		"image:" + string(jpegC),
		"text:Group 2 (33 photos):",
		"image:" + string(jpegD),
		"text:" + PromptInstructions + "\nOnly answer for the numbered groups; their terms must be distinct from the already-named parts.",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("content blocks:\n got %q\nwant %q", got, want)
	}
}

func TestProposeTerms_NoContextBlockWhenEmpty(t *testing.T) {
	rt := &fakeRuntime{converse: okConverse(map[string]any{"groups": []any{}})}
	c := newTestClient(rt, Config{})
	if _, err := c.ProposeTerms(context.Background(), ProposeRequest{Groups: []Group{{Number: 1, Size: 8, JPEGs: [][]byte{jpegA}}}}); err != nil {
		t.Fatal(err)
	}
	got := describe(t, rt.converseInputs[0].Messages[0].Content)
	want := []string{"text:" + PromptIntro, "text:Group 1 (8 photos):", "image:" + string(jpegA), "text:" + PromptInstructions}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("content blocks:\n got %q\nwant %q", got, want)
	}
	for _, s := range got {
		if strings.Contains(s, "already-named") || strings.Contains(s, "already have suggestions") {
			t.Errorf("context text present without context: %q", s)
		}
	}
}

func TestProposeTerms_ParsesTermsUsageAndRefusal(t *testing.T) {
	rt := &fakeRuntime{converse: okConverse(map[string]any{"groups": []any{
		map[string]any{"group": 1, "terms": []any{"Keynote Speech!", "main stage talks", "audience listening"}},
		map[string]any{"group": 2, "terms": []any{}},
		map[string]any{"group": 7, "terms": []any{"out of range"}},
		map[string]any{"group": 0, "terms": []any{"also out of range"}},
		map[string]any{"group": 1, "terms": []any{"duplicate ignored"}},
	}})}
	c := newTestClient(rt, Config{})
	res, err := c.ProposeTerms(context.Background(), ProposeRequest{Groups: []Group{
		{Number: 1, Size: 50, JPEGs: [][]byte{jpegA}},
		{Number: 2, Size: 30, JPEGs: [][]byte{jpegB}},
		{Number: 3, Size: 20, JPEGs: [][]byte{jpegC}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	want := map[int][]string{
		1: {"Keynote Speech!", "main stage talks", "audience listening"}, // raw: B normalises
		2: {},
	}
	if !reflect.DeepEqual(res.Terms, want) {
		t.Errorf("Terms = %#v, want %#v", res.Terms, want)
	}
	if refused, ok := res.Terms[2]; !ok || refused == nil || len(refused) != 0 {
		t.Errorf("group 2 should be a non-nil empty slice (refusal), got %#v (present=%v)", refused, ok)
	}
	if _, ok := res.Terms[3]; ok {
		t.Error("group 3 was not answered and must be absent")
	}
	if res.InputTokens != 1234 || res.OutputTokens != 56 {
		t.Errorf("tokens = %d/%d, want 1234/56", res.InputTokens, res.OutputTokens)
	}
}

func TestProposeTerms_LenientGroupNumbersAndTerms(t *testing.T) {
	rt := &fakeRuntime{converse: okConverse(map[string]any{"groups": []any{
		map[string]any{"group": "1", "terms": "stage show"},
		map[string]any{"group": 2.0, "terms": nil},
	}})}
	c := newTestClient(rt, Config{})
	res, err := c.ProposeTerms(context.Background(), ProposeRequest{Groups: []Group{{Number: 1, Size: 9}, {Number: 2, Size: 9}}})
	if err != nil {
		t.Fatal(err)
	}
	want := map[int][]string{1: {"stage show"}, 2: {}}
	if !reflect.DeepEqual(res.Terms, want) {
		t.Errorf("Terms = %#v, want %#v", res.Terms, want)
	}
}

func TestProposeTerms_MissingToolBlockIsErrorButReportsTokens(t *testing.T) {
	rt := &fakeRuntime{converse: func(*bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
		return &bedrockruntime.ConverseOutput{
			Output: &types.ConverseOutputMemberMessage{Value: types.Message{
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "I think group 1 is a party"}},
			}},
			StopReason: types.StopReasonEndTurn,
			Usage:      &types.TokenUsage{InputTokens: aws.Int32(900), OutputTokens: aws.Int32(12)},
		}, nil
	}}
	c := newTestClient(rt, Config{})
	res, err := c.ProposeTerms(context.Background(), ProposeRequest{Groups: []Group{{Number: 1, Size: 9}}})
	if !errors.Is(err, ErrUnexpectedResponse) {
		t.Fatalf("err = %v, want ErrUnexpectedResponse", err)
	}
	if res.InputTokens != 900 || res.OutputTokens != 12 {
		t.Errorf("tokens = %d/%d, want 900/12 even on error", res.InputTokens, res.OutputTokens)
	}
	if len(rt.converseInputs) != 1 {
		t.Errorf("malformed output must not be retried, got %d calls", len(rt.converseInputs))
	}
}

func TestProposeTerms_Validation(t *testing.T) {
	c := newTestClient(&fakeRuntime{}, Config{})
	cases := map[string]ProposeRequest{
		"no groups":         {},
		"wrong numbering":   {Groups: []Group{{Number: 2, Size: 5}}},
		"temperature high":  {Groups: []Group{{Number: 1}}, Temperature: 1.5},
		"empty image":       {Groups: []Group{{Number: 1, JPEGs: [][]byte{{}}}}},
		"context w/o photo": {Groups: []Group{{Number: 1}}, Context: []NamedPhoto{{Term: "x y"}}},
	}
	for name, req := range cases {
		if _, err := c.ProposeTerms(context.Background(), req); !errors.Is(err, ErrInvalidRequest) {
			t.Errorf("%s: err = %v, want ErrInvalidRequest", name, err)
		}
	}
}
