package suggestaws

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const (
	// ToolName is the only tool Nova Lite is given, and it is forced.
	ToolName = "submit_terms"
	// ProposeMaxTokens is the inferenceConfig.maxTokens of every call.
	ProposeMaxTokens = 800

	toolDescription = "Submit candidate search terms for each photo group."
)

// The prompt texts from section 3.2. Changing any of them changes what
// Lite proposes: fingerprint them with the settings (design doc
// section 10) so stored suggestions are invalidated.
const (
	PromptIntro = "Below are groups of visually similar photos from ONE photo album, " +
		"largest group first. People click a search term to jump to a group's photos, " +
		"so each term should name the group the way a guest would remember that part of the event."

	PromptContextHeader = "These parts of the album already have suggestions; " +
		"they are shown for context only, do not name them again:"

	PromptInstructions = `For each group, propose 3 search terms, best first, each from a different angle.
How to choose a term, in order of preference:
1. The moment or activity: what people are doing together (for example toasting, cheering, dancing, presenting, receiving awards, playing a game).
2. The part of the event or occasion the photos show (for example welcome dinner, prize giving, panel q&a, team photo).
3. A distinctive setting (for example rooftop terrace, outdoor stage, game arcade).
4. An object only when it is clearly the subject of most photos in the group (for example a trophy display or a wedding cake), never when it is furniture or background.
Avoid:
- generic furniture, fixtures or background: chairs, tables, walls, windows, lights, screens, carpets
- photo or visual details: colours, lighting, camera angle, things that are merely visible
- anything not clearly shown in most of the group's photos: no invented event names, purposes or outcomes
The examples above only illustrate the kind of term; do not reuse them unless they truly describe the photos.
Rules:
- exactly 2 or 3 words, lowercase, no punctuation
- say what sets the group apart from the OTHER groups; don't repeat the album-wide theme in every term
- never describe people's identity, gender, age, ethnicity, body or clothing
- never copy visible text and never describe what slides, banners or talks are about
- safe for work and family friendly; return an empty list for a group that is not
Call submit_terms with your answer.`

	PromptContextSuffix = "Only answer for the numbered groups; their terms must be distinct from the already-named parts."
)

// toolInputSchema is the JSON schema of submit_terms.
func toolInputSchema() map[string]any {
	return map[string]any{
		"type":     "object",
		"required": []any{"groups"},
		"properties": map[string]any{
			"groups": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type":     "object",
					"required": []any{"group", "terms"},
					"properties": map[string]any{
						"group": map[string]any{"type": "integer"},
						"terms": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
					},
				},
			},
		},
	}
}

// ProposeTerms implements Client. On a malformed answer the token counts
// are still filled in, because the call was billed.
func (c *AWSClient) ProposeTerms(ctx context.Context, req ProposeRequest) (ProposeResult, error) {
	if err := validateProposeRequest(req); err != nil {
		return ProposeResult{}, err
	}
	input := c.buildConverseInput(req)

	var out *bedrockruntime.ConverseOutput
	start := time.Now()
	err := withRetry(ctx, c.retry, "ProposeTerms", func() error {
		var callErr error
		out, callErr = c.rt.Converse(ctx, input)
		return callErr
	})
	attrs := []any{"op", "ProposeTerms", "model_id", c.cfg.LiteModelID, "groups", len(req.Groups), "context", len(req.Context),
		"images", countImages(req), "temperature", strconv.FormatFloat(float64(req.Temperature), 'f', -1, 32), "duration_ms", time.Since(start).Milliseconds()}
	if err != nil {
		getLogger().Error("bedrock converse failed", append(attrs, "error", err)...)
		return ProposeResult{}, fmt.Errorf("suggestaws: ProposeTerms: %w", err)
	}

	result, err := parseConverseOutput(out, len(req.Groups))
	attrs = append(attrs, "input_tokens", result.InputTokens, "output_tokens", result.OutputTokens, "stop_reason", out.StopReason)
	if err != nil {
		getLogger().Error("nova lite answer could not be parsed", append(attrs, "error", err)...)
		return result, fmt.Errorf("suggestaws: ProposeTerms: %w", err)
	}
	getLogger().Info("bedrock converse", append(attrs, "answered_groups", len(result.Terms))...)
	return result, nil
}

func validateProposeRequest(req ProposeRequest) error {
	if len(req.Groups) == 0 {
		return fmt.Errorf("%w: ProposeTerms needs at least one group", ErrInvalidRequest)
	}
	if req.Temperature < 0 || req.Temperature > 1 {
		return fmt.Errorf("%w: temperature %v is outside [0, 1]", ErrInvalidRequest, req.Temperature)
	}
	for i, g := range req.Groups {
		if g.Number != i+1 {
			return fmt.Errorf("%w: group at index %d has Number %d, want %d (numbers are 1-based positions)", ErrInvalidRequest, i, g.Number, i+1)
		}
		for j, img := range g.JPEGs {
			if len(img) == 0 {
				return fmt.Errorf("%w: group %d image %d is empty", ErrInvalidRequest, g.Number, j)
			}
		}
	}
	for i, np := range req.Context {
		if strings.TrimSpace(np.Term) == "" || len(np.JPEG) == 0 {
			return fmt.Errorf("%w: context entry %d needs both a term and a photo", ErrInvalidRequest, i)
		}
	}
	return nil
}

func countImages(req ProposeRequest) int {
	n := len(req.Context)
	for _, g := range req.Groups {
		n += len(g.JPEGs)
	}
	return n
}

func (c *AWSClient) buildConverseInput(req ProposeRequest) *bedrockruntime.ConverseInput {
	return &bedrockruntime.ConverseInput{
		ModelId: aws.String(c.cfg.LiteModelID),
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: buildContent(req),
		}},
		InferenceConfig: &types.InferenceConfiguration{
			MaxTokens:   aws.Int32(ProposeMaxTokens),
			Temperature: aws.Float32(req.Temperature),
		},
		ToolConfig: &types.ToolConfiguration{
			Tools: []types.Tool{&types.ToolMemberToolSpec{Value: types.ToolSpecification{
				Name:        aws.String(ToolName),
				Description: aws.String(toolDescription),
				InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(toolInputSchema())},
			}}},
			ToolChoice: &types.ToolChoiceMemberTool{Value: types.SpecificToolChoice{Name: aws.String(ToolName)}},
		},
	}
}

// buildContent lays out the user message in the order of section 3.2:
// intro, optional context block, each group with its images, instructions.
func buildContent(req ProposeRequest) []types.ContentBlock {
	text := func(s string) types.ContentBlock { return &types.ContentBlockMemberText{Value: s} }
	image := func(b []byte) types.ContentBlock {
		return &types.ContentBlockMemberImage{Value: types.ImageBlock{
			Format: types.ImageFormatJpeg,
			Source: &types.ImageSourceMemberBytes{Value: b},
		}}
	}

	blocks := []types.ContentBlock{text(PromptIntro)}
	if len(req.Context) > 0 {
		blocks = append(blocks, text(PromptContextHeader))
		for _, np := range req.Context {
			blocks = append(blocks, text(fmt.Sprintf("Already named %q:", np.Term)), image(np.JPEG))
		}
	}
	for _, g := range req.Groups {
		blocks = append(blocks, text(groupHeader(g)))
		for _, img := range g.JPEGs {
			blocks = append(blocks, image(img))
		}
	}
	instructions := PromptInstructions
	if len(req.Context) > 0 {
		instructions += "\n" + PromptContextSuffix
	}
	return append(blocks, text(instructions))
}

func groupHeader(g Group) string {
	h := fmt.Sprintf("Group %d (%d photos):", g.Number, g.Size)
	if len(g.Avoid) > 0 {
		h += " do not suggest these rejected terms or close variants: " + strings.Join(g.Avoid, ", ") + "."
	}
	return h
}

// submitTermsInput is the tool input. Group numbers and term lists are
// decoded leniently: models occasionally send "2" for 2 or a bare string
// for a one-element list.
type submitTermsInput struct {
	Groups []struct {
		Group flexInt     `json:"group"`
		Terms flexStrings `json:"terms"`
	} `json:"groups"`
}

type flexInt int

func (f *flexInt) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(bytes.TrimSpace(b)), `"`)
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v != float64(int(v)) {
		return fmt.Errorf("group %s is not an integer", b)
	}
	*f = flexInt(v)
	return nil
}

type flexStrings []string

func (f *flexStrings) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if bytes.Equal(b, []byte("null")) {
		*f = []string{}
		return nil
	}
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*f = []string{s}
		return nil
	}
	var list []string
	if err := json.Unmarshal(b, &list); err != nil {
		return err
	}
	*f = list
	return nil
}

// parseConverseOutput reads the first toolUse block, keeps groups numbered
// 1..groupCount (first entry wins on duplicates) and the token usage.
func parseConverseOutput(out *bedrockruntime.ConverseOutput, groupCount int) (ProposeResult, error) {
	result := ProposeResult{}
	if out == nil {
		return result, fmt.Errorf("%w: nil Converse output", ErrUnexpectedResponse)
	}
	if out.Usage != nil {
		result.InputTokens = int(aws.ToInt32(out.Usage.InputTokens))
		result.OutputTokens = int(aws.ToInt32(out.Usage.OutputTokens))
	}

	msg, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		return result, fmt.Errorf("%w: Converse output has no message", ErrUnexpectedResponse)
	}
	var toolUse *types.ToolUseBlock
	for _, block := range msg.Value.Content {
		if tu, ok := block.(*types.ContentBlockMemberToolUse); ok {
			toolUse = &tu.Value
			break
		}
	}
	if toolUse == nil || toolUse.Input == nil {
		return result, fmt.Errorf("%w: no %s tool call in the answer (stop reason %q)", ErrUnexpectedResponse, ToolName, out.StopReason)
	}

	raw, err := toolUse.Input.MarshalSmithyDocument()
	if err != nil {
		return result, fmt.Errorf("%w: read tool input: %v", ErrUnexpectedResponse, err)
	}
	var input submitTermsInput
	if err := json.Unmarshal(raw, &input); err != nil {
		return result, fmt.Errorf("%w: decode tool input %s: %v", ErrUnexpectedResponse, truncate(string(raw), 300), err)
	}

	result.Terms = make(map[int][]string, len(input.Groups))
	for _, g := range input.Groups {
		n := int(g.Group)
		if n < 1 || n > groupCount {
			getLogger().Warn("nova lite answered for a group that was not asked", "group", n, "groups", groupCount)
			continue
		}
		if _, seen := result.Terms[n]; seen {
			continue
		}
		terms := []string(g.Terms)
		if terms == nil {
			terms = []string{}
		}
		result.Terms[n] = terms
	}
	return result, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
