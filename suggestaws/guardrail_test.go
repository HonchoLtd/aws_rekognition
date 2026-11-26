package suggestaws

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
)

type fakeAdmin struct {
	pages   []*bedrock.ListGuardrailsOutput
	listed  int
	created []*bedrock.CreateGuardrailInput
	updated []*bedrock.UpdateGuardrailInput
}

func (f *fakeAdmin) ListGuardrails(_ context.Context, in *bedrock.ListGuardrailsInput, _ ...func(*bedrock.Options)) (*bedrock.ListGuardrailsOutput, error) {
	page := f.pages[f.listed]
	f.listed++
	return page, nil
}

func (f *fakeAdmin) CreateGuardrail(_ context.Context, in *bedrock.CreateGuardrailInput, _ ...func(*bedrock.Options)) (*bedrock.CreateGuardrailOutput, error) {
	f.created = append(f.created, in)
	return &bedrock.CreateGuardrailOutput{GuardrailId: aws.String("new-id"), GuardrailArn: aws.String("arn:new"), Version: aws.String("DRAFT")}, nil
}

func (f *fakeAdmin) UpdateGuardrail(_ context.Context, in *bedrock.UpdateGuardrailInput, _ ...func(*bedrock.Options)) (*bedrock.UpdateGuardrailOutput, error) {
	f.updated = append(f.updated, in)
	return &bedrock.UpdateGuardrailOutput{GuardrailId: in.GuardrailIdentifier, GuardrailArn: aws.String("arn:old"), Version: aws.String("DRAFT")}, nil
}

func (f *fakeAdmin) CreateGuardrailVersion(_ context.Context, in *bedrock.CreateGuardrailVersionInput, _ ...func(*bedrock.Options)) (*bedrock.CreateGuardrailVersionOutput, error) {
	return &bedrock.CreateGuardrailVersionOutput{GuardrailId: in.GuardrailIdentifier, Version: aws.String("4")}, nil
}

func TestEnsureGuardrail_CreatesWhenMissing(t *testing.T) {
	f := &fakeAdmin{pages: []*bedrock.ListGuardrailsOutput{{Guardrails: []bedrocktypes.GuardrailSummary{{Name: aws.String("other"), Id: aws.String("x")}}}}}
	ref, err := EnsureGuardrail(context.Background(), f, "suggest")
	if err != nil {
		t.Fatal(err)
	}
	if !ref.Created || ref.ID != "new-id" || ref.Version != "DRAFT" || len(f.created) != 1 || len(f.updated) != 0 {
		t.Errorf("ref = %+v, created %d, updated %d; want one create", ref, len(f.created), len(f.updated))
	}
}

func TestEnsureGuardrail_UpdatesExistingAcrossPages(t *testing.T) {
	f := &fakeAdmin{pages: []*bedrock.ListGuardrailsOutput{
		{Guardrails: []bedrocktypes.GuardrailSummary{{Name: aws.String("other"), Id: aws.String("x")}}, NextToken: aws.String("p2")},
		{Guardrails: []bedrocktypes.GuardrailSummary{{Name: aws.String("suggest"), Id: aws.String("gr-9")}}},
	}}
	ref, err := EnsureGuardrail(context.Background(), f, "suggest")
	if err != nil {
		t.Fatal(err)
	}
	if ref.Created || ref.ID != "gr-9" || len(f.updated) != 1 || len(f.created) != 0 {
		t.Fatalf("ref = %+v; want an update of gr-9", ref)
	}
	if f.updated[0].TopicPolicyConfig == nil || f.updated[0].WordPolicyConfig == nil || f.updated[0].ContentPolicyConfig == nil {
		t.Error("update must send the full policy")
	}
}

func TestGuardrailPolicyMatchesDesign(t *testing.T) {
	p := GuardrailPolicy("suggest")
	if aws.ToString(p.BlockedInputMessaging) != "blocked" || aws.ToString(p.BlockedOutputsMessaging) != "blocked" {
		t.Error("blocked messaging must be \"blocked\"")
	}

	filters := map[bedrocktypes.GuardrailContentFilterType]bool{}
	for _, f := range p.ContentPolicyConfig.FiltersConfig {
		if f.InputStrength != bedrocktypes.GuardrailFilterStrengthHigh || f.OutputStrength != bedrocktypes.GuardrailFilterStrengthHigh {
			t.Errorf("filter %s not HIGH/HIGH", f.Type)
		}
		filters[f.Type] = true
	}
	for _, want := range []bedrocktypes.GuardrailContentFilterType{"SEXUAL", "VIOLENCE", "HATE", "INSULTS", "MISCONDUCT"} {
		if !filters[want] {
			t.Errorf("missing content filter %s", want)
		}
	}

	mw := p.WordPolicyConfig.ManagedWordListsConfig
	if len(mw) != 1 || mw[0].Type != bedrocktypes.GuardrailManagedWordsTypeProfanity {
		t.Error("want the PROFANITY managed word list")
	}
	words := map[string]bool{}
	for _, w := range p.WordPolicyConfig.WordsConfig {
		if words[aws.ToString(w.Text)] {
			t.Errorf("duplicate custom word %q", aws.ToString(w.Text))
		}
		words[aws.ToString(w.Text)] = true
	}
	for _, w := range []string{"women", "girls", "teenagers", "outfits", "clothing", "drugs", "guns", "victims", "idiots", "crap"} {
		if !words[w] {
			t.Errorf("custom word %q missing", w)
		}
	}
	for _, w := range []string{"white", "black", "old", "high", "strip", "dress"} {
		if words[w] {
			t.Errorf("ambiguous word %q must be left to the denied topics", w)
		}
	}

	topics := map[string]bedrocktypes.GuardrailTopicConfig{}
	for _, tc := range p.TopicPolicyConfig.TopicsConfig {
		topics[aws.ToString(tc.Name)] = tc
		if tc.Type != bedrocktypes.GuardrailTopicTypeDeny {
			t.Errorf("topic %s is not DENY", aws.ToString(tc.Name))
		}
		if n := len(aws.ToString(tc.Definition)); n > 200 {
			t.Errorf("topic %s definition is %d chars, AWS allows 200", aws.ToString(tc.Name), n)
		}
		if len(tc.Examples) != 5 {
			t.Errorf("topic %s has %d examples, want 5", aws.ToString(tc.Name), len(tc.Examples))
		}
	}
	if _, ok := topics["PersonalAttributes"]; !ok {
		t.Error("missing PersonalAttributes topic")
	}
	if _, ok := topics["UnsafeScenes"]; !ok {
		t.Error("missing UnsafeScenes topic")
	}
}

func TestPublishGuardrailVersion(t *testing.T) {
	v, err := PublishGuardrailVersion(context.Background(), &fakeAdmin{}, "gr-9", "launch")
	if err != nil || v != "4" {
		t.Errorf("PublishGuardrailVersion = %q, %v; want 4", v, err)
	}
}
