package suggestaws

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	bedrocktypes "github.com/aws/aws-sdk-go-v2/service/bedrock/types"
)

// DefaultGuardrailName is the name the admin command uses when none is
// given.
const DefaultGuardrailName = "smart-search-suggestions"

// GuardrailAdminAPI is the subset of the Bedrock control-plane client
// (*bedrock.Client) EnsureGuardrail and PublishGuardrailVersion use. It
// needs bedrock:ListGuardrails, CreateGuardrail, UpdateGuardrail (and
// CreateGuardrailVersion to publish).
type GuardrailAdminAPI interface {
	ListGuardrails(ctx context.Context, params *bedrock.ListGuardrailsInput, optFns ...func(*bedrock.Options)) (*bedrock.ListGuardrailsOutput, error)
	CreateGuardrail(ctx context.Context, params *bedrock.CreateGuardrailInput, optFns ...func(*bedrock.Options)) (*bedrock.CreateGuardrailOutput, error)
	UpdateGuardrail(ctx context.Context, params *bedrock.UpdateGuardrailInput, optFns ...func(*bedrock.Options)) (*bedrock.UpdateGuardrailOutput, error)
	CreateGuardrailVersion(ctx context.Context, params *bedrock.CreateGuardrailVersionInput, optFns ...func(*bedrock.Options)) (*bedrock.CreateGuardrailVersionOutput, error)
}

// GuardrailRef identifies the guardrail EnsureGuardrail created or
// updated. Version is "DRAFT" until PublishGuardrailVersion is called.
type GuardrailRef struct {
	ID      string
	ARN     string
	Version string
	Created bool // false when an existing guardrail was updated
}

// Words blocked as people descriptors (section 3.3). Ambiguous words
// (white, black, old, high, strip, dress) are deliberately left to the
// denied topics, which judge meaning.
var guardrailPeopleWords = []string{
	"woman", "women", "man", "men", "girl", "girls", "boy", "boys", "guy", "guys",
	"lady", "ladies", "male", "males", "female", "females", "kid", "kids", "child",
	"children", "baby", "babies", "toddler", "teen", "teens", "teenager", "teenagers",
	"elderly", "senior", "seniors", "young", "asian", "chinese", "indian", "malay",
	"caucasian", "skin", "fat", "thin", "skinny", "pretty", "beautiful", "handsome",
	"ugly", "attire", "outfit", "outfits", "clothing",
}

// Words blocked as unsafe (section 3.3).
var guardrailUnsafeWords = []string{
	"nude", "naked", "nudity", "topless", "lingerie", "underwear", "sexy", "sex",
	"sexual", "erotic", "porn", "stripper", "drunk", "intoxicated", "drug", "drugs",
	"weed", "cocaine", "marijuana", "stoned", "blood", "bloody", "gore", "brawl",
	"fight", "fighting", "gun", "guns", "violence", "violent", "dead", "death",
	"corpse", "injury", "injured", "victim", "victims", "crash", "stupid", "idiot",
	"idiots", "dumb", "damn", "hell", "crap",
}

const guardrailBlockedMessage = "blocked"

// GuardrailPolicy returns the full CreateGuardrail request for name. It is
// exported so the policy can be reviewed or diffed without calling AWS.
func GuardrailPolicy(name string) *bedrock.CreateGuardrailInput {
	filters := make([]bedrocktypes.GuardrailContentFilterConfig, 0, 5)
	for _, t := range []bedrocktypes.GuardrailContentFilterType{
		bedrocktypes.GuardrailContentFilterTypeSexual,
		bedrocktypes.GuardrailContentFilterTypeViolence,
		bedrocktypes.GuardrailContentFilterTypeHate,
		bedrocktypes.GuardrailContentFilterTypeInsults,
		bedrocktypes.GuardrailContentFilterTypeMisconduct,
	} {
		filters = append(filters, bedrocktypes.GuardrailContentFilterConfig{
			Type:           t,
			InputStrength:  bedrocktypes.GuardrailFilterStrengthHigh,
			OutputStrength: bedrocktypes.GuardrailFilterStrengthHigh,
		})
	}

	words := make([]bedrocktypes.GuardrailWordConfig, 0, len(guardrailPeopleWords)+len(guardrailUnsafeWords))
	for _, w := range append(append([]string{}, guardrailPeopleWords...), guardrailUnsafeWords...) {
		words = append(words, bedrocktypes.GuardrailWordConfig{Text: aws.String(w)})
	}

	return &bedrock.CreateGuardrailInput{
		Name:                    aws.String(name),
		Description:             aws.String("Screens smart search suggestion terms: unsafe scenes and descriptions of people."),
		BlockedInputMessaging:   aws.String(guardrailBlockedMessage),
		BlockedOutputsMessaging: aws.String(guardrailBlockedMessage),
		ContentPolicyConfig:     &bedrocktypes.GuardrailContentPolicyConfig{FiltersConfig: filters},
		WordPolicyConfig: &bedrocktypes.GuardrailWordPolicyConfig{
			ManagedWordListsConfig: []bedrocktypes.GuardrailManagedWordsConfig{{Type: bedrocktypes.GuardrailManagedWordsTypeProfanity}},
			WordsConfig:            words,
		},
		TopicPolicyConfig: &bedrocktypes.GuardrailTopicPolicyConfig{TopicsConfig: []bedrocktypes.GuardrailTopicConfig{
			{
				Name:       aws.String("PersonalAttributes"),
				Type:       bedrocktypes.GuardrailTopicTypeDeny,
				Definition: aws.String("Describing a person's gender, age, ethnicity, race, body, size, physical appearance or clothing."),
				Examples:   []string{"women networking", "elderly guests", "casual attire", "pretty bridesmaids", "plus size models"},
			},
			{
				Name: aws.String("UnsafeScenes"),
				Type: bedrocktypes.GuardrailTopicTypeDeny,
				Definition: aws.String("Scenes involving nudity or sexual activity, violence, fighting, weapons used against people, " +
					"injury, death, accidents with victims, intoxication, illegal drug use, or insulting people."),
				Examples: []string{"street brawl", "gun violence", "car crash victims", "drug party", "drunk guests"},
			},
		}},
	}
}

// EnsureGuardrail creates the guardrail called name with GuardrailPolicy,
// or, when one with that name exists, overwrites its DRAFT with the same
// policy. Run it from an admin command, not per request.
func EnsureGuardrail(ctx context.Context, api GuardrailAdminAPI, name string) (GuardrailRef, error) {
	if name == "" {
		return GuardrailRef{}, fmt.Errorf("%w: guardrail name is required", ErrInvalidRequest)
	}
	existing, err := findGuardrail(ctx, api, name)
	if err != nil {
		return GuardrailRef{}, fmt.Errorf("suggestaws: EnsureGuardrail: %w", err)
	}
	policy := GuardrailPolicy(name)

	if existing == nil {
		var out *bedrock.CreateGuardrailOutput
		err := withRetry(ctx, defaultRetryPolicy, "CreateGuardrail", func() error {
			var callErr error
			out, callErr = api.CreateGuardrail(ctx, policy)
			return callErr
		})
		if err != nil {
			return GuardrailRef{}, fmt.Errorf("suggestaws: EnsureGuardrail: create %q: %w", name, err)
		}
		ref := GuardrailRef{ID: aws.ToString(out.GuardrailId), ARN: aws.ToString(out.GuardrailArn), Version: aws.ToString(out.Version), Created: true}
		getLogger().Info("guardrail created", "name", name, "id", ref.ID, "version", ref.Version)
		return ref, nil
	}

	update := &bedrock.UpdateGuardrailInput{
		GuardrailIdentifier:              existing.Id,
		Name:                             policy.Name,
		Description:                      policy.Description,
		BlockedInputMessaging:            policy.BlockedInputMessaging,
		BlockedOutputsMessaging:          policy.BlockedOutputsMessaging,
		ContentPolicyConfig:              policy.ContentPolicyConfig,
		WordPolicyConfig:                 policy.WordPolicyConfig,
		TopicPolicyConfig:                policy.TopicPolicyConfig,
		SensitiveInformationPolicyConfig: policy.SensitiveInformationPolicyConfig,
		ContextualGroundingPolicyConfig:  policy.ContextualGroundingPolicyConfig,
	}
	var out *bedrock.UpdateGuardrailOutput
	err = withRetry(ctx, defaultRetryPolicy, "UpdateGuardrail", func() error {
		var callErr error
		out, callErr = api.UpdateGuardrail(ctx, update)
		return callErr
	})
	if err != nil {
		return GuardrailRef{}, fmt.Errorf("suggestaws: EnsureGuardrail: update %q: %w", name, err)
	}
	ref := GuardrailRef{ID: aws.ToString(out.GuardrailId), ARN: aws.ToString(out.GuardrailArn), Version: aws.ToString(out.Version)}
	getLogger().Info("guardrail updated", "name", name, "id", ref.ID, "version", ref.Version)
	return ref, nil
}

// PublishGuardrailVersion snapshots the guardrail's DRAFT as a numbered
// version and returns it, for pinning GUARDRAIL_VERSION in production.
func PublishGuardrailVersion(ctx context.Context, api GuardrailAdminAPI, guardrailID, description string) (string, error) {
	input := &bedrock.CreateGuardrailVersionInput{GuardrailIdentifier: aws.String(guardrailID)}
	if description != "" {
		input.Description = aws.String(description)
	}
	var out *bedrock.CreateGuardrailVersionOutput
	err := withRetry(ctx, defaultRetryPolicy, "CreateGuardrailVersion", func() error {
		var callErr error
		out, callErr = api.CreateGuardrailVersion(ctx, input)
		return callErr
	})
	if err != nil {
		return "", fmt.Errorf("suggestaws: PublishGuardrailVersion: %w", err)
	}
	return aws.ToString(out.Version), nil
}

// findGuardrail pages through ListGuardrails (which lists each guardrail's
// DRAFT) and returns the one called name, or nil.
func findGuardrail(ctx context.Context, api GuardrailAdminAPI, name string) (*bedrocktypes.GuardrailSummary, error) {
	var token *string
	for {
		var out *bedrock.ListGuardrailsOutput
		err := withRetry(ctx, defaultRetryPolicy, "ListGuardrails", func() error {
			var callErr error
			out, callErr = api.ListGuardrails(ctx, &bedrock.ListGuardrailsInput{NextToken: token, MaxResults: aws.Int32(100)})
			return callErr
		})
		if err != nil {
			return nil, err
		}
		for i := range out.Guardrails {
			if aws.ToString(out.Guardrails[i].Name) == name {
				return &out.Guardrails[i], nil
			}
		}
		if aws.ToString(out.NextToken) == "" {
			return nil, nil
		}
		token = out.NextToken
	}
}
