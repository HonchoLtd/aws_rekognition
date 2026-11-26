# suggestaws

System A of *Smart Search Suggestions: Clustering & Tracking Design*: a
stateless Go package that only talks to AWS. It proposes search terms for
numbered groups of photos (Nova 2 Lite), screens them (Bedrock Guardrails)
and embeds them (Nova Multimodal Embeddings). It has no database and no
clustering, and it never decides which term wins. All of that is System B
(the suggestion service).

| Operation | AWS call | Returns |
|---|---|---|
| `ProposeTerms` | Bedrock Runtime `Converse`, model `NOVA_LITE_MODEL_ID`, forced tool `submit_terms` | raw terms per group number, input/output tokens |
| `Screen` | `ApplyGuardrail` (source `OUTPUT`), batched ≤ 900 chars, split on intervention | allowed terms (input order), blocked terms with reasons, billed calls |
| `EmbedText` | `InvokeModel` on `NOVA_MODEL_ID`, purpose `IMAGE_RETRIEVAL` | raw 1024-d vector (B normalises) |
| `EnsureGuardrail` | `ListGuardrails` + `CreateGuardrail` / `UpdateGuardrail` | guardrail ID and version (admin only) |

## Using it from System B

```go
awsCfg, err := suggestaws.NewAWSConfig(ctx, cfg.Region) // tuned HTTP client, SDK retries off
cfg, err := suggestaws.ConfigFromEnv()
client, err := suggestaws.NewFromAWSConfig(awsCfg, cfg)

ctx, cancel := context.WithTimeout(ctx, 300*time.Second) // GENERATION_TIMEOUT for the cycle
defer cancel()

res, err := client.ProposeTerms(ctx, suggestaws.ProposeRequest{
    Context: []suggestaws.NamedPhoto{{Term: "drum circle", JPEG: chipPhoto}},
    Groups: []suggestaws.Group{ // largest first; Number is the 1-based position, not a cluster ID
        {Number: 1, Size: 91, JPEGs: reps1},
        {Number: 2, Size: 33, JPEGs: reps2, Avoid: []string{"panel discussion"}},
    },
    Temperature: 0, // 0.7 on retries
})
// res.Terms[1] = raw terms, best first; res.Terms[2] = []string{} means Lite refused.
// B normalises (lowercase, 2–3 words), de-duplicates and drops avoided terms.

screened, err := client.Screen(ctx, normalisedTerms)
vectors, err := suggestaws.EmbedTexts(ctx, client, screened.Allowed, 4)
```

B maps group numbers back to its own cluster IDs. Depend on the `Client`
interface so B's tests can use a fake.

## Errors

| Error | Meaning | B should |
|---|---|---|
| `ErrRetryable` | throttling, 5xx, `ModelNotReady`, connection error or read timeout, still failing after 6 attempts (backoff 1 s → 30 s cap, jittered) | fail the cycle; the next trigger retries |
| `ErrAuth` | `AccessDenied`, `UnrecognizedClient`, `InvalidSignature` (no retry) | fail the cycle and alert |
| `ErrGuardrailNotConfigured` | `GUARDRAIL_ID` is empty | turn suggestions off (fail closed) |
| `ErrUnexpectedResponse` | e.g. no `submit_terms` tool call; token counts are still returned | treat as a failed attempt |
| `ErrInvalidRequest` | bad input (no groups, wrong numbering, empty image, temperature outside 0–1) | fix the caller |

Every error wraps the original AWS error, so `errors.As` still works. The
caller's context ending is never retried.

## Configuration

| Env | Default | Notes |
|---|---|---|
| `AWS_REGION` | `us-east-1` | |
| `NOVA_LITE_MODEL_ID` | `us.amazon.nova-2-lite-v1:0` | cross-region inference profile |
| `NOVA_MODEL_ID` | `amazon.nova-2-multimodal-embeddings-v1:0` | |
| `NOVA_DIM` | `1024` | must match the stored image vectors |
| `GUARDRAIL_ID` | none | without it, `Screen` fails closed |
| `GUARDRAIL_VERSION` | `DRAFT` | pin a numbered version in production |

HTTP client (`NewHTTPClient`): connect timeout 10 s, response timeout 60 s,
TCP keepalive, 32 connections. The prompt texts are exported
(`PromptIntro`, `PromptInstructions`, …) so B can include them in its
settings fingerprint.

IAM: `bedrock:InvokeModel` on both models and the inference profile,
`bedrock:ApplyGuardrail`. The admin command also needs
`bedrock:ListGuardrails`, `CreateGuardrail`, `UpdateGuardrail`, and
`CreateGuardrailVersion` when run with `-publish`.

## Commands

```sh
# Create or update the guardrail (section 3.3 policy) and print GUARDRAIL_ID / GUARDRAIL_VERSION
go run ./cmd/suggestguardrail -env facev2/.env [-publish]

# One real call of each operation (well under $0.001)
go run ./cmd/suggestawssmoketest -env facev2/.env -input ./cmd/input_image -guardrail-id <id>
```

## Tests

`go test ./suggestaws/` runs against a mocked SDK, with no AWS calls. The tests cover the
System A rows of section 12.1: the forced tool and content order, avoid lists
and the context block, usage parsing, refusal as an empty slice, guardrail
batching (≤ 900 chars) and splitting with reasons, throttling →
`ErrRetryable` after 6 attempts, and `AccessDenied` → `ErrAuth` at once.
