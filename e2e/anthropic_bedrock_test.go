//go:build e2e

package e2e

import (
	"context"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	anthropify "github.com/pejash666/anthropify"
	"github.com/pejash666/anthropify/e2e/helpers"
)

// TestE2E_Bedrock_BasicStream is the smoke test for the AWS Bedrock
// backend (anthropic:bedrock). The wire protocol is byte-identical to
// Direct/MiniMax -- only SigV4 signing, URL rewrite, and AWS
// event-stream decoding differ, all delegated to anthropic-sdk-go's
// bedrock subpackage (see docs/design/v0.2.0-aws-bedrock.md) -- so the
// assertions mirror TestE2E_Anthropic_BasicStream verbatim.
func TestE2E_Bedrock_BasicStream(t *testing.T) {
	cli, model := newBedrockClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.BasicTextPrompt(model, 1024))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertEventEnvelope(t, sum)
	helpers.AssertNonEmptyText(t, sum)
	helpers.AssertStopReason(t, sum, "end_turn", "stop_sequence")
	helpers.AssertInputTokensPositive(t, sum)
}

// TestE2E_Bedrock_NonStreaming routes through CreateMessage; Bedrock
// shares the anthropic adapter's native non-streaming endpoint
// (bedrockClient.Messages.New), same as Direct/MiniMax.
func TestE2E_Bedrock_NonStreaming(t *testing.T) {
	cli, model := newBedrockClient(t)
	msg, err := cli.CreateMessage(context.Background(),
		helpers.BasicTextPrompt(model, 128))
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	helpers.AssertNonStreamingMessage(t, msg)
}

// TestE2E_Bedrock_ToolUseSingleRound expects a tool_use block.
func TestE2E_Bedrock_ToolUseSingleRound(t *testing.T) {
	cli, model := newBedrockClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.WeatherToolPrompt(model))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertEventEnvelope(t, sum)
	_ = helpers.AssertHasToolUse(t, sum)
	helpers.AssertStopReason(t, sum, "tool_use")
}

// TestE2E_Bedrock_ToolUseMultiRound submits a synthetic tool_result and
// expects the model to summarise it back to the user.
func TestE2E_Bedrock_ToolUseMultiRound(t *testing.T) {
	cli, model := newBedrockClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.WeatherToolFollowUpPrompt(model, "toolu_test_round2"))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertEventEnvelope(t, sum)
	helpers.AssertNonEmptyText(t, sum)
	helpers.AssertStopReason(t, sum, "end_turn")
}

// TestE2E_Bedrock_StopReasonMapping clips max_tokens=16 against an
// open-ended prompt so the upstream returns stop_reason=max_tokens.
func TestE2E_Bedrock_StopReasonMapping(t *testing.T) {
	cli, model := newBedrockClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.MaxTokensClipPrompt(model))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertStopReason(t, sum, "max_tokens")
}

// TestE2E_Bedrock_CacheControl_TopLevel_Rejected documents actual
// observed behavior for the top-level (automatic-caching) form of
// cache_control on Bedrock: it is forwarded verbatim by anthropify,
// but Bedrock's schema hard-rejects it.
//
// Live verification against us-east-1 /
// us.anthropic.claude-haiku-4-5-20251001-v1:0 (both via raw `aws
// bedrock-runtime invoke-model` and through this client) shows the
// top-level cache_control field anthropify.SetCacheControl installs
// (a bare {"cache_control":{"type":"ephemeral"}} at the request root,
// mirroring native Anthropic's automatic-caching mode) is REJECTED by
// Bedrock with a hard 400:
//
//	ValidationException: cache_control: Extra inputs are not permitted
//
// This is not a soft "unsupported, ignored" no-op like Azure/OpenAI --
// it is a schema-validation error that breaks the entire request. This
// is a Bedrock-side platform limitation, not an anthropify bug: the
// per-block form (see TestE2E_Bedrock_CacheControl_PerBlock_CacheHit
// below) is the supported path for prompt caching on Bedrock.
//
// This test asserts the actual (negative) behavior so a regression
// that silently starts succeeding -- or a regression that starts
// failing for a different reason -- is caught.
func TestE2E_Bedrock_CacheControl_TopLevel_Rejected(t *testing.T) {
	cli, model := newBedrockClient(t)
	ctx := context.Background()

	req := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 16,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(
				"Reply with just the word OK.")),
		},
	}
	anthropify.SetCacheControl(&req, anthropify.CacheControl{
		Type: anthropify.CacheControlEphemeral,
	})

	_, err := cli.CreateMessage(ctx, req)
	if err == nil {
		t.Fatalf("expected Bedrock to reject top-level cache_control with a " +
			"400 ValidationException (observed live behavior as of this test's " +
			"writing); got nil error instead -- either Bedrock started accepting " +
			"the field (update README.md's cache_control matrix + this test) or " +
			"anthropify started stripping it for the bedrock branch (also update " +
			"this test to assert success)")
	}
	if !strings.Contains(err.Error(), "cache_control") &&
		!strings.Contains(err.Error(), "400") &&
		!strings.Contains(err.Error(), "ValidationException") {
		t.Fatalf("CreateMessage failed as expected but with an unrecognized "+
			"error shape (want a cache_control-related 400/ValidationException): %v", err)
	}
	t.Logf("confirmed: Bedrock rejects top-level cache_control as documented "+
		"by this test (error: %v)", err)
}

// romanEmpireParagraph is a filler paragraph repeated N times to build
// system prompts of a controlled, roughly-known token size for the
// cache-threshold tests below. ~165 words / ~660 chars per repeat.
const romanEmpireParagraph = "Context paragraph: The Roman Empire was a vast political " +
	"and economic entity that emerged from the Roman Republic, formally " +
	"established under Augustus in 27 BCE and persisting in the West " +
	"until 476 CE. At its territorial height under Trajan in 117 CE it " +
	"covered roughly five million square kilometres, encompassing the " +
	"entire Mediterranean basin, Gaul, Britannia, the Iberian peninsula, " +
	"the Balkans, Asia Minor, the Levant, Egypt and parts of North " +
	"Africa. Its administrative ingenuity included a professional " +
	"standing army, a network of paved roads, aqueducts, an integrated " +
	"legal code, a single trade currency, and a layered system of " +
	"provincial governance. "

// buildRomanEmpireSystemText assembles a system prompt of `repeat`
// copies of romanEmpireParagraph behind a short tutor preamble.
func buildRomanEmpireSystemText(repeat int) string {
	var sb strings.Builder
	sb.WriteString("You are a meticulous historical-geography tutor. ")
	sb.WriteString("When the user asks about a topic, recall the relevant context below before answering.\n\n")
	for i := 0; i < repeat; i++ {
		sb.WriteString(romanEmpireParagraph)
	}
	return sb.String()
}

// TestE2E_Bedrock_CacheControl_PerBlock_BelowThreshold exercises the
// per-block cache_control form (set directly on a content block via
// the SDK's SetExtraFields, independent of anthropify.SetCacheControl
// / the v0.2.0 top-level helper) with a system prompt of ~3500 tokens
// -- BELOW the per-model minimum cacheable-block size (Claude Haiku
// 4.5 requires >=4096 tokens per cache checkpoint on Bedrock). Unlike
// the top-level form (see TestE2E_Bedrock_CacheControl_TopLevel_Rejected),
// Bedrock's schema accepts per-block cache_control without erroring,
// but a prompt under the minimum-token threshold is never actually
// cached (cache_creation_input_tokens / cache_read_input_tokens stay
// at 0 on both rounds). That is expected, threshold-driven behavior --
// not a sign that Bedrock's prompt caching is broken or unsupported.
// See TestE2E_Bedrock_CacheControl_PerBlock_CacheHit below for the
// positive case once the prompt clears the threshold.
func TestE2E_Bedrock_CacheControl_PerBlock_BelowThreshold(t *testing.T) {
	cli, model := newBedrockClient(t)
	ctx := context.Background()

	systemText := buildRomanEmpireSystemText(24) // ~3500 tokens, below the 4096-token minimum

	sysBlock := anthropic.TextBlockParam{Text: systemText}
	sysBlock.SetExtraFields(map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	})

	build := func() anthropic.MessageNewParams {
		return anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 64,
			System:    []anthropic.TextBlockParam{sysBlock},
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock(
					"Reply with just the word OK.")),
			},
		}
	}

	first, err := cli.CreateMessage(ctx, build())
	if err != nil {
		t.Fatalf("CreateMessage round 1 (per-block cache_control): %v", err)
	}
	second, err := cli.CreateMessage(ctx, build())
	if err != nil {
		t.Fatalf("CreateMessage round 2 (per-block cache_control): %v", err)
	}

	helpers.AssertNonStreamingMessage(t, first)
	helpers.AssertNonStreamingMessage(t, second)

	t.Logf("round1 usage: input=%d cache_create=%d cache_read=%d",
		first.Usage.InputTokens, first.Usage.CacheCreationInputTokens, first.Usage.CacheReadInputTokens)
	t.Logf("round2 usage: input=%d cache_create=%d cache_read=%d",
		second.Usage.InputTokens, second.Usage.CacheCreationInputTokens, second.Usage.CacheReadInputTokens)

	if first.Usage.CacheCreationInputTokens == 0 && second.Usage.CacheReadInputTokens == 0 {
		t.Logf("no cache_creation/cache_read activity observed on either round; " +
			"this is EXPECTED here because the ~3500-token system prompt is below " +
			"the model's minimum cacheable-block size (Claude Haiku 4.5 = 4096 " +
			"tokens/checkpoint on Bedrock), not evidence that per-block caching is " +
			"unsupported -- see TestE2E_Bedrock_CacheControl_PerBlock_CacheHit for " +
			"the positive case above the threshold")
	}
}

// TestE2E_Bedrock_CacheControl_PerBlock_CacheHit is the regression
// test for the positive case: per-block cache_control DOES produce
// real cache hits on Bedrock once the cached block clears the model's
// minimum-token threshold (Claude Haiku 4.5 = 4096 tokens/checkpoint).
// The system prompt here (~9900 tokens, well above the 4096 floor
// with margin for tokenizer variance) is held constant across two
// rounds:
//
//   - Round 1 is expected to report cache_creation_input_tokens > 0
//     (the cache write).
//   - Round 2 is expected to report cache_read_input_tokens > 0 (the
//     cache read hitting the checkpoint written by round 1).
//
// This was cross-validated during the investigation both via a raw
// `aws bedrock-runtime invoke-model` call and through this anthropify
// Bedrock adapter against the same model/region, with both paths
// showing cache_read landing at ~= the cached prompt's token count on
// the second round -- i.e. anthropify's Bedrock wire path forwards
// per-block cache_control correctly and Bedrock/the model honour it.
func TestE2E_Bedrock_CacheControl_PerBlock_CacheHit(t *testing.T) {
	cli, model := newBedrockClient(t)
	ctx := context.Background()

	// 60 repeats of a ~660-char paragraph is comfortably >8192 tokens
	// (chars/4 estimate lands near 9900), giving headroom over the
	// 4096-token minimum even accounting for real-tokenizer variance.
	systemText := buildRomanEmpireSystemText(60)

	sysBlock := anthropic.TextBlockParam{Text: systemText}
	sysBlock.SetExtraFields(map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	})

	build := func() anthropic.MessageNewParams {
		return anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 64,
			System:    []anthropic.TextBlockParam{sysBlock},
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock(
					"Reply with just the word OK.")),
			},
		}
	}

	first, err := cli.CreateMessage(ctx, build())
	if err != nil {
		t.Fatalf("CreateMessage round 1 (per-block cache_control, above threshold): %v", err)
	}
	second, err := cli.CreateMessage(ctx, build())
	if err != nil {
		t.Fatalf("CreateMessage round 2 (per-block cache_control, above threshold): %v", err)
	}

	helpers.AssertNonStreamingMessage(t, first)
	helpers.AssertNonStreamingMessage(t, second)

	t.Logf("round1 usage: input=%d cache_create=%d cache_read=%d",
		first.Usage.InputTokens, first.Usage.CacheCreationInputTokens, first.Usage.CacheReadInputTokens)
	t.Logf("round2 usage: input=%d cache_create=%d cache_read=%d",
		second.Usage.InputTokens, second.Usage.CacheCreationInputTokens, second.Usage.CacheReadInputTokens)

	if first.Usage.CacheCreationInputTokens <= 0 {
		t.Fatalf("round 1 cache_creation_input_tokens = %d, want > 0 "+
			"(system prompt is above the model's minimum cacheable-block size; "+
			"if this regresses, either the prompt shrank below threshold or "+
			"per-block cache_control stopped being honoured by Bedrock/the model)",
			first.Usage.CacheCreationInputTokens)
	}
	if second.Usage.CacheReadInputTokens <= 0 {
		t.Fatalf("round 2 cache_read_input_tokens = %d, want > 0 "+
			"(expected a cache hit against the checkpoint written by round 1)",
			second.Usage.CacheReadInputTokens)
	}
}
