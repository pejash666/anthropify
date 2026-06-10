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

// TestE2E_Anthropic_BasicStream exercises the passthrough adapter with
// a trivial single-turn prompt and verifies that the envelope, text,
// stop reason, and usage are all reported.
func TestE2E_Anthropic_BasicStream(t *testing.T) {
	cli, model := newAnthropicClient(t)
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

// TestE2E_Anthropic_ToolUseSingleRound asks the model to call the
// weather tool and asserts a tool_use block was emitted.
func TestE2E_Anthropic_ToolUseSingleRound(t *testing.T) {
	cli, model := newAnthropicClient(t)
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

// TestE2E_Anthropic_ToolUseMultiRound feeds a synthetic tool_result and
// expects the model to summarise it back to the user.
func TestE2E_Anthropic_ToolUseMultiRound(t *testing.T) {
	cli, model := newAnthropicClient(t)
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

// TestE2E_Anthropic_NonStreaming routes through CreateMessage; for the
// passthrough adapter this still hits the native non-streaming endpoint
// (no drain fallback).
func TestE2E_Anthropic_NonStreaming(t *testing.T) {
	cli, model := newAnthropicClient(t)
	msg, err := cli.CreateMessage(context.Background(),
		helpers.BasicTextPrompt(model, 128))
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	helpers.AssertNonStreamingMessage(t, msg)
}

// TestE2E_Anthropic_StopReasonMapping clips max_tokens=10 against a
// long prompt so the upstream returns stop_reason=max_tokens.
func TestE2E_Anthropic_StopReasonMapping(t *testing.T) {
	cli, model := newAnthropicClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.MaxTokensClipPrompt(model))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertStopReason(t, sum, "max_tokens")
}

// TestE2E_Anthropic_CacheControl_TopLevel proves the v0.2.0 Track C
// canonical helper anthropify.SetCacheControl actually reaches
// Anthropic and triggers prompt caching upstream.
//
// Strategy:
//  1. Build a system prompt deliberately long enough to clear
//     Anthropic's per-model minimum cacheable size (~1024 tokens for
//     sonnet / opus). The prompt body is fixed so two consecutive
//     calls cache-hit on the second.
//  2. Call CreateMessage twice back-to-back with the same payload and
//     SetCacheControl(CacheControlEphemeral).
//  3. Assert the upstream reports cache_creation_input_tokens > 0 on
//     the first call OR cache_read_input_tokens > 0 on the second.
//
// We OR the two signals because Anthropic's cache may already be warm
// from a previous test run within the 5-minute TTL window, in which
// case both calls are reads.
func TestE2E_Anthropic_CacheControl_TopLevel(t *testing.T) {
	cli, model := newAnthropicClient(t)
	ctx := context.Background()

	// ~2k tokens of stable English text. Repeated paragraphs are fine
	// for the cache test; the upstream caches by content prefix.
	var sysSb strings.Builder
	sysSb.WriteString("You are a meticulous historical-geography tutor. ")
	sysSb.WriteString("When the user asks about a topic, recall the relevant context below before answering.\n\n")
	for i := 0; i < 24; i++ {
		sysSb.WriteString("Context paragraph: The Roman Empire was a vast political " +
			"and economic entity that emerged from the Roman Republic, formally " +
			"established under Augustus in 27 BCE and persisting in the West " +
			"until 476 CE. At its territorial height under Trajan in 117 CE it " +
			"covered roughly five million square kilometres, encompassing the " +
			"entire Mediterranean basin, Gaul, Britannia, the Iberian peninsula, " +
			"the Balkans, Asia Minor, the Levant, Egypt and parts of North " +
			"Africa. Its administrative ingenuity included a professional " +
			"standing army, a network of paved roads, aqueducts, an integrated " +
			"legal code, a single trade currency, and a layered system of " +
			"provincial governance. ")
	}
	systemText := sysSb.String()

	build := func() anthropic.MessageNewParams {
		req := anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 64,
			System: []anthropic.TextBlockParam{
				{Text: systemText},
			},
			Messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock(
					"Reply with just the word OK.")),
			},
		}
		anthropify.SetCacheControl(&req, anthropify.CacheControl{
			Type: anthropify.CacheControlEphemeral,
		})
		return req
	}

	first, err := cli.CreateMessage(ctx, build())
	if err != nil {
		t.Fatalf("CreateMessage round 1: %v", err)
	}
	second, err := cli.CreateMessage(ctx, build())
	if err != nil {
		t.Fatalf("CreateMessage round 2: %v", err)
	}

	cc1 := first.Usage.CacheCreationInputTokens
	cr1 := first.Usage.CacheReadInputTokens
	cc2 := second.Usage.CacheCreationInputTokens
	cr2 := second.Usage.CacheReadInputTokens
	t.Logf("round1 usage: input=%d cache_create=%d cache_read=%d",
		first.Usage.InputTokens, cc1, cr1)
	t.Logf("round2 usage: input=%d cache_create=%d cache_read=%d",
		second.Usage.InputTokens, cc2, cr2)

	// Either the cache was created here (round 1 cc>0) or it was
	// already warm (any cache_read>0). Anything else means top-level
	// cache_control did not reach the upstream.
	if cc1 == 0 && cc2 == 0 && cr1 == 0 && cr2 == 0 {
		t.Fatalf("no cache activity reported on either round; "+
			"top-level cache_control did not take effect. "+
			"round1=%+v round2=%+v", first.Usage, second.Usage)
	}
}
