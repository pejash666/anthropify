//go:build e2e

package e2e

import (
	"context"
	"testing"

	"github.com/shahao/anthropify/e2e/helpers"
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
