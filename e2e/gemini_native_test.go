//go:build e2e

package e2e

import (
	"context"
	"testing"

	"github.com/shahao/anthropify/e2e/helpers"
)

// TestE2E_Gemini_BasicStream is the smoke test for gemini_native.
func TestE2E_Gemini_BasicStream(t *testing.T) {
	cli, model := newGeminiClient(t)
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

// TestE2E_Gemini_ToolUseSingleRound expects a tool_use block.
func TestE2E_Gemini_ToolUseSingleRound(t *testing.T) {
	cli, model := newGeminiClient(t)
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

// TestE2E_Gemini_ToolUseMultiRound submits a synthetic tool_result.
func TestE2E_Gemini_ToolUseMultiRound(t *testing.T) {
	cli, model := newGeminiClient(t)
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

// TestE2E_Gemini_NonStreaming exercises CreateMessage. Gemini's native
// endpoint is stream-only, so the top-level client drains Stream and
// assembles the message via assembleFromStream.
func TestE2E_Gemini_NonStreaming(t *testing.T) {
	cli, model := newGeminiClient(t)
	msg, err := cli.CreateMessage(context.Background(),
		helpers.BasicTextPrompt(model, 128))
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	helpers.AssertNonStreamingMessage(t, msg)
}

// TestE2E_Gemini_StopReasonMapping asserts max_tokens clipping.
func TestE2E_Gemini_StopReasonMapping(t *testing.T) {
	cli, model := newGeminiClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.MaxTokensClipPrompt(model))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertStopReason(t, sum, "max_tokens")
}
