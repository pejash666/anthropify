//go:build e2e

package e2e

import (
	"context"
	"testing"

	"github.com/shahao/anthropify/e2e/helpers"
)

// TestE2E_GLM_BasicStream is the smoke test for chat_completions_glm.
func TestE2E_GLM_BasicStream(t *testing.T) {
	cli, model := newGLMClient(t)
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

// TestE2E_GLM_ToolUseSingleRound expects a tool_use block.
func TestE2E_GLM_ToolUseSingleRound(t *testing.T) {
	cli, model := newGLMClient(t)
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

// TestE2E_GLM_ToolUseMultiRound submits a synthetic tool_result.
func TestE2E_GLM_ToolUseMultiRound(t *testing.T) {
	cli, model := newGLMClient(t)
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

// TestE2E_GLM_NonStreaming exercises CreateMessage. The chat_completions
// adapter has no native non-streaming endpoint, so the top-level client
// drains Stream and assembles the message via assembleFromStream.
func TestE2E_GLM_NonStreaming(t *testing.T) {
	cli, model := newGLMClient(t)
	msg, err := cli.CreateMessage(context.Background(),
		helpers.BasicTextPrompt(model, 128))
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	helpers.AssertNonStreamingMessage(t, msg)
}

// TestE2E_GLM_StopReasonMapping exercises the stop-reason mapping path
// when the prompt is open-ended and max_tokens is tiny. Note: GLM-4.6
// does NOT bound reasoning tokens by max_tokens -- the provider may
// either (a) eventually hit its internal cap and return
// finish_reason="length" (mapped to max_tokens) or (b) happen to
// finish its reasoning + answer inside the cap and return
// finish_reason="stop" (mapped to end_turn). Both outcomes confirm
// the chat_completions converter mapped the stop reason correctly;
// only "unknown" would indicate a real adapter bug. Direct curl
// probes against GLM with max_tokens=5..16 and the same prompt
// observed completion_tokens between ~1300 and >1900 of which the
// vast majority were reasoning_tokens, with finish_reason cycling
// between "length" and "stop" across runs.
func TestE2E_GLM_StopReasonMapping(t *testing.T) {
	cli, model := newGLMClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.MaxTokensClipPrompt(model))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertStopReason(t, sum, "max_tokens", "end_turn")
}
