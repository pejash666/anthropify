//go:build e2e

package e2e

import (
	"context"
	"testing"

	"github.com/shahao/hybridstream/e2e/helpers"
)

// TestE2E_OpenAI_BasicStream is the smoke test for openai_responses.
func TestE2E_OpenAI_BasicStream(t *testing.T) {
	cli, model := newOpenAIClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.BasicTextPrompt(model, 256))
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

// TestE2E_OpenAI_ToolUseSingleRound expects a tool_use block.
func TestE2E_OpenAI_ToolUseSingleRound(t *testing.T) {
	cli, model := newOpenAIClient(t)
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

// TestE2E_OpenAI_ToolUseMultiRound submits a synthetic tool_result.
func TestE2E_OpenAI_ToolUseMultiRound(t *testing.T) {
	cli, model := newOpenAIClient(t)
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

// TestE2E_OpenAI_NonStreaming exercises CreateMessage. For adapters that
// lack a native non-streaming endpoint (OpenAI Responses, Gemini) the
// top-level client drains the stream and assembles the message.
func TestE2E_OpenAI_NonStreaming(t *testing.T) {
	cli, model := newOpenAIClient(t)
	msg, err := cli.CreateMessage(context.Background(),
		helpers.BasicTextPrompt(model, 128))
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	helpers.AssertNonStreamingMessage(t, msg)
}

// TestE2E_OpenAI_StopReasonMapping asserts max_tokens clipping.
func TestE2E_OpenAI_StopReasonMapping(t *testing.T) {
	cli, model := newOpenAIClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.MaxTokensClipPrompt(model))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertStopReason(t, sum, "max_tokens")
}
