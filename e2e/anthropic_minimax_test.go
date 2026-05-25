//go:build e2e

package e2e

import (
	"context"
	"testing"

	"github.com/shahao/anthropify/e2e/helpers"
)

// TestE2E_MiniMax_BasicStream exercises the anthropic adapter pointed
// at MiniMax's Anthropic-compatible endpoint with a trivial single-turn
// prompt.
func TestE2E_MiniMax_BasicStream(t *testing.T) {
	cli, model := newMiniMaxClient(t)
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

// TestE2E_MiniMax_NonStreaming routes through CreateMessage; MiniMax
// shares the anthropic adapter's native non-streaming endpoint.
func TestE2E_MiniMax_NonStreaming(t *testing.T) {
	cli, model := newMiniMaxClient(t)
	msg, err := cli.CreateMessage(context.Background(),
		helpers.BasicTextPrompt(model, 256))
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	helpers.AssertNonStreamingMessage(t, msg)
}

// TestE2E_MiniMax_ToolUseSingleRound asks MiniMax to call the weather
// tool and asserts a tool_use block was emitted.
func TestE2E_MiniMax_ToolUseSingleRound(t *testing.T) {
	cli, model := newMiniMaxClient(t)
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

// TestE2E_MiniMax_Thinking exercises extended-thinking pass-through.
// The anthropic adapter forwards SSE verbatim, so a `thinking` content
// block on MiniMax-M2.7 reaches the caller unchanged.
func TestE2E_MiniMax_Thinking(t *testing.T) {
	cli, model := newMiniMaxClient(t)
	stream, err := cli.CreateMessageStream(context.Background(),
		helpers.ThinkingPrompt(model, 2048))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertEventEnvelope(t, sum)
	helpers.AssertHasThinkingBlock(t, sum)
}
