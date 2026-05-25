//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
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

// TestE2E_Gemini_ToolUseMultiRound performs a real two-round tool-use
// trip. Gemini 3.x rejects synthetic assistant turns whose tool_use
// blocks lack a `thought_signature`, so we cannot reuse the shared
// WeatherToolFollowUpPrompt fixture (which fabricates a fake tool
// call). Instead we:
//
//  1. Issue the single-round weather prompt and capture the live
//     tool_use block, including its provider-attached
//     thought_signature, from the response.
//  2. Replay the assistant turn unchanged via SetExtraFields and
//     append a tool_result; then stream round 2 and assert the model
//     produced a natural-language answer with stop_reason=end_turn.
//
// This mirrors examples/04-tool-use-portable/main.go.
func TestE2E_Gemini_ToolUseMultiRound(t *testing.T) {
	cli, model := newGeminiClient(t)
	ctx := context.Background()

	// Round 1: ask the question, capture the tool_use block.
	round1Params := helpers.WeatherToolPrompt(model)
	msg, err := cli.CreateMessage(ctx, round1Params)
	if err != nil {
		t.Fatalf("round1 CreateMessage: %v", err)
	}
	if msg == nil || len(msg.Content) == 0 {
		t.Fatalf("round1 returned empty message")
	}

	var toolUseID, toolName string
	var toolInput any
	var toolSig string
	for _, b := range msg.Content {
		if b.Type == "tool_use" {
			toolUseID = b.ID
			toolName = b.Name
			_ = json.Unmarshal(b.Input, &toolInput)
			toolSig = extractGeminiThoughtSignature(b.RawJSON())
			break
		}
	}
	if toolUseID == "" {
		t.Fatalf("round1 emitted no tool_use block; stop=%s blocks=%d",
			msg.StopReason, len(msg.Content))
	}

	// Build round 2: replay the original user turn, the assistant
	// turn (with thought_signature passed through ExtraFields), and
	// a synthetic tool_result.
	assistantContent := assistantBlocksFromMessage(msg)
	round2Params := round1Params
	round2Params.Messages = []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(
			"What is the weather in Beijing right now? Use the tool.")),
		{Role: anthropic.MessageParamRoleAssistant, Content: assistantContent},
		anthropic.NewUserMessage(anthropic.NewToolResultBlock(
			toolUseID,
			`{"city":"Beijing","temperature_c":21,"condition":"sunny"}`,
			false,
		)),
	}

	stream, err := cli.CreateMessageStream(ctx, round2Params)
	if err != nil {
		t.Fatalf("round2 CreateMessageStream: %v (tool_use_id=%s name=%s sig_present=%v)",
			err, toolUseID, toolName, toolSig != "")
	}
	defer stream.Close()
	sum := helpers.DrainStream(t, stream)
	helpers.AssertEventEnvelope(t, sum)
	helpers.AssertNonEmptyText(t, sum)
	helpers.AssertStopReason(t, sum, "end_turn")
}

// assistantBlocksFromMessage rebuilds an assistant turn as
// ContentBlockParamUnion so it can be appended to the next request.
// For Gemini 3.x tool_use blocks the upstream attaches a
// `thought_signature` that must round-trip; we extract it from the
// response block's raw JSON and replay it via SetExtraFields. Mirrors
// examples/04-tool-use-portable/main.go:assistantBlocks.
func assistantBlocksFromMessage(msg *anthropic.Message) []anthropic.ContentBlockParamUnion {
	out := make([]anthropic.ContentBlockParamUnion, 0, len(msg.Content))
	for _, b := range msg.Content {
		switch b.Type {
		case "text":
			out = append(out, anthropic.NewTextBlock(b.Text))
		case "tool_use":
			var input any
			_ = json.Unmarshal(b.Input, &input)
			block := anthropic.NewToolUseBlock(b.ID, input, b.Name)
			if sig := extractGeminiThoughtSignature(b.RawJSON()); sig != "" && block.OfToolUse != nil {
				block.OfToolUse.SetExtraFields(map[string]any{
					"thought_signature": sig,
				})
			}
			out = append(out, block)
		}
	}
	return out
}

// extractGeminiThoughtSignature pulls a Gemini-style thought_signature
// from a tool_use block's raw JSON, or returns "" if absent.
func extractGeminiThoughtSignature(raw string) string {
	if raw == "" {
		return ""
	}
	var probe struct {
		ThoughtSignature string `json:"thought_signature"`
	}
	if err := json.Unmarshal([]byte(raw), &probe); err != nil {
		return ""
	}
	return probe.ThoughtSignature
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
