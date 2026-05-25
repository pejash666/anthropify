package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/shahao/anthropify"
)

// portable defines a target provider plus the model name used against it.
type portable struct {
	label string
	model string
}

func main() {
	client, err := ap.New(
		ap.WithAnthropic(ap.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")}),
		ap.WithOpenAI(ap.OpenAIConfig{APIKey: os.Getenv("OPENAI_API_KEY")}),
		ap.WithGemini(ap.GeminiConfig{
			Mode:   ap.GeminiModeStudio,
			APIKey: os.Getenv("GEMINI_API_KEY"),
		}),
		ap.WithChatCompletion("kimi", ap.OpenAICompatConfig{
			BaseURL: envOr("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
			APIKey:  os.Getenv("KIMI_API_KEY"),
		}),
		ap.WithModelRoute("kimi-", ap.Route{Provider: ap.ProviderChatCompletions, Backend: "kimi"}),
	)
	if err != nil {
		panic(err)
	}

	// One Anthropic-shaped tool definition, reused unchanged for all four
	// providers. Each adapter translates this into the upstream's native
	// function-calling schema.
	tools := []anthropic.ToolUnionParam{
		{OfTool: &anthropic.ToolParam{
			Name:        "get_weather",
			Description: anthropic.String("Look up the current weather for a given city."),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"city": map[string]any{
						"type":        "string",
						"description": "City name, e.g. Beijing",
					},
				},
				Required: []string{"city"},
			},
		}},
	}

	providers := []portable{
		{label: "Anthropic", model: envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5")},
		{label: "OpenAI", model: envOr("OPENAI_MODEL", "gpt-5-mini")},
		{label: "Gemini", model: envOr("GEMINI_MODEL", "gemini-3.5-flash")},
		{label: "Kimi", model: envOr("KIMI_MODEL", "kimi-k2-thinking")},
	}

	for _, p := range providers {
		fmt.Printf("\n=== %s (%s) ===\n", p.label, p.model)
		runRoundTrip(client, p.model, tools)
	}
}

func runRoundTrip(client *ap.Client, model string, tools []anthropic.ToolUnionParam) {
	ctx := context.Background()

	// Round 1: ask the question, expect a tool_use block.
	round1 := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(
			"What is the weather in Beijing right now? You must call the get_weather tool.")),
	}
	msg, err := client.CreateMessage(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 1024,
		Tools:     tools,
		Messages:  round1,
	})
	if err != nil {
		fmt.Printf("round1 error: %v\n", err)
		return
	}

	var toolUseID, toolName string
	var toolInput map[string]any
	for _, b := range msg.Content {
		if b.Type == "tool_use" {
			toolUseID = b.ID
			toolName = b.Name
			_ = json.Unmarshal(b.Input, &toolInput)
			break
		}
	}
	if toolUseID == "" {
		fmt.Printf("model did not emit tool_use; stop=%s text=%q\n",
			msg.StopReason, joinText(msg))
		return
	}
	fmt.Printf("round1 tool_use: name=%s input=%v\n", toolName, toolInput)

	// Mock the tool: pretend Beijing is 21C and sunny.
	toolResult := `{"city":"Beijing","temperature_c":21,"condition":"sunny"}`

	// Round 2: feed the tool_result back so the model can compose a
	// natural-language reply.
	round2 := []anthropic.MessageParam{
		round1[0],
		// Build the assistant turn from the actual response message.
		{Role: anthropic.MessageParamRoleAssistant, Content: assistantBlocks(msg)},
		anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolUseID, toolResult, false)),
	}
	final, err := client.CreateMessage(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 1024,
		Tools:     tools,
		Messages:  round2,
	})
	if err != nil {
		fmt.Printf("round2 error: %v\n", err)
		return
	}
	fmt.Printf("round2 final: %s\n", trim(joinText(final), 240))
}

// assistantBlocks rebuilds the assistant turn as ContentBlockParamUnion so
// it can be appended back to the conversation. For Gemini 3.x tool_use
// blocks the upstream attaches a `thought_signature` that must round-trip
// on the next turn; we extract it from the response block's raw JSON and
// replay it via SetExtraFields.
func assistantBlocks(msg *anthropic.Message) []anthropic.ContentBlockParamUnion {
	out := make([]anthropic.ContentBlockParamUnion, 0, len(msg.Content))
	for _, b := range msg.Content {
		switch b.Type {
		case "text":
			out = append(out, anthropic.NewTextBlock(b.Text))
		case "tool_use":
			var input any
			_ = json.Unmarshal(b.Input, &input)
			block := anthropic.NewToolUseBlock(b.ID, input, b.Name)
			if sig := extractThoughtSignature(b.RawJSON()); sig != "" && block.OfToolUse != nil {
				block.OfToolUse.SetExtraFields(map[string]any{
					"thought_signature": sig,
				})
			}
			out = append(out, block)
		}
	}
	return out
}

// extractThoughtSignature pulls a Gemini-style thought_signature from a
// tool_use block's raw JSON, or returns "" if absent.
func extractThoughtSignature(raw string) string {
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

func joinText(msg *anthropic.Message) string {
	var sb strings.Builder
	for _, b := range msg.Content {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return strings.TrimSpace(sb.String())
}

func trim(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
