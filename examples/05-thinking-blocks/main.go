package main

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/shahao/anthropify"
)

// thinker pairs a label with a model that supports thinking blocks.
type thinker struct {
	label string
	model string
}

func main() {
	client, err := ap.New(
		ap.WithAnthropic(ap.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")}),
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

	question := "I have two children. One is a boy born on a Tuesday. " +
		"What is the probability the other is also a boy? Show your reasoning."

	for _, t := range []thinker{
		{label: "Anthropic", model: envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5")},
		{label: "Kimi", model: envOr("KIMI_MODEL", "kimi-k2.6")},
		{label: "Gemini", model: envOr("GEMINI_MODEL", "gemini-3.5-flash")},
	} {
		fmt.Printf("\n=== %s (%s) ===\n", t.label, t.model)
		runOne(client, t.model, question)
	}
}

func runOne(client *ap.Client, model, question string) {
	req := anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 16384,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(question)),
		},
	}
	// Enable extended thinking on Anthropic and Gemini via the canonical
	// `Thinking` field. The adapter for Gemini 3.x maps the budget to a
	// `thinkingLevel` bucket (low/medium/high) on generationConfig. Kimi's
	// k2-thinking model emits thinking blocks server-side without an
	// explicit gate.
	if isAnthropic(model) || isGemini(model) {
		req.Thinking = anthropic.ThinkingConfigParamOfEnabled(12000)
	}

	stream, err := client.CreateMessageStream(context.Background(), req)
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	defer stream.Close()

	// Track which content block index is which type so we know how to
	// label deltas as they arrive.
	blockType := map[int64]string{}
	var thinkingChars, textChars int

	for stream.Next() {
		evt := stream.Current()
		switch evt.Type {
		case "content_block_start":
			blockType[evt.Index] = evt.ContentBlock.Type
			fmt.Printf("[block %d start: %s]\n", evt.Index, evt.ContentBlock.Type)
		case "content_block_delta":
			t := blockType[evt.Index]
			switch evt.Delta.Type {
			case "thinking_delta":
				thinkingChars += len(evt.Delta.Thinking)
			case "text_delta":
				textChars += len(evt.Delta.Text)
			}
			_ = t
		case "content_block_stop":
			fmt.Printf("[block %d stop: %s]\n", evt.Index, blockType[evt.Index])
		}
	}
	if err := stream.Err(); err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	fmt.Printf("totals: thinking=%d chars, text=%d chars\n", thinkingChars, textChars)
}

func isAnthropic(model string) bool {
	return len(model) >= 7 && model[:7] == "claude-"
}

func isGemini(model string) bool {
	return len(model) >= 7 && model[:7] == "gemini-"
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
