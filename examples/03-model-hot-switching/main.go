package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/pejash666/anthropify"
)

// turn binds a user question to the model that should answer it. The
// conversation slice is shared across turns; only req.Model and the
// per-turn token budget change.
type turn struct {
	model     string
	question  string
	label     string // human-readable provider name for the >>> banner
	maxTokens int64
}

func main() {
	client, err := ap.New(
		ap.WithAnthropic(ap.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")}),
		ap.WithChatCompletion("kimi", ap.OpenAICompatConfig{
			BaseURL: envOr("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
			APIKey:  os.Getenv("KIMI_API_KEY"),
		}),
		ap.WithChatCompletion("glm", ap.OpenAICompatConfig{
			BaseURL: envOr("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
			APIKey:  os.Getenv("GLM_API_KEY"),
		}),
		ap.WithModelRoute("kimi-", ap.Route{Provider: ap.ProviderChatCompletions, Backend: "kimi"}),
		ap.WithModelRoute("glm-", ap.Route{Provider: ap.ProviderChatCompletions, Backend: "glm"}),
	)
	if err != nil {
		panic(err)
	}

	turns := []turn{
		{
			label:     "Anthropic (planning)",
			model:     envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
			question:  "Plan a 3-day Tokyo trip in 5 short bullets. No prose.",
			maxTokens: 1024,
		},
		{
			label:     "Kimi (code)",
			model:     envOr("KIMI_MODEL", "kimi-k2.6"),
			question:  "Now write a tiny Python function that returns those 5 bullets as a list of strings. Code only, no comments.",
			maxTokens: 8192, // kimi-k2.6 spends most of its budget on thinking
		},
		{
			label:     "GLM (translation)",
			model:     envOr("GLM_MODEL", "glm-5.1"),
			question:  "把上面这 5 条要点翻译成日语,保持要点编号。",
			maxTokens: 1024,
		},
		{
			label:     "Anthropic (summary)",
			model:     envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
			question:  "In 3 bullets, summarise the entire conversation so far in English.",
			maxTokens: 512,
		},
	}

	// Single shared conversation history. Anthropic-canonical message
	// shape is identical across providers so we just keep appending.
	var messages []anthropic.MessageParam

	for i, t := range turns {
		fmt.Printf("\n>>> Turn %d -> Switching to %s (%s)\n", i+1, t.label, t.model)
		fmt.Printf("user: %s\n", t.question)

		messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(t.question)))

		assistant, err := runTurn(client, t.model, t.maxTokens, messages)
		if err != nil {
			fmt.Printf("error on turn %d (%s): %v\n", i+1, t.label, err)
			return
		}
		// Some thinking-heavy models (e.g. kimi-k2.6) can exhaust
		// their token budget on thinking blocks before emitting any
		// text. Keep the conversation valid by substituting a sentinel
		// rather than appending an empty assistant turn (which the
		// Anthropic API rejects on the next round).
		display := assistant
		if assistant == "" {
			assistant = "(no text emitted; thinking budget exhausted)"
			display = assistant
		}
		fmt.Printf("assistant: %s\n", trim(display, 400))

		// Append the assistant turn so the next provider sees it as
		// part of the canonical context.
		messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(assistant)))
	}
}

func runTurn(client *ap.Client, model string, maxTokens int64, messages []anthropic.MessageParam) (string, error) {
	stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: maxTokens,
		Messages:  messages,
	})
	if err != nil {
		return "", err
	}
	defer stream.Close()

	var out strings.Builder
	for stream.Next() {
		evt := stream.Current()
		if evt.Type == "content_block_delta" && evt.Delta.Type == "text_delta" {
			out.WriteString(evt.Delta.Text)
		}
	}
	return strings.TrimSpace(out.String()), stream.Err()
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
