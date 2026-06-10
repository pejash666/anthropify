package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/pejash666/anthropify"
)

// Demonstrates that one anthropify.Client can dispatch to five
// different providers — across two protocols — based purely on the
// req.Model prefix. Anthropic and MiniMax share the *anthropic*
// protocol but ride two distinct backends; the application code is
// identical across providers.
func main() {
	client, err := ap.New(
		ap.WithAnthropic(ap.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")}),
		ap.WithAnthropicCompat("minimax", ap.AnthropicConfig{
			BaseURL: envOr("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
			APIKey:  os.Getenv("MINIMAX_API_KEY"),
		}),
		ap.WithGemini(ap.GeminiConfig{
			Mode:   ap.GeminiModeStudio,
			APIKey: os.Getenv("GEMINI_API_KEY"),
		}),
		ap.WithChatCompletion("kimi", ap.OpenAICompatConfig{
			BaseURL: envOr("KIMI_BASE_URL", "https://api.moonshot.ai/v1"),
			APIKey:  os.Getenv("KIMI_API_KEY"),
		}),
		ap.WithChatCompletion("glm", ap.OpenAICompatConfig{
			BaseURL: envOr("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
			APIKey:  os.Getenv("GLM_API_KEY"),
		}),
		// claude-/gpt-/gemini- are routed by default. Kimi, GLM, and
		// MiniMax need explicit prefix rules tied to their backend
		// names. MiniMax pins the "MiniMax-" prefix to the second
		// anthropic-protocol backend so the dispatcher does not confuse
		// it with real Claude.
		ap.WithModelRoute("kimi-", ap.Route{
			Provider: ap.ProviderChatCompletions, Backend: "kimi",
		}),
		ap.WithModelRoute("glm-", ap.Route{
			Provider: ap.ProviderChatCompletions, Backend: "glm",
		}),
		ap.WithModelRoute("MiniMax-", ap.Route{
			Provider: ap.ProviderAnthropic, Backend: "minimax",
		}),
	)
	if err != nil {
		panic(err)
	}

	question := "In one short sentence: what year was your model trained?"
	for _, model := range []string{
		envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
		envOr("GEMINI_MODEL", "gemini-3.5-flash"),
		envOr("KIMI_MODEL", "kimi-k2.6"),
		envOr("GLM_MODEL", "glm-5.1"),
		envOr("MINIMAX_MODEL", "MiniMax-M2.7"),
	} {
		fmt.Printf("\n--- %s ---\n", model)
		askOnce(client, model, question)
	}
}

func askOnce(client *ap.Client, model, question string) {
	stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 1024,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(question)),
		},
	})
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	defer stream.Close()

	var out strings.Builder
	for stream.Next() {
		evt := stream.Current()
		if evt.Type == "content_block_delta" && evt.Delta.Type == "text_delta" {
			out.WriteString(evt.Delta.Text)
		}
	}
	if err := stream.Err(); err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}
	text := strings.TrimSpace(out.String())
	if len(text) > 200 {
		text = text[:200] + "..."
	}
	fmt.Println(text)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
