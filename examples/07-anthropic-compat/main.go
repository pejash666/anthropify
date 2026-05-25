package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/shahao/anthropify"
)

// Protocol != vendor. The Anthropic Messages API is a *protocol*, and
// any provider speaking it can plug in as a named backend. Here we
// register Anthropic Inc. (real Claude) and MiniMax against the same
// in-process adapter, then dispatch by model prefix without changing a
// single line of caller code.
func main() {
	client, err := ap.New(
		// Default-named "anthropic" slot — sugar for
		// WithAnthropicCompat("anthropic", cfg). Picks up the built-in
		// claude-* prefix route.
		ap.WithAnthropic(ap.AnthropicConfig{
			APIKey: os.Getenv("ANTHROPIC_API_KEY"),
		}),
		// Second backend on the same anthropic protocol.
		ap.WithAnthropicCompat("minimax", ap.AnthropicConfig{
			BaseURL: envOr("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"),
			APIKey:  os.Getenv("MINIMAX_API_KEY"),
		}),
		// Pin the MiniMax- model prefix to the "minimax" backend so the
		// dispatcher does not confuse it with real Claude.
		ap.WithModelRoute("MiniMax-", ap.Route{
			Provider: ap.ProviderAnthropic, Backend: "minimax",
		}),
	)
	if err != nil {
		panic(err)
	}

	question := "In one sentence: which company trained you?"
	for _, model := range []string{
		envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
		envOr("MINIMAX_MODEL", "MiniMax-M2.7"),
	} {
		fmt.Printf("\n--- %s ---\n", model)
		ask(client, model, question)
	}
}

func ask(client *ap.Client, model, question string) {
	stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
		Model:     anthropic.Model(model),
		MaxTokens: 256,
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
	fmt.Println(strings.TrimSpace(out.String()))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
