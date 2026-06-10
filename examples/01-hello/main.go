package main

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/pejash666/anthropify"
)

func main() {
	client, err := ap.New(
		ap.WithAnthropic(ap.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")}),
	)
	if err != nil {
		panic(err)
	}

	stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
		Model:     anthropic.Model(envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5")),
		MaxTokens: 256,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("In one sentence, what is Go's main strength?")),
		},
	})
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	for stream.Next() {
		evt := stream.Current()
		if delta := evt.Delta.Text; delta != "" {
			fmt.Print(delta)
		}
	}
	fmt.Println()
	if err := stream.Err(); err != nil {
		panic(err)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
