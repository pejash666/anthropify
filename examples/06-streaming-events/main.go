package main

import (
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	ap "github.com/shahao/anthropify"
)

// Walks one streaming response and prints a one-line summary of every
// MessageStreamEvent emitted, in order. The point is to make the
// Anthropic-canonical event vocabulary concrete: which event types
// exist, what fields they carry, and in what sequence.
func main() {
	client, err := ap.New(
		ap.WithAnthropic(ap.AnthropicConfig{APIKey: os.Getenv("ANTHROPIC_API_KEY")}),
	)
	if err != nil {
		panic(err)
	}

	stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
		Model:     anthropic.Model(envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5")),
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(
				"Reply with exactly: 'streaming events demo'.")),
		},
	})
	if err != nil {
		panic(err)
	}
	defer stream.Close()

	var i int
	for stream.Next() {
		i++
		evt := stream.Current()
		switch evt.Type {
		case "message_start":
			fmt.Printf("%2d  message_start         id=%s model=%s role=%s\n",
				i, evt.Message.ID, evt.Message.Model, evt.Message.Role)
		case "content_block_start":
			fmt.Printf("%2d  content_block_start   index=%d type=%s\n",
				i, evt.Index, evt.ContentBlock.Type)
		case "content_block_delta":
			fmt.Printf("%2d  content_block_delta   index=%d delta.type=%s text=%q\n",
				i, evt.Index, evt.Delta.Type, evt.Delta.Text)
		case "content_block_stop":
			fmt.Printf("%2d  content_block_stop    index=%d\n", i, evt.Index)
		case "message_delta":
			fmt.Printf("%2d  message_delta         stop_reason=%s output_tokens=%d\n",
				i, evt.Delta.StopReason, evt.Usage.OutputTokens)
		case "message_stop":
			fmt.Printf("%2d  message_stop\n", i)
		default:
			fmt.Printf("%2d  %-20s (raw=%s)\n", i, evt.Type, string(stream.CurrentRaw()))
		}
	}
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
