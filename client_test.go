package anthropify

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// TestClient_RoutesToOpenAIResponses wires a fake /v1/responses server
// and drives a full request -> stream round-trip through the public
// Client. It replays an SSE fixture and asserts the emitted Anthropic
// events match the expected sequence.
func TestClient_RoutesToOpenAIResponses(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join("adapter", "openai_responses", "testdata", "text_stream.sse"))
	if err != nil {
		t.Fatalf("fixture read: %v", err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			t.Errorf("auth header missing: %v", r.Header)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.Copy(w, strings.NewReader(string(fixture)))
	}))
	defer srv.Close()

	client, err := New(
		WithOpenAI(OpenAIConfig{APIKey: "sk-test", BaseURL: srv.URL}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	src := []byte(`{"model":"gpt-5","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}

	stream, err := client.CreateMessageStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	var types []string
	var text strings.Builder
	for stream.Next() {
		e := stream.Current()
		types = append(types, e.Type)
		if e.Type == "content_block_delta" {
			var d struct {
				Delta struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"delta"`
			}
			_ = json.Unmarshal(stream.CurrentRaw(), &d)
			if d.Delta.Type == "text_delta" {
				text.WriteString(d.Delta.Text)
			}
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	if text.String() != "ABC" {
		t.Fatalf("text = %q", text.String())
	}
	want := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Fatalf("types = %v\nwant %v", types, want)
	}
}

// TestClient_ProviderNotConfigured exercises the "registered adapters do
// not cover the routed provider" error path.
func TestClient_ProviderNotConfigured(t *testing.T) {
	client, err := New() // no providers
	if err != nil {
		t.Fatal(err)
	}
	src := []byte(`{"model":"claude-3-5-sonnet","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"."}]}]}`)
	var req anthropicsdk.MessageNewParams
	_ = json.Unmarshal(src, &req)
	_, err = client.CreateMessageStream(context.Background(), req)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "provider not configured") {
		t.Fatalf("err = %v", err)
	}
}
