//go:build e2e

package helpers

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/pejash666/anthropify"
)

// StreamSummary distills a finished stream into the values every E2E
// test wants to assert on (text, stop reason, usage, block types).
type StreamSummary struct {
	EventTypes           []string
	Text                 string
	StopReason           string
	InputTokens          int64
	OutputTokens         int64
	CacheReadInputTokens int64
	ToolUseIDs           []string
	BlockTypes           []string
}

// DrainStream consumes a *anthropify.StreamReader and returns a
// summary plus the first transport error, if any. Tests call this with
// the StreamReader produced by client.CreateMessageStream.
func DrainStream(t *testing.T, stream *anthropify.StreamReader) StreamSummary {
	t.Helper()
	var s StreamSummary
	var text strings.Builder
	for stream.Next() {
		raw := stream.CurrentRaw()
		var peek struct {
			Type    string          `json:"type"`
			Index   int             `json:"index"`
			Message json.RawMessage `json:"message"`
			Content json.RawMessage `json:"content_block"`
			Delta   json.RawMessage `json:"delta"`
			Usage   json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(raw, &peek); err != nil {
			continue
		}
		s.EventTypes = append(s.EventTypes, peek.Type)
		switch peek.Type {
		case "message_start":
			var m struct {
				Message struct {
					Usage struct {
						InputTokens          int64 `json:"input_tokens"`
						OutputTokens         int64 `json:"output_tokens"`
						CacheReadInputTokens int64 `json:"cache_read_input_tokens"`
					} `json:"usage"`
				} `json:"message"`
			}
			if err := json.Unmarshal(raw, &m); err == nil {
				if m.Message.Usage.InputTokens > 0 {
					s.InputTokens = m.Message.Usage.InputTokens
				}
				if m.Message.Usage.OutputTokens > 0 {
					s.OutputTokens = m.Message.Usage.OutputTokens
				}
				if m.Message.Usage.CacheReadInputTokens > 0 {
					s.CacheReadInputTokens = m.Message.Usage.CacheReadInputTokens
				}
			}
		case "content_block_start":
			var cb struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			}
			_ = json.Unmarshal(peek.Content, &cb)
			s.BlockTypes = append(s.BlockTypes, cb.Type)
			if cb.Type == "tool_use" && cb.ID != "" {
				s.ToolUseIDs = append(s.ToolUseIDs, cb.ID)
			}
		case "content_block_delta":
			var d struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			_ = json.Unmarshal(peek.Delta, &d)
			if d.Type == "text_delta" {
				text.WriteString(d.Text)
			}
		case "message_delta":
			var d struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
				Usage struct {
					InputTokens          int64 `json:"input_tokens"`
					OutputTokens         int64 `json:"output_tokens"`
					CacheReadInputTokens int64 `json:"cache_read_input_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(raw, &d); err == nil {
				if d.Delta.StopReason != "" {
					s.StopReason = d.Delta.StopReason
				}
				if d.Usage.InputTokens > 0 {
					s.InputTokens = d.Usage.InputTokens
				}
				if d.Usage.OutputTokens > 0 {
					s.OutputTokens = d.Usage.OutputTokens
				}
				if d.Usage.CacheReadInputTokens > 0 {
					s.CacheReadInputTokens = d.Usage.CacheReadInputTokens
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	s.Text = text.String()
	return s
}

// AssertNonEmptyText fails the test when no text_delta content was
// observed.
func AssertNonEmptyText(t *testing.T, s StreamSummary) {
	t.Helper()
	if strings.TrimSpace(s.Text) == "" {
		t.Fatalf("expected non-empty text, got %q (events: %v)", s.Text, s.EventTypes)
	}
}

// AssertStopReason fails the test when the observed stop_reason doesn't
// match any of the accepted values. Multiple values are accepted because
// provider mapping has minor variations (e.g. tool_use vs end_turn).
func AssertStopReason(t *testing.T, s StreamSummary, accepted ...string) {
	t.Helper()
	for _, want := range accepted {
		if s.StopReason == want {
			return
		}
	}
	t.Fatalf("stop_reason = %q, want one of %v", s.StopReason, accepted)
}

// AssertInputTokensPositive fails when no input-side usage was reported.
// Counts either input_tokens or cache_read_input_tokens as evidence that
// the adapter forwarded usage: providers such as Moonshot may report
// the entire prompt as cached (input_tokens=0, cache_read>0), which is
// still a valid usage report under Anthropic semantics.
func AssertInputTokensPositive(t *testing.T, s StreamSummary) {
	t.Helper()
	if s.InputTokens <= 0 && s.CacheReadInputTokens <= 0 {
		t.Fatalf("usage.input_tokens = %d, cache_read_input_tokens = %d, want at least one > 0",
			s.InputTokens, s.CacheReadInputTokens)
	}
}

// AssertEventEnvelope checks that the stream emitted the canonical
// message_start ... message_stop bookends.
func AssertEventEnvelope(t *testing.T, s StreamSummary) {
	t.Helper()
	if len(s.EventTypes) == 0 {
		t.Fatalf("no events emitted")
	}
	if s.EventTypes[0] != "message_start" {
		t.Fatalf("first event = %q, want message_start", s.EventTypes[0])
	}
	last := s.EventTypes[len(s.EventTypes)-1]
	if last != "message_stop" {
		t.Fatalf("last event = %q, want message_stop", last)
	}
}

// AssertHasToolUse fails when no tool_use block was emitted. Returns
// the first tool_use_id for follow-up requests.
func AssertHasToolUse(t *testing.T, s StreamSummary) string {
	t.Helper()
	if len(s.ToolUseIDs) == 0 {
		t.Fatalf("expected a tool_use block; got block types %v", s.BlockTypes)
	}
	return s.ToolUseIDs[0]
}

// AssertHasThinkingBlock fails when no `thinking` content block was
// emitted. Used by providers (Anthropic, MiniMax, …) that round-trip
// extended-thinking through the canonical thinking block on the
// streaming response.
func AssertHasThinkingBlock(t *testing.T, s StreamSummary) {
	t.Helper()
	for _, bt := range s.BlockTypes {
		if bt == "thinking" {
			return
		}
	}
	t.Fatalf("expected a thinking content block; got block types %v", s.BlockTypes)
}

// AssertNonStreamingMessage validates the *anthropic.Message returned by
// client.CreateMessage. Checks that role==assistant, content is
// non-empty, and stop_reason is set.
func AssertNonStreamingMessage(t *testing.T, msg *anthropic.Message) {
	t.Helper()
	if msg == nil {
		t.Fatalf("nil message")
	}
	if msg.Role != "assistant" {
		t.Fatalf("role = %q, want assistant", msg.Role)
	}
	if len(msg.Content) == 0 {
		t.Fatalf("empty content blocks")
	}
	if msg.StopReason == "" {
		t.Fatalf("stop_reason is empty")
	}
}

// RecordingTransport is the placeholder hook for fixture recording. When
// E2E_RECORD_FIXTURES=true the harness should wrap the
// anthropify.WithHTTPClient transport to mirror raw SSE bytes into
// testdata/fixtures/recorded/<provider>/<test>.sse.
//
// TODO: implement. Today this is a no-op so the type compiles cleanly.
type RecordingTransport struct {
	Provider string
	TestName string
}

// WrapContext is reserved for future use; today it returns ctx
// unchanged. The signature is fixed so we can wire it through every
// test without churn later.
func (r *RecordingTransport) WrapContext(ctx context.Context) context.Context { return ctx }

// SkipIfUnsupported short-circuits a test with t.Skip when err signals
// that the feature is not yet implemented. This is used by the
// non-streaming E2E tests on adapters whose drain-and-assemble path is
// still scaffolded (openai_responses, gemini_native, chat_completions).
// Known limitation: the top-level client only natively non-streams via
// the anthropic passthrough adapter today; other providers return
// anthropify.ErrUnsupported (or an error message containing
// "feature not yet supported"). Once the fallback drain is wired up
// these skips can be removed.
func SkipIfUnsupported(t *testing.T, err error) bool {
	t.Helper()
	if err == nil {
		return false
	}
	if errors.Is(err, anthropify.ErrUnsupported) {
		t.Skipf("skipping: non-streaming not yet supported by adapter: %v", err)
		return true
	}
	msg := err.Error()
	if strings.Contains(msg, "feature not yet supported") ||
		strings.Contains(msg, "not yet supported") {
		t.Skipf("skipping: non-streaming not yet supported by adapter: %v", err)
		return true
	}
	return false
}
