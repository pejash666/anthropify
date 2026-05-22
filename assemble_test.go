package anthropify

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/anthropify/adapter"
	chatadapter "github.com/shahao/anthropify/adapter/chat_completions"
	geminiadapter "github.com/shahao/anthropify/adapter/gemini_native"
	openairesponses "github.com/shahao/anthropify/adapter/openai_responses"
)

// mockAdapter is a hand-rolled adapter.Adapter that returns
// ErrUnsupported from Invoke and replays a canned sequence of
// Anthropic-shaped events from Stream. It is the smallest harness
// needed to drive client.CreateMessage through the drain-and-assemble
// fallback path without any real network I/O.
type mockAdapter struct {
	events []string
	// streamErr, if non-nil, is emitted as the final RawEvent.Err.
	streamErr error
	// invokeErr overrides the default Invoke return (ErrUnsupported).
	invokeErr error
	// invokeMsg, when set, makes Invoke succeed instead of falling back.
	invokeMsg *anthropicsdk.Message
}

func (m *mockAdapter) Name() string { return "mock" }

func (m *mockAdapter) Invoke(_ context.Context, _ anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	if m.invokeMsg != nil {
		return m.invokeMsg, nil
	}
	if m.invokeErr != nil {
		return nil, m.invokeErr
	}
	return nil, adapter.ErrUnsupported
}

func (m *mockAdapter) Stream(ctx context.Context, _ anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	ch := make(chan adapter.RawEvent, len(m.events)+1)
	go func() {
		defer close(ch)
		for _, e := range m.events {
			select {
			case ch <- adapter.RawEvent{Data: []byte(e)}:
			case <-ctx.Done():
				ch <- adapter.RawEvent{Err: ctx.Err()}
				return
			}
		}
		if m.streamErr != nil {
			ch <- adapter.RawEvent{Err: m.streamErr}
		}
	}()
	return ch, nil
}

// newClientWithMock installs mock under the chat_completions slot named
// "mock" and routes every model to it. The constructor in New() does not
// expose a hook for tests, so we build a Client by hand. This is
// permissible because we live in the same package.
func newClientWithMock(t *testing.T, mock adapter.Adapter) *Client {
	t.Helper()
	cfg := defaultConfig()
	cfg.overrideFn = func(model string) (Route, bool) {
		return Route{Provider: ProviderChatCompletions, Backend: "mock"}, true
	}
	return &Client{
		cfg:   cfg,
		chats: map[string]adapter.Adapter{"mock": mock},
	}
}

// TestAssemble_TextOnly verifies the drain path produces a Message with
// a single text block and the stop_reason from message_delta.
func TestAssemble_TextOnly(t *testing.T) {
	mock := &mockAdapter{events: []string{
		`{"type":"message_start","message":{"id":"msg_1","role":"assistant","model":"mock-1","content":[],"stop_reason":null,"usage":{"input_tokens":3,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello, "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":""},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	}}
	cli := newClientWithMock(t, mock)

	src := []byte(`{"model":"any","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}

	msg, err := cli.CreateMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if msg.Role != "assistant" {
		t.Fatalf("role = %q", msg.Role)
	}
	if msg.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %q", msg.StopReason)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(msg.Content))
	}
	tb := msg.Content[0].AsAny()
	tx, ok := tb.(anthropicsdk.TextBlock)
	if !ok {
		t.Fatalf("block 0 = %T, want TextBlock", tb)
	}
	if tx.Text != "Hello, world!" {
		t.Fatalf("text = %q", tx.Text)
	}
	if msg.Usage.InputTokens != 3 {
		t.Fatalf("input_tokens = %d, want 3", msg.Usage.InputTokens)
	}
	if msg.Usage.OutputTokens != 5 {
		t.Fatalf("output_tokens = %d, want 5", msg.Usage.OutputTokens)
	}
}

// TestAssemble_ToolUse verifies that tool_use blocks are emitted with
// the id/name from content_block_start and the input rebuilt by
// concatenating input_json_delta partial_json frames.
func TestAssemble_ToolUse(t *testing.T) {
	mock := &mockAdapter{events: []string{
		`{"type":"message_start","message":{"id":"msg_2","role":"assistant","model":"mock-1","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_42","name":"get_weather","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Tokyo\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":""},"usage":{"output_tokens":12}}`,
		`{"type":"message_stop"}`,
	}}
	cli := newClientWithMock(t, mock)
	var req anthropicsdk.MessageNewParams
	_ = json.Unmarshal([]byte(`{"model":"any","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"weather?"}]}]}`), &req)

	msg, err := cli.CreateMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if msg.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %q, want tool_use", msg.StopReason)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(msg.Content))
	}
	tu, ok := msg.Content[0].AsAny().(anthropicsdk.ToolUseBlock)
	if !ok {
		t.Fatalf("block 0 = %T, want ToolUseBlock", msg.Content[0].AsAny())
	}
	if tu.ID != "call_42" {
		t.Fatalf("tool id = %q", tu.ID)
	}
	if tu.Name != "get_weather" {
		t.Fatalf("tool name = %q", tu.Name)
	}
	// Input is exposed as json.RawMessage; decode to verify the merge.
	var input map[string]any
	if err := json.Unmarshal(tu.Input, &input); err != nil {
		t.Fatalf("tool input not valid JSON: %v (%s)", err, string(tu.Input))
	}
	if input["city"] != "Tokyo" {
		t.Fatalf("tool input = %v, want city=Tokyo", input)
	}
	if msg.Usage.OutputTokens != 12 {
		t.Fatalf("output_tokens = %d", msg.Usage.OutputTokens)
	}
}

// TestAssemble_Thinking exercises thinking_delta accumulation alongside
// a trailing text block at index 1.
func TestAssemble_Thinking(t *testing.T) {
	mock := &mockAdapter{events: []string{
		`{"type":"message_start","message":{"id":"msg_3","role":"assistant","model":"mock-1","content":[]}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Step 1. "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Step 2."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Done."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`,
		`{"type":"message_stop"}`,
	}}
	cli := newClientWithMock(t, mock)
	var req anthropicsdk.MessageNewParams
	_ = json.Unmarshal([]byte(`{"model":"any","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"think"}]}]}`), &req)
	msg, err := cli.CreateMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("content blocks = %d, want 2", len(msg.Content))
	}
	tb, ok := msg.Content[0].AsAny().(anthropicsdk.ThinkingBlock)
	if !ok {
		t.Fatalf("block 0 = %T, want ThinkingBlock", msg.Content[0].AsAny())
	}
	if tb.Thinking != "Step 1. Step 2." {
		t.Fatalf("thinking = %q", tb.Thinking)
	}
	tx, ok := msg.Content[1].AsAny().(anthropicsdk.TextBlock)
	if !ok {
		t.Fatalf("block 1 = %T, want TextBlock", msg.Content[1].AsAny())
	}
	if tx.Text != "Done." {
		t.Fatalf("text = %q", tx.Text)
	}
}

// TestAssemble_UsageMerge confirms that usage from message_start is
// preserved when message_delta only re-declares output_tokens.
func TestAssemble_UsageMerge(t *testing.T) {
	mock := &mockAdapter{events: []string{
		`{"type":"message_start","message":{"id":"msg_4","role":"assistant","model":"mock-1","content":[],"usage":{"input_tokens":42,"output_tokens":1,"cache_read_input_tokens":8}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":11}}`,
		`{"type":"message_stop"}`,
	}}
	cli := newClientWithMock(t, mock)
	var req anthropicsdk.MessageNewParams
	_ = json.Unmarshal([]byte(`{"model":"any","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"ok?"}]}]}`), &req)
	msg, err := cli.CreateMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if msg.Usage.InputTokens != 42 {
		t.Fatalf("input_tokens = %d, want 42", msg.Usage.InputTokens)
	}
	if msg.Usage.OutputTokens != 11 {
		t.Fatalf("output_tokens = %d, want 11", msg.Usage.OutputTokens)
	}
	if msg.Usage.CacheReadInputTokens != 8 {
		t.Fatalf("cache_read_input_tokens = %d, want 8", msg.Usage.CacheReadInputTokens)
	}
}

// TestCreateMessage_InvokeError forwards a non-ErrUnsupported Invoke
// error directly without attempting the drain fallback.
func TestCreateMessage_InvokeError(t *testing.T) {
	want := errors.New("upstream 500")
	mock := &mockAdapter{
		invokeErr: want,
		events:    []string{`{"type":"message_start","message":{"role":"assistant"}}`},
	}
	cli := newClientWithMock(t, mock)
	var req anthropicsdk.MessageNewParams
	_ = json.Unmarshal([]byte(`{"model":"any","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`), &req)
	_, err := cli.CreateMessage(context.Background(), req)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "upstream 500") {
		t.Fatalf("err = %v", err)
	}
}

// TestCreateMessage_InvokeNative confirms that when Invoke succeeds
// natively the drain path is skipped.
func TestCreateMessage_InvokeNative(t *testing.T) {
	native := &anthropicsdk.Message{Role: "assistant", StopReason: "end_turn"}
	mock := &mockAdapter{invokeMsg: native}
	cli := newClientWithMock(t, mock)
	var req anthropicsdk.MessageNewParams
	_ = json.Unmarshal([]byte(`{"model":"any","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`), &req)
	msg, err := cli.CreateMessage(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessage: %v", err)
	}
	if msg != native {
		t.Fatalf("expected native message identity")
	}
}

// TestErrUnsupported_Aliased ensures every adapter package exports a
// sentinel that errors.Is matches against the top-level one. Without
// this, the Client's fallback would silently misfire.
func TestErrUnsupported_Aliased(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"adapter.ErrUnsupported", adapter.ErrUnsupported},
		{"openai_responses.ErrUnsupported", openairesponses.ErrUnsupported},
		{"chat_completions.ErrUnsupported", chatadapter.ErrUnsupported},
		{"gemini_native.ErrUnsupported", geminiadapter.ErrUnsupported},
	}
	for _, tc := range cases {
		if !errors.Is(tc.err, ErrUnsupported) {
			t.Errorf("%s: errors.Is(_, anthropify.ErrUnsupported) = false", tc.name)
		}
	}
}
