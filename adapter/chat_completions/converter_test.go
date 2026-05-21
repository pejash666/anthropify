package chat_completions

import (
	"encoding/json"
	"strings"
	"testing"
)

// runConverter feeds each non-blank line in input to a fresh Converter,
// then calls OnStreamDone, and returns the decoded Anthropic events.
func runConverter(t *testing.T, input string) []map[string]any {
	t.Helper()
	c := NewConverter("chatcmpl-test", "msg_test")
	var out []map[string]any
	for _, line := range strings.Split(input, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		for _, emitted := range c.ConvertEvent(line) {
			var m map[string]any
			if err := json.Unmarshal([]byte(emitted), &m); err != nil {
				t.Fatalf("emitted non-JSON event %q: %v", emitted, err)
			}
			out = append(out, m)
		}
	}
	for _, emitted := range c.OnStreamDone() {
		var m map[string]any
		if err := json.Unmarshal([]byte(emitted), &m); err != nil {
			t.Fatalf("emitted non-JSON event %q: %v", emitted, err)
		}
		out = append(out, m)
	}
	return out
}

func eventTypes(evts []map[string]any) []string {
	out := make([]string, 0, len(evts))
	for _, e := range evts {
		if s, _ := e["type"].(string); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// TestConverter_TextOnly: text content -> end_turn.
func TestConverter_TextOnly(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"Hello"}}]}
{"choices":[{"delta":{"content":" world"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":2}}
`
	events := runConverter(t, input)
	got := eventTypes(events)
	want := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence mismatch:\n got %v\nwant %v", got, want)
	}
	var accum string
	for _, e := range events {
		if e["type"] != "content_block_delta" {
			continue
		}
		d := e["delta"].(map[string]any)
		if d["type"] == "text_delta" {
			accum += d["text"].(string)
		}
	}
	if accum != "Hello world" {
		t.Fatalf("text accum = %q, want %q", accum, "Hello world")
	}
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "end_turn" {
				t.Fatalf("stop_reason = %v, want end_turn", d["stop_reason"])
			}
		}
	}
}

// TestConverter_ReasoningThenText: reasoning_content delta followed by
// content delta. Thinking block must close before text block opens.
func TestConverter_ReasoningThenText(t *testing.T) {
	input := `
{"choices":[{"delta":{"reasoning_content":"Think...","content":""}}]}
{"choices":[{"delta":{"reasoning_content":" more","content":""}}]}
{"choices":[{"delta":{"content":"Answer"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}]}
`
	events := runConverter(t, input)
	got := eventTypes(events)
	want := []string{
		"message_start",
		"content_block_start", // thinking
		"content_block_delta", // thinking_delta
		"content_block_delta", // thinking_delta
		"content_block_stop",  // close thinking
		"content_block_start", // text
		"content_block_delta", // text_delta
		"content_block_stop",  // close text
		"message_delta",
		"message_stop",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence mismatch:\n got %v\nwant %v", got, want)
	}
	// First content_block_start must be thinking.
	for _, e := range events {
		if e["type"] == "content_block_start" {
			block := e["content_block"].(map[string]any)
			if block["type"] != "thinking" {
				t.Fatalf("first content_block_start type = %v, want thinking", block["type"])
			}
			break
		}
	}
	// Indexes: thinking=0, text=1.
	var thinkingIdx, textIdx float64 = -1, -1
	for _, e := range events {
		if e["type"] == "content_block_start" {
			block := e["content_block"].(map[string]any)
			switch block["type"] {
			case "thinking":
				thinkingIdx = e["index"].(float64)
			case "text":
				textIdx = e["index"].(float64)
			}
		}
	}
	if thinkingIdx != 0 || textIdx != 1 {
		t.Fatalf("indices wrong: thinking=%v text=%v", thinkingIdx, textIdx)
	}
}

// TestConverter_ToolUse: single tool_call.
func TestConverter_ToolUse(t *testing.T) {
	input := `
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"read_file","arguments":"{\"path\":"}}]}}]}
{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"/etc/hosts\"}"}}]}}]}
{"choices":[{"finish_reason":"tool_calls","delta":{}}]}
`
	events := runConverter(t, input)
	var toolBlock map[string]any
	for _, e := range events {
		if e["type"] == "content_block_start" {
			b := e["content_block"].(map[string]any)
			if b["type"] == "tool_use" {
				toolBlock = b
				break
			}
		}
	}
	if toolBlock == nil {
		t.Fatalf("no tool_use content_block_start emitted")
	}
	if toolBlock["name"] != "read_file" {
		t.Fatalf("tool name = %v, want read_file", toolBlock["name"])
	}
	if toolBlock["id"] != "call_abc" {
		t.Fatalf("tool id = %v, want call_abc", toolBlock["id"])
	}
	// Reassemble partial_json.
	var argsAccum string
	for _, e := range events {
		if e["type"] != "content_block_delta" {
			continue
		}
		d := e["delta"].(map[string]any)
		if d["type"] == "input_json_delta" {
			argsAccum += d["partial_json"].(string)
		}
	}
	if argsAccum != `{"path":"/etc/hosts"}` {
		t.Fatalf("args accum = %q", argsAccum)
	}
	// stop_reason should be tool_use (override since no trailing text).
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "tool_use" {
				t.Fatalf("stop_reason = %v, want tool_use", d["stop_reason"])
			}
		}
	}
}

// TestConverter_ParallelToolUse: multiple tool_calls in parallel.
func TestConverter_ParallelToolUse(t *testing.T) {
	input := `
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"fn_a","arguments":"{\"x\":1}"}}]}}]}
{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"fn_b","arguments":"{\"y\":2}"}}]}}]}
{"choices":[{"finish_reason":"tool_calls","delta":{}}]}
`
	events := runConverter(t, input)
	var toolBlocks []map[string]any
	for _, e := range events {
		if e["type"] == "content_block_start" {
			b := e["content_block"].(map[string]any)
			if b["type"] == "tool_use" {
				toolBlocks = append(toolBlocks, b)
			}
		}
	}
	if len(toolBlocks) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got %d", len(toolBlocks))
	}
	if toolBlocks[0]["name"] != "fn_a" || toolBlocks[1]["name"] != "fn_b" {
		t.Fatalf("tool names: %v %v", toolBlocks[0]["name"], toolBlocks[1]["name"])
	}
	if toolBlocks[0]["id"] != "call_1" || toolBlocks[1]["id"] != "call_2" {
		t.Fatalf("tool ids: %v %v", toolBlocks[0]["id"], toolBlocks[1]["id"])
	}
	// Each tool block opens at consecutive indexes 0 and 1.
	var idxs []float64
	for _, e := range events {
		if e["type"] == "content_block_start" {
			idxs = append(idxs, e["index"].(float64))
		}
	}
	if len(idxs) != 2 || idxs[0] != 0 || idxs[1] != 1 {
		t.Fatalf("tool block indices = %v, want [0 1]", idxs)
	}
}

// TestConverter_ToolUseFollowedByText: a tool_use followed by text
// should restore end_turn (text proves the model went past the tool).
func TestConverter_ToolUseFollowedByText(t *testing.T) {
	input := `
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"ping","arguments":"{}"}}]}}]}
{"choices":[{"delta":{"content":"done"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}]}
`
	events := runConverter(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "end_turn" {
				t.Fatalf("stop_reason = %v, want end_turn", d["stop_reason"])
			}
			return
		}
	}
	t.Fatalf("no message_delta")
}

// TestConverter_FinishReasonStop -> end_turn.
func TestConverter_FinishReasonStop(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"hi"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}]}
`
	events := runConverter(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "end_turn" {
				t.Fatalf("stop_reason = %v, want end_turn", d["stop_reason"])
			}
		}
	}
}

// TestConverter_FinishReasonLength -> max_tokens.
func TestConverter_FinishReasonLength(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"hi"}}]}
{"choices":[{"finish_reason":"length","delta":{}}]}
`
	events := runConverter(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "max_tokens" {
				t.Fatalf("stop_reason = %v, want max_tokens", d["stop_reason"])
			}
		}
	}
}

// TestConverter_FinishReasonToolCalls -> tool_use (with no trailing text).
func TestConverter_FinishReasonToolCalls(t *testing.T) {
	input := `
{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{}"}}]}}]}
{"choices":[{"finish_reason":"tool_calls","delta":{}}]}
`
	events := runConverter(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "tool_use" {
				t.Fatalf("stop_reason = %v, want tool_use", d["stop_reason"])
			}
		}
	}
}

// TestConverter_FinishReasonContentFilter -> refusal.
func TestConverter_FinishReasonContentFilter(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"x"}}]}
{"choices":[{"finish_reason":"content_filter","delta":{}}]}
`
	events := runConverter(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "refusal" {
				t.Fatalf("stop_reason = %v, want refusal", d["stop_reason"])
			}
		}
	}
}

// TestConverter_UsageMapping ensures usage on message_stop maps
// prompt_tokens / completion_tokens / cached_tokens correctly.
func TestConverter_UsageMapping(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"x"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":100,"completion_tokens":42,"prompt_tokens_details":{"cached_tokens":30}}}
`
	events := runConverter(t, input)
	var usage map[string]any
	for _, e := range events {
		if e["type"] == "message_stop" {
			if u, ok := e["usage"].(map[string]any); ok {
				usage = u
				break
			}
		}
	}
	if usage == nil {
		t.Fatalf("no usage block in message_stop")
	}
	if usage["input_tokens"].(float64) != 70 { // 100 - 30 cached
		t.Fatalf("input_tokens = %v, want 70", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 42 {
		t.Fatalf("output_tokens = %v, want 42", usage["output_tokens"])
	}
	if usage["cache_read_input_tokens"].(float64) != 30 {
		t.Fatalf("cache_read_input_tokens = %v, want 30", usage["cache_read_input_tokens"])
	}
	if usage["cache_creation_input_tokens"].(float64) != 0 {
		t.Fatalf("cache_creation_input_tokens = %v, want 0", usage["cache_creation_input_tokens"])
	}
}

// TestConverter_UsageMappingFlatCached: some providers put
// cached_tokens at the root of usage rather than under prompt_tokens_details.
func TestConverter_UsageMappingFlatCached(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"x"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":50,"completion_tokens":7,"cached_tokens":10}}
`
	events := runConverter(t, input)
	var usage map[string]any
	for _, e := range events {
		if e["type"] == "message_stop" {
			if u, ok := e["usage"].(map[string]any); ok {
				usage = u
				break
			}
		}
	}
	if usage == nil {
		t.Fatalf("no usage block in message_stop")
	}
	if usage["input_tokens"].(float64) != 40 { // 50 - 10 cached
		t.Fatalf("input_tokens = %v, want 40", usage["input_tokens"])
	}
	if usage["cache_read_input_tokens"].(float64) != 10 {
		t.Fatalf("cache_read_input_tokens = %v, want 10", usage["cache_read_input_tokens"])
	}
}

// TestConverter_MalformedEvent: bad JSON is silently skipped.
func TestConverter_MalformedEvent(t *testing.T) {
	c := NewConverter("chatcmpl-test", "msg_test")
	got := c.ConvertEvent("{not json")
	if len(got) != 0 {
		t.Fatalf("malformed event produced %d events: %v", len(got), got)
	}
}

// TestConverter_OnStreamDoneIdempotent ensures double-flush is safe.
func TestConverter_OnStreamDoneIdempotent(t *testing.T) {
	c := NewConverter("chatcmpl-test", "msg_test")
	c.ConvertEvent(`{"choices":[{"delta":{"content":"hi"}}]}`)
	first := c.OnStreamDone()
	second := c.OnStreamDone()
	if len(first) == 0 {
		t.Fatalf("first OnStreamDone returned no events")
	}
	if len(second) != 0 {
		t.Fatalf("second OnStreamDone returned %d events, want 0", len(second))
	}
}

// TestConverter_GeminiViaChatCompletionsThinking ensures the
// extra_content.google.thought=true signal routes text into a thinking
// block (this is the Gemini-over-OpenAI-Chat-Completions case used
// when a Gemini-compatible endpoint is fronted by the chat_completions
// adapter).
func TestConverter_GeminiViaChatCompletionsThinking(t *testing.T) {
	input := `
{"choices":[{"delta":{"content":"Reasoning here","extra_content":{"google":{"thought":true}}}}]}
{"choices":[{"delta":{"content":"Actual answer"}}]}
{"choices":[{"finish_reason":"stop","delta":{}}]}
`
	events := runConverter(t, input)
	// First content_block_start must be thinking.
	for _, e := range events {
		if e["type"] == "content_block_start" {
			block := e["content_block"].(map[string]any)
			if block["type"] != "thinking" {
				t.Fatalf("first content_block_start = %v, want thinking", block["type"])
			}
			break
		}
	}
	// Second content_block_start must be text.
	var nStart int
	var sawText bool
	for _, e := range events {
		if e["type"] == "content_block_start" {
			nStart++
			block := e["content_block"].(map[string]any)
			if nStart == 2 {
				if block["type"] != "text" {
					t.Fatalf("second content_block_start = %v, want text", block["type"])
				}
				sawText = true
			}
		}
	}
	if !sawText {
		t.Fatalf("expected a second (text) content_block_start")
	}
}
