package openai_responses

import (
	"encoding/json"
	"strings"
	"testing"
)

// decodeAll runs the Converter over every line in input (one JSON event
// per line, blank lines ignored) and returns the resulting Anthropic
// events parsed into maps for easy assertion.
func decodeAll(t *testing.T, input string) []map[string]any {
	t.Helper()
	c := NewConverter("msg_test")
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
	return out
}

func typesOf(evts []map[string]any) []string {
	out := make([]string, 0, len(evts))
	for _, e := range evts {
		if s, _ := e["type"].(string); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// TestConverter_TextOnly verifies the simplest happy path: a response
// that emits a plain text block and completes.
func TestConverter_TextOnly(t *testing.T) {
	input := `
{"type":"response.created","response":{"model":"gpt-5"}}
{"type":"response.output_text.delta","delta":"Hello"}
{"type":"response.output_text.delta","delta":" world"}
{"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":2}}}
`
	events := decodeAll(t, input)
	got := typesOf(events)
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

	// Check the text was preserved.
	var accum string
	for _, e := range events {
		if e["type"] != "content_block_delta" {
			continue
		}
		d, _ := e["delta"].(map[string]any)
		if d["type"] == "text_delta" {
			accum += d["text"].(string)
		}
	}
	if accum != "Hello world" {
		t.Fatalf("text accum = %q, want %q", accum, "Hello world")
	}

	// Check stop_reason.
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "end_turn" {
				t.Fatalf("stop_reason = %v, want end_turn", d["stop_reason"])
			}
		}
	}
}

// TestConverter_ThinkingThenText verifies that a reasoning block is
// closed before a text block is opened.
func TestConverter_ThinkingThenText(t *testing.T) {
	input := `
{"type":"response.created","response":{"model":"gpt-5"}}
{"type":"response.reasoning_summary_text.delta","delta":"Plan..."}
{"type":"response.output_text.delta","delta":"Answer"}
{"type":"response.completed","response":{}}
`
	events := decodeAll(t, input)
	got := typesOf(events)
	want := []string{
		"message_start",
		"content_block_start",   // thinking
		"content_block_delta",   // thinking_delta
		"content_block_stop",    // close thinking
		"content_block_start",   // text
		"content_block_delta",   // text_delta
		"content_block_stop",    // close text
		"message_delta",
		"message_stop",
	}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence mismatch:\n got %v\nwant %v", got, want)
	}

	// First content_block_start should be type thinking.
	for _, e := range events {
		if e["type"] == "content_block_start" {
			block := e["content_block"].(map[string]any)
			if block["type"] != "thinking" {
				t.Fatalf("first content_block_start type = %v, want thinking", block["type"])
			}
			break
		}
	}
}

// TestConverter_ToolUse verifies a function_call produces the expected
// tool_use block and tool_use stop reason when no post-tool text arrives.
func TestConverter_ToolUse(t *testing.T) {
	input := `
{"type":"response.created","response":{"model":"gpt-5"}}
{"type":"response.output_item.added","item":{"type":"function_call","id":"call_1","name":"read_file"}}
{"type":"response.function_call_arguments.delta","delta":"{\"path\":"}
{"type":"response.function_call_arguments.delta","delta":"\"/a\"}"}
{"type":"response.function_call_arguments.done"}
{"type":"response.completed","response":{}}
`
	events := decodeAll(t, input)

	// Find the tool_use block start.
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
	if toolBlock["id"] != "call_1" || toolBlock["name"] != "read_file" {
		t.Fatalf("tool_use block = %v", toolBlock)
	}

	// Stop reason must be tool_use.
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "tool_use" {
				t.Fatalf("stop_reason = %v, want tool_use", d["stop_reason"])
			}
		}
	}

	// Arguments were forwarded as input_json_delta partial_json.
	var accum string
	for _, e := range events {
		if e["type"] != "content_block_delta" {
			continue
		}
		d := e["delta"].(map[string]any)
		if d["type"] == "input_json_delta" {
			accum += d["partial_json"].(string)
		}
	}
	if accum != `{"path":"/a"}` {
		t.Fatalf("accumulated args = %q", accum)
	}
}

// TestConverter_IncompleteMaxTokens maps the incomplete_details reason
// to the Anthropic max_tokens stop reason.
func TestConverter_IncompleteMaxTokens(t *testing.T) {
	input := `
{"type":"response.created","response":{"model":"gpt-5"}}
{"type":"response.output_text.delta","delta":"x"}
{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}
`
	events := decodeAll(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "max_tokens" {
				t.Fatalf("stop_reason = %v, want max_tokens", d["stop_reason"])
			}
			return
		}
	}
	t.Fatalf("no message_delta emitted")
}

// TestConverter_UsageMapping ensures the OpenAI usage block is
// translated into Anthropic's cache_* / input_tokens / output_tokens
// shape.
func TestConverter_UsageMapping(t *testing.T) {
	input := `
{"type":"response.created","response":{"model":"gpt-5"}}
{"type":"response.output_text.delta","delta":"x"}
{"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":42,"input_tokens_details":{"cached_tokens":30}}}}
`
	events := decodeAll(t, input)
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
}
