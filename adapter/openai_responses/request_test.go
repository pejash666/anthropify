package openai_responses

import (
	"context"
	"encoding/json"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// TestBuildRequest_Basic exercises BuildRequest on a minimal Anthropic
// request and checks that the key shape transformations landed.
func TestBuildRequest_Basic(t *testing.T) {
	src := []byte(`{
		"model": "gpt-5",
		"max_tokens": 1024,
		"system": [{"type":"text","text":"You are helpful."}],
		"messages": [
			{"role":"user","content":[{"type":"text","text":"Hello"}]}
		]
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	out, err := BuildRequest(context.Background(), req, true)
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if got["model"] != "gpt-5" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["stream"] != true {
		t.Fatalf("stream = %v", got["stream"])
	}
	if !startsWith(asString(got["instructions"]), "You are helpful.") {
		t.Fatalf("instructions = %q", got["instructions"])
	}
	if got["max_output_tokens"].(float64) != 1024 {
		t.Fatalf("max_output_tokens = %v", got["max_output_tokens"])
	}
	input, ok := got["input"].([]any)
	if !ok || len(input) == 0 {
		t.Fatalf("input missing or wrong type: %v", got["input"])
	}
	first := input[0].(map[string]any)
	if first["role"] != "user" {
		t.Fatalf("first role = %v", first["role"])
	}
	content := first["content"].([]any)
	if len(content) == 0 {
		t.Fatalf("empty content")
	}
	cb := content[0].(map[string]any)
	if cb["type"] != "input_text" || cb["text"] != "Hello" {
		t.Fatalf("content[0] = %v", cb)
	}
}

// TestBuildRequest_ToolChoiceAndTools checks the tool translation path
// and the verbatim tool_choice forwarding.
func TestBuildRequest_ToolChoiceAndTools(t *testing.T) {
	src := []byte(`{
		"model":"gpt-5",
		"max_tokens":100,
		"messages":[{"role":"user","content":[{"type":"text","text":"."}]}],
		"tools":[{
			"name":"read_file",
			"description":"read a file",
			"input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
		}],
		"tool_choice":{"type":"any"}
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	out, _ := BuildRequest(context.Background(), req, false)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	tools, ok := got["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", got["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "read_file" {
		t.Fatalf("tool = %v", tool)
	}
	if _, ok := tool["parameters"].(map[string]any); !ok {
		t.Fatalf("parameters missing: %v", tool)
	}
	tc, ok := got["tool_choice"].(map[string]any)
	if !ok || tc["type"] != "any" {
		t.Fatalf("tool_choice = %v", got["tool_choice"])
	}
}

// TestBuildRequest_Thinking ensures the thinking budget is mapped to the
// reasoning block.
func TestBuildRequest_Thinking(t *testing.T) {
	src := []byte(`{
		"model":"gpt-5",
		"max_tokens":1,
		"messages":[{"role":"user","content":[{"type":"text","text":"."}]}],
		"thinking":{"type":"enabled","budget_tokens":2048}
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	out, _ := BuildRequest(context.Background(), req, false)
	var got map[string]any
	_ = json.Unmarshal(out, &got)
	r, ok := got["reasoning"].(map[string]any)
	if !ok {
		t.Fatalf("reasoning missing: %v", got)
	}
	if r["budget_tokens"].(float64) != 2048 {
		t.Fatalf("budget = %v", r["budget_tokens"])
	}
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
