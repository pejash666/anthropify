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

// TestBuildRequest_TopLevelCacheControl_DroppedAsNoOp documents and
// asserts the v0.2.0 Track C contract: when the canonical anthropify
// layer installs a top-level cache_control directive (via
// SetCacheControl), the openai_responses adapter must NOT forward it
// to the OpenAI Responses /v1/responses wire body. OpenAI prompt
// caching is fully automatic on prompts > 1024 tokens server-side
// and has no analogous request field.
func TestBuildRequest_TopLevelCacheControl_DroppedAsNoOp(t *testing.T) {
	src := []byte(`{
		"model": "gpt-5",
		"max_tokens": 64,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	// Mirror anthropify.SetCacheControl from the canonical layer.
	req.SetExtraFields(map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	})
	out, err := BuildRequest(context.Background(), req, false)
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := got["cache_control"]; ok {
		t.Fatalf("openai_responses wire body must drop cache_control; got=%s", out)
	}
}

// TestBuildRequestForAzure_OverridesModel asserts the layered model
// fallback rule from docs/design/v0.2.0-azure-responses.md §5.4: when
// effectiveModel is non-empty, the request body's `model` field is
// replaced with it. Azure deployment-bound URLs already encode the
// deployment in the path, but the body field still expects the
// deployment name on Azure resources, so we set both.
func TestBuildRequestForAzure_OverridesModel(t *testing.T) {
	src := []byte(`{
		"model": "gpt-5",
		"max_tokens": 64,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	out, err := BuildRequestForAzure(context.Background(), req, true, "azure-gpt5-PTU")
	if err != nil {
		t.Fatalf("BuildRequestForAzure: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["model"] != "azure-gpt5-PTU" {
		t.Fatalf("model = %v, want azure-gpt5-PTU", got["model"])
	}
}

// TestBuildRequestForAzure_EmptyOverrideKeepsCanonical proves the
// pass-through layer of the fallback: empty effectiveModel must leave
// req.Model untouched. This is what the deployment-less Azure path
// relies on (Route.UpstreamModel rewrite happens upstream in
// Client.dispatch; the adapter sees the rewritten value as req.Model).
func TestBuildRequestForAzure_EmptyOverrideKeepsCanonical(t *testing.T) {
	src := []byte(`{
		"model": "gpt-5",
		"max_tokens": 64,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	out, err := BuildRequestForAzure(context.Background(), req, true, "")
	if err != nil {
		t.Fatalf("BuildRequestForAzure: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got["model"] != "gpt-5" {
		t.Fatalf("model = %v, want gpt-5 (canonical pass-through)", got["model"])
	}
}

// TestBuildRequestForAzure_MatchesBuildRequestWhenEmpty is a stronger
// invariant: byte-identical output to BuildRequest when no override
// is requested. Guards against accidental regression of the vanilla
// path through the new entry point.
func TestBuildRequestForAzure_MatchesBuildRequestWhenEmpty(t *testing.T) {
	src := []byte(`{
		"model": "gpt-5",
		"max_tokens": 64,
		"system": [{"type":"text","text":"You are helpful."}],
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	a, err := BuildRequest(context.Background(), req, true)
	if err != nil {
		t.Fatal(err)
	}
	b, err := BuildRequestForAzure(context.Background(), req, true, "")
	if err != nil {
		t.Fatal(err)
	}
	if string(a) != string(b) {
		t.Fatalf("byte mismatch:\n  BuildRequest         = %s\n  BuildRequestForAzure = %s", a, b)
	}
}

// TestBuildRequestForAzure_CacheControlNoop extends the Track C
// silent-no-op contract to the Azure entry point. Azure OpenAI's
// Responses API uses the same automatic server-side cache as vanilla
// OpenAI; the body must NOT carry cache_control.
func TestBuildRequestForAzure_CacheControlNoop(t *testing.T) {
	src := []byte(`{
		"model": "gpt-5",
		"max_tokens": 64,
		"messages": [{"role":"user","content":[{"type":"text","text":"hi"}]}]
	}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	req.SetExtraFields(map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
	})
	out, err := BuildRequestForAzure(context.Background(), req, false, "azure-deployment")
	if err != nil {
		t.Fatalf("BuildRequestForAzure: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if _, ok := got["cache_control"]; ok {
		t.Fatalf("Azure wire body must drop cache_control; got=%s", out)
	}
	if got["model"] != "azure-deployment" {
		t.Fatalf("model not overridden: %v", got["model"])
	}
}
