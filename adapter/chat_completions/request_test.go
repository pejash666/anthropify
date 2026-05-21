package chat_completions

import (
	"encoding/json"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// buildRequestFromRaw bypasses the SDK union types by feeding raw
// Anthropic JSON through the internal converter helpers. We replicate
// the BuildRequest top-level wiring here so tests can exercise every
// branch (thinking, tools, system, kimi-k2 rules) without hand-rolling
// every union constructor in the SDK.
func buildRequestFromRaw(raw []byte, stream bool) ([]byte, error) {
	var anth map[string]any
	if err := json.Unmarshal(raw, &anth); err != nil {
		return nil, err
	}

	out := map[string]any{}

	model := ""
	if m, ok := anth["model"].(string); ok {
		out["model"] = m
		model = m
	}

	systemContent := convertSystemContent(anth["system"])
	var messages []map[string]any
	if ms, ok := anth["messages"].([]any); ok {
		messages = convertAnthropicMessages(ms, systemContent, model)
	}
	out["messages"] = messages

	needDisableThinking := false
	if isKimiK2Family(model) {
		for _, msg := range messages {
			if role, ok := msg["role"].(string); ok && role == "assistant" {
				hasToolCalls := false
				if tc, ok := msg["tool_calls"].([]map[string]any); ok && len(tc) > 0 {
					hasToolCalls = true
				} else if tc, ok := msg["tool_calls"].([]any); ok && len(tc) > 0 {
					hasToolCalls = true
				}
				if hasToolCalls {
					if _, hasReasoning := msg["reasoning_content"]; !hasReasoning {
						needDisableThinking = true
						break
					}
				}
			}
		}
	}

	if tools, ok := anth["tools"].([]any); ok {
		outTools := convertAnthropicTools(tools)
		if len(outTools) > 0 {
			out["tools"] = outTools
		}
	}

	if toolChoice := convertToolChoice(anth["tool_choice"]); toolChoice != nil {
		out["tool_choice"] = toolChoice
	}

	if !isKimiK2Family(model) {
		if v, ok := anth["temperature"]; ok {
			out["temperature"] = v
		}
	}

	isGLM := false
	if model != "" && len(model) >= 4 && (model[:4] == "glm-" || model[:4] == "GLM-") {
		isGLM = true
	}
	isDS := isDeepSeekPassthrough(model)

	switch {
	case needDisableThinking && isKimiK2Family(model):
		out["thinking"] = map[string]any{"type": "disabled"}
	case anth["thinking"] != nil:
		if thinkingMap, ok := anth["thinking"].(map[string]any); ok {
			if isGLM || isDS {
				out["thinking"] = thinkingMap
			} else if enabled, ok := thinkingMap["type"].(string); ok && enabled == "enabled" {
				budgetValue := thinkingMap["budget_tokens"]
				if budgetValue == nil {
					budgetValue = thinkingMap["thinking_budget"]
				}
				var budget int64
				okv := false
				switch v := budgetValue.(type) {
				case float64:
					budget = int64(v)
					okv = true
				case int64:
					budget = v
					okv = true
				case int:
					budget = int64(v)
					okv = true
				}
				if okv {
					switch {
					case budget < 20000:
						out["reasoning_effort"] = "low"
					case budget < 31999:
						out["reasoning_effort"] = "medium"
					default:
						out["reasoning_effort"] = "high"
					}
				}
			}
		}
	case isGLM:
		out["thinking"] = map[string]any{"type": "disabled"}
	}

	if v, ok := anth["max_tokens"]; ok {
		out["max_completion_tokens"] = v
	}

	out["stream"] = stream
	return json.Marshal(out)
}

// TestBuildRequest_BasicShape: simplest request → minimal OpenAI body.
func TestBuildRequest_BasicShape(t *testing.T) {
	raw := []byte(`{
		"model": "deepseek-chat",
		"max_tokens": 1024,
		"temperature": 0.7,
		"system": "You are helpful.",
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello"}]}
		]
	}`)
	body, err := buildRequestFromRaw(raw, true)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("output not JSON: %v", err)
	}
	if got["model"] != "deepseek-chat" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["max_completion_tokens"].(float64) != 1024 {
		t.Fatalf("max_completion_tokens = %v", got["max_completion_tokens"])
	}
	if got["temperature"].(float64) != 0.7 {
		t.Fatalf("temperature = %v", got["temperature"])
	}
	if got["stream"] != true {
		t.Fatalf("stream = %v", got["stream"])
	}
	messages := got["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("messages length = %d, want 2 (system+user)", len(messages))
	}
	sys := messages[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "You are helpful." {
		t.Fatalf("system message wrong: %v", sys)
	}
	usr := messages[1].(map[string]any)
	if usr["role"] != "user" || usr["content"] != "Hello" {
		t.Fatalf("user message wrong: %v", usr)
	}
}

// TestBuildRequest_WithTools: tools and tool_choice mapping.
func TestBuildRequest_WithTools(t *testing.T) {
	raw := []byte(`{
		"model": "deepseek-chat",
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "x"}]}
		],
		"tools": [
			{"name": "read_file", "description": "Read a file",
			 "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
		],
		"tool_choice": {"type": "tool", "name": "read_file"}
	}`)
	body, err := buildRequestFromRaw(raw, false)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	_ = json.Unmarshal(body, &got)
	tools := got["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools length = %d", len(tools))
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Fatalf("tool type = %v", tool["type"])
	}
	fn := tool["function"].(map[string]any)
	if fn["name"] != "read_file" {
		t.Fatalf("fn name = %v", fn["name"])
	}
	params := fn["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Fatalf("params.type = %v", params["type"])
	}
	tc := got["tool_choice"].(map[string]any)
	if tc["type"] != "function" {
		t.Fatalf("tool_choice.type = %v", tc["type"])
	}
	tcf := tc["function"].(map[string]any)
	if tcf["name"] != "read_file" {
		t.Fatalf("tool_choice.function.name = %v", tcf["name"])
	}
}

// TestBuildRequest_ThinkingConfig: thinking with budget_tokens maps to
// reasoning_effort buckets (low / medium / high).
func TestBuildRequest_ThinkingConfig(t *testing.T) {
	cases := []struct {
		budget float64
		want   string
	}{
		{8000, "low"},
		{20000, "medium"},
		{25000, "medium"},
		{32000, "high"},
		{64000, "high"},
	}
	for _, c := range cases {
		raw := []byte(`{
			"model": "qwen-max",
			"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
			"thinking": {"type": "enabled", "budget_tokens": ` + jsonNumber(c.budget) + `}
		}`)
		body, err := buildRequestFromRaw(raw, false)
		if err != nil {
			t.Fatalf("BuildRequest error: %v", err)
		}
		var got map[string]any
		_ = json.Unmarshal(body, &got)
		if got["reasoning_effort"] != c.want {
			t.Fatalf("budget=%v: reasoning_effort = %v, want %v", c.budget, got["reasoning_effort"], c.want)
		}
		// Should not echo the raw thinking block for non-passthrough providers.
		if _, ok := got["thinking"]; ok {
			t.Fatalf("budget=%v: thinking block must not be echoed for non-passthrough providers", c.budget)
		}
	}
}

// TestBuildRequest_GLMThinkingPassthrough: GLM models keep the raw
// thinking block (passthrough) and never emit reasoning_effort.
func TestBuildRequest_GLMThinkingPassthrough(t *testing.T) {
	raw := []byte(`{
		"model": "glm-4.6",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
		"thinking": {"type": "enabled", "budget_tokens": 8000}
	}`)
	body, err := buildRequestFromRaw(raw, false)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	_ = json.Unmarshal(body, &got)
	th := got["thinking"].(map[string]any)
	if th["type"] != "enabled" {
		t.Fatalf("thinking.type = %v, want enabled (passthrough)", th["type"])
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort must not be set for GLM")
	}
}

// TestBuildRequest_GLMDefaultDisabled: GLM with no thinking block must
// get thinking={type:"disabled"} attached so the model does not
// silently emit reasoning.
func TestBuildRequest_GLMDefaultDisabled(t *testing.T) {
	raw := []byte(`{
		"model": "glm-4.6",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
	}`)
	body, err := buildRequestFromRaw(raw, false)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	_ = json.Unmarshal(body, &got)
	th, ok := got["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking block missing")
	}
	if th["type"] != "disabled" {
		t.Fatalf("thinking.type = %v, want disabled", th["type"])
	}
}

// TestBuildRequest_Kimi2FamilyDisableThinking: kimi-k2 family with a
// historical assistant tool_call but no reasoning_content -> the
// adapter must inject thinking={type:"disabled"} and skip temperature.
func TestBuildRequest_Kimi2FamilyDisableThinking(t *testing.T) {
	raw := []byte(`{
		"model": "kimi-k2-5",
		"temperature": 0.7,
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "search"}]},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "foo"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "ok"}]}
			]}
		]
	}`)
	body, err := buildRequestFromRaw(raw, false)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	_ = json.Unmarshal(body, &got)
	th := got["thinking"].(map[string]any)
	if th["type"] != "disabled" {
		t.Fatalf("thinking.type = %v, want disabled", th["type"])
	}
	// temperature must be stripped for kimi-k2 family.
	if _, ok := got["temperature"]; ok {
		t.Fatalf("temperature must not be present for kimi-k2 family")
	}
}

// TestBuildRequest_PublicEntrypoint: tests the exported BuildRequest
// API with a real SDK MessageNewParams (built via JSON unmarshal).
func TestBuildRequest_PublicEntrypoint(t *testing.T) {
	rawIn := []byte(`{"model":"deepseek-chat","max_tokens":256,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var params anthropicsdk.MessageNewParams
	if err := json.Unmarshal(rawIn, &params); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	body, err := BuildRequest(params, true)
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got["model"] != "deepseek-chat" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["max_completion_tokens"].(float64) != 256 {
		t.Fatalf("max_completion_tokens = %v", got["max_completion_tokens"])
	}
}

// TestBuildRequest_ReasoningModelKeepsThinking: a reasoning model
// (kimi-k2-thinking) preserves the assistant thinking content as
// reasoning_content on the converted history.
func TestBuildRequest_ReasoningModelKeepsThinking(t *testing.T) {
	raw := []byte(`{
		"model": "kimi-k2-thinking",
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "q"}]},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "let me think"},
				{"type": "text", "text": "answer"}
			]}
		]
	}`)
	body, err := buildRequestFromRaw(raw, false)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	_ = json.Unmarshal(body, &got)
	messages := got["messages"].([]any)
	asst := messages[1].(map[string]any)
	if asst["reasoning_content"] != "let me think" {
		t.Fatalf("assistant reasoning_content = %v", asst["reasoning_content"])
	}
}

// jsonNumber is a tiny helper to embed a float into a JSON string body
// without losing precision for round numbers used in tests.
func jsonNumber(f float64) string {
	b, _ := json.Marshal(f)
	return string(b)
}
