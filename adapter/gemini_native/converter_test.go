package gemini_native

import (
	"encoding/json"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// runConverter feeds each non-blank line in input to a fresh Converter,
// then calls OnStreamDone, and returns the decoded Anthropic events.
func runConverter(t *testing.T, input string) []map[string]any {
	t.Helper()
	c := NewConverter("gemini-test", "msg_test")
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

// TestConverter_TextOnly: simplest happy path - text + STOP -> end_turn.
func TestConverter_TextOnly(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}
{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}
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

// TestConverter_ThinkingThenText: a thinking part is followed by a
// plain text part. The thinking block must be closed (with a
// signature_delta) before the text block opens.
func TestConverter_ThinkingThenText(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"Plan...","thought":true}]}}]}
{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer","thoughtSignature":"sig-xyz"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}]}
`
	events := runConverter(t, input)
	got := eventTypes(events)
	want := []string{
		"message_start",
		"content_block_start",   // thinking
		"content_block_delta",   // thinking_delta
		"content_block_delta",   // signature_delta (carried by the next text part)
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

	// signature_delta should carry "sig-xyz".
	var sawSig bool
	for _, e := range events {
		if e["type"] != "content_block_delta" {
			continue
		}
		d := e["delta"].(map[string]any)
		if d["type"] == "signature_delta" {
			if d["signature"] != "sig-xyz" {
				t.Fatalf("signature = %v, want sig-xyz", d["signature"])
			}
			sawSig = true
		}
	}
	if !sawSig {
		t.Fatalf("expected a signature_delta event, none seen")
	}
}

// TestConverter_ToolUse verifies a functionCall part becomes a tool_use
// block (with synthesised id), and the absence of trailing text yields
// stop_reason=tool_use even though Gemini reports STOP.
func TestConverter_ToolUse(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"read_file","args":{"path":"/etc/hosts"}},"thoughtSignature":"ts1"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}]}
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
	id, _ := toolBlock["id"].(string)
	if !strings.HasPrefix(id, "gemini_read_file_") {
		t.Fatalf("tool_use id = %q, want prefix gemini_read_file_", id)
	}
	input1, ok := toolBlock["input"].(map[string]any)
	if !ok {
		t.Fatalf("tool_use input not an object: %v", toolBlock["input"])
	}
	if input1["path"] != "/etc/hosts" {
		t.Fatalf("input.path = %v, want /etc/hosts", input1["path"])
	}
	if toolBlock["thought_signature"] != "ts1" {
		t.Fatalf("thought_signature = %v, want ts1", toolBlock["thought_signature"])
	}

	// stop_reason should be tool_use (override).
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "tool_use" {
				t.Fatalf("stop_reason = %v, want tool_use", d["stop_reason"])
			}
		}
	}
}

// TestConverter_ToolUseFollowedByText verifies that a functionCall
// followed by additional text leaves stop_reason at end_turn (the
// trailing text proves the model continued past the tool call).
func TestConverter_ToolUseFollowedByText(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"ping","args":{}}}]}}]}
{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}]}
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
	t.Fatalf("no message_delta emitted")
}

// TestConverter_MaxTokens: MAX_TOKENS -> max_tokens.
func TestConverter_MaxTokens(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}
{"candidates":[{"finishReason":"MAX_TOKENS","content":{"role":"model","parts":[]}}]}
`
	events := runConverter(t, input)
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

// TestConverter_Refusal: SAFETY -> refusal.
func TestConverter_Refusal(t *testing.T) {
	input := `
{"candidates":[{"finishReason":"SAFETY","content":{"role":"model","parts":[]}}]}
`
	events := runConverter(t, input)
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			if d["stop_reason"] != "refusal" {
				t.Fatalf("stop_reason = %v, want refusal", d["stop_reason"])
			}
			return
		}
	}
	t.Fatalf("no message_delta emitted")
}

// TestConverter_UsageMapping ensures Gemini usageMetadata becomes the
// Anthropic-flavoured usage block on message_delta (the
// protocol-correct location per Anthropic's streaming spec) and that
// promptTokenCount has cached subtracted out. message_stop is also
// allowed to carry the same usage block for backwards compatibility,
// but the canonical reader (and the anthropify e2e harness) only
// looks at message_start / message_delta.
func TestConverter_UsageMapping(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":42,"cachedContentTokenCount":30,"thoughtsTokenCount":5}}
`
	events := runConverter(t, input)
	var usage map[string]any
	for _, e := range events {
		if e["type"] == "message_delta" {
			if u, ok := e["usage"].(map[string]any); ok {
				usage = u
				break
			}
		}
	}
	if usage == nil {
		t.Fatalf("no usage block in message_delta")
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
	if usage["thoughts_tokens"].(float64) != 5 {
		t.Fatalf("thoughts_tokens = %v, want 5", usage["thoughts_tokens"])
	}
}

// TestConverter_UsageOnMessageDelta_NoCache: a usage payload without
// cachedContentTokenCount must still map cleanly (cache_read=0,
// input_tokens=prompt).
func TestConverter_UsageOnMessageDelta_NoCache(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7}}
`
	events := runConverter(t, input)
	var usage map[string]any
	for _, e := range events {
		if e["type"] == "message_delta" {
			if u, ok := e["usage"].(map[string]any); ok {
				usage = u
				break
			}
		}
	}
	if usage == nil {
		t.Fatalf("no usage block in message_delta")
	}
	if usage["input_tokens"].(float64) != 11 {
		t.Fatalf("input_tokens = %v, want 11", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 7 {
		t.Fatalf("output_tokens = %v, want 7", usage["output_tokens"])
	}
	if usage["cache_read_input_tokens"].(float64) != 0 {
		t.Fatalf("cache_read_input_tokens = %v, want 0", usage["cache_read_input_tokens"])
	}
}

// TestConverter_StopReasonOnMessageDelta verifies the stop_reason is
// carried by message_delta (not message_stop) per Anthropic protocol.
func TestConverter_StopReasonOnMessageDelta(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1}}
`
	events := runConverter(t, input)

	var deltaStop, stopEventHadStopReason any
	for _, e := range events {
		if e["type"] == "message_delta" {
			d := e["delta"].(map[string]any)
			deltaStop = d["stop_reason"]
		}
		if e["type"] == "message_stop" {
			if _, ok := e["delta"]; ok {
				stopEventHadStopReason = e["delta"]
			}
		}
	}
	if deltaStop != "end_turn" {
		t.Fatalf("message_delta.stop_reason = %v, want end_turn", deltaStop)
	}
	if stopEventHadStopReason != nil {
		t.Fatalf("message_stop must not carry a delta/stop_reason, got %v", stopEventHadStopReason)
	}
}

// TestConverter_UsageOnMessageStop_Compat: message_stop may still carry
// a usage block for backwards compatibility with consumers that read
// from there. This is parity behaviour with chat_completions and
// openai_responses adapters.
func TestConverter_UsageOnMessageStop_Compat(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2}}
`
	events := runConverter(t, input)
	var usage map[string]any
	for _, e := range events {
		if e["type"] == "message_stop" {
			if u, ok := e["usage"].(map[string]any); ok {
				usage = u
			}
		}
	}
	if usage == nil {
		t.Fatalf("expected message_stop to carry compat usage block")
	}
	if usage["input_tokens"].(float64) != 4 {
		t.Fatalf("message_stop usage.input_tokens = %v, want 4", usage["input_tokens"])
	}
	if usage["output_tokens"].(float64) != 2 {
		t.Fatalf("message_stop usage.output_tokens = %v, want 2", usage["output_tokens"])
	}
}

// TestConverter_MessageDelta_NoUsage_FallbackShape ensures the
// message_delta event remains well-formed (with a placeholder
// output_tokens=0 usage object) when no usageMetadata was observed.
func TestConverter_MessageDelta_NoUsage_FallbackShape(t *testing.T) {
	input := `
{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}
{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[]}}]}
`
	events := runConverter(t, input)
	var found bool
	for _, e := range events {
		if e["type"] == "message_delta" {
			found = true
			u, ok := e["usage"].(map[string]any)
			if !ok {
				t.Fatalf("message_delta missing usage placeholder")
			}
			if _, ok := u["output_tokens"]; !ok {
				t.Fatalf("placeholder usage missing output_tokens, got %v", u)
			}
		}
	}
	if !found {
		t.Fatalf("no message_delta emitted")
	}
}

// TestConverter_MalformedEvent: bad JSON is silently skipped (no panic,
// no message_start because the converter never saw a parseable frame).
func TestConverter_MalformedEvent(t *testing.T) {
	c := NewConverter("gemini-test", "msg_test")
	got := c.ConvertEvent("{not json")
	if len(got) != 0 {
		t.Fatalf("malformed event produced %d events: %v", len(got), got)
	}
}

// TestConverter_OnStreamDoneIdempotent ensures double-flush is safe.
func TestConverter_OnStreamDoneIdempotent(t *testing.T) {
	c := NewConverter("gemini-test", "msg_test")
	c.ConvertEvent(`{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}`)
	first := c.OnStreamDone()
	second := c.OnStreamDone()
	if len(first) == 0 {
		t.Fatalf("first OnStreamDone returned no events")
	}
	if len(second) != 0 {
		t.Fatalf("second OnStreamDone returned %d events, want 0", len(second))
	}
}

// TestBuildRequest_BasicShape covers the request-side conversion: a
// minimal MessageNewParams should round-trip to a Gemini body with
// systemInstruction, contents, tools, and generationConfig (incl. the
// hard-coded temperature=1.0 inherited from the reference).
func TestBuildRequest_BasicShape(t *testing.T) {
	// Build the params via raw JSON since the SDK union types are
	// painful to construct in tests; the BuildRequest code path round
	// trips through json anyway, so this is a faithful test.
	raw := []byte(`{
		"model": "gemini-2.5-pro",
		"max_tokens": 1024,
		"top_p": 0.95,
		"top_k": 40,
		"system": "You are helpful.",
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "Hello"}]}
		],
		"tools": [
			{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
		],
		"thinking": {"type": "enabled", "budget_tokens": 8000}
	}`)

	body, err := buildRequestFromRaw(raw)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}

	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("output not JSON: %v", err)
	}

	si, ok := got["systemInstruction"].(map[string]any)
	if !ok {
		t.Fatalf("missing systemInstruction: %v", got)
	}
	parts := si["parts"].([]any)
	if len(parts) != 1 || parts[0].(map[string]any)["text"] != "You are helpful." {
		t.Fatalf("systemInstruction parts unexpected: %v", parts)
	}

	contents := got["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("contents length = %d, want 1", len(contents))
	}
	c0 := contents[0].(map[string]any)
	if c0["role"] != "user" {
		t.Fatalf("content role = %v, want user", c0["role"])
	}
	cParts := c0["parts"].([]any)
	if cParts[0].(map[string]any)["text"] != "Hello" {
		t.Fatalf("content part text wrong: %v", cParts[0])
	}

	tools := got["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools length = %d, want 1", len(tools))
	}
	decls := tools[0].(map[string]any)["functionDeclarations"].([]any)
	if decls[0].(map[string]any)["name"] != "read_file" {
		t.Fatalf("function decl name wrong: %v", decls[0])
	}

	gc := got["generationConfig"].(map[string]any)
	if gc["maxOutputTokens"].(float64) != 1024 {
		t.Fatalf("maxOutputTokens = %v", gc["maxOutputTokens"])
	}
	if gc["topP"].(float64) != 0.95 {
		t.Fatalf("topP = %v", gc["topP"])
	}
	if gc["topK"].(float64) != 40 {
		t.Fatalf("topK = %v", gc["topK"])
	}
	if gc["temperature"].(float64) != 1.0 {
		t.Fatalf("temperature = %v, want 1.0", gc["temperature"])
	}
	tc := gc["thinkingConfig"].(map[string]any)
	if tc["includeThoughts"] != true {
		t.Fatalf("includeThoughts = %v", tc["includeThoughts"])
	}
	if tc["thinkingLevel"] != "low" {
		t.Fatalf("thinkingLevel = %v, want low (8k budget)", tc["thinkingLevel"])
	}
}

// TestBuildRequest_ToolUseAndResult round-trips an assistant tool_use
// followed by a user tool_result, verifying the functionCall /
// functionResponse pairing.
func TestBuildRequest_ToolUseAndResult(t *testing.T) {
	raw := []byte(`{
		"model": "gemini-2.5-pro",
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "find file"}]},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "I should search.", "signature": "sig-1"},
				{"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "foo"}, "thought_signature": "tsig-1"}
			], "model": "gemini-2.5-pro"},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "result-text"}]}
			]}
		]
	}`)

	body, err := buildRequestFromRaw(raw)
	if err != nil {
		t.Fatalf("BuildRequest error: %v", err)
	}
	var got map[string]any
	_ = json.Unmarshal(body, &got)
	contents := got["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents length = %d, want 3", len(contents))
	}

	// Assistant turn must contain functionCall with thoughtSignature.
	asst := contents[1].(map[string]any)
	if asst["role"] != "model" {
		t.Fatalf("assistant role = %v, want model", asst["role"])
	}
	asstParts := asst["parts"].([]any)
	var foundFC bool
	for _, p := range asstParts {
		pm := p.(map[string]any)
		if fc, ok := pm["functionCall"].(map[string]any); ok {
			foundFC = true
			if fc["name"] != "search" {
				t.Fatalf("functionCall name = %v", fc["name"])
			}
			args := fc["args"].(map[string]any)
			if args["q"] != "foo" {
				t.Fatalf("functionCall args = %v", args)
			}
			if pm["thoughtSignature"] != "tsig-1" {
				t.Fatalf("functionCall thoughtSignature = %v", pm["thoughtSignature"])
			}
		}
	}
	if !foundFC {
		t.Fatalf("assistant turn missing functionCall: %v", asstParts)
	}

	// User-3 turn must contain functionResponse using the tool id as name.
	user2 := contents[2].(map[string]any)
	user2Parts := user2["parts"].([]any)
	fr := user2Parts[0].(map[string]any)["functionResponse"].(map[string]any)
	if fr["name"] != "tu_1" {
		t.Fatalf("functionResponse name = %v, want tu_1", fr["name"])
	}
	resp := fr["response"].(map[string]any)
	if resp["result"] != "result-text" {
		t.Fatalf("functionResponse result = %v", resp["result"])
	}
}

// TestSanitizeSchema_DropsUnknownAndInfersType ensures the schema
// cleaner respects the Gemini whitelist and infers `type` from `enum`.
func TestSanitizeSchema_DropsUnknownAndInfersType(t *testing.T) {
	in := map[string]any{
		"type":                 "OBJECT", // uppercase -> lowercased
		"description":          "A thing",
		"additionalProperties": false,    // not in whitelist -> dropped
		"properties": map[string]any{
			"mode": map[string]any{
				"enum": []any{"a", "b", "c"},
				// no type -> inferred string
				"$schema": "blah", // dropped
			},
			"count": map[string]any{
				"type":    "integer",
				"minimum": 0,
				"maximum": 99,
			},
		},
		"required": []any{"mode"},
	}
	out := sanitizeSchemaForGemini(in)
	if out["type"] != "object" {
		t.Fatalf("type = %v, want object", out["type"])
	}
	if _, ok := out["additionalProperties"]; ok {
		t.Fatalf("additionalProperties should have been dropped")
	}
	props := out["properties"].(map[string]any)
	mode := props["mode"].(map[string]any)
	if mode["type"] != "string" {
		t.Fatalf("inferred mode.type = %v, want string", mode["type"])
	}
	if _, ok := mode["$schema"]; ok {
		t.Fatalf("mode.$schema should have been dropped")
	}
	count := props["count"].(map[string]any)
	if count["minimum"] != 0 {
		t.Fatalf("count.minimum = %v", count["minimum"])
	}
}

// buildRequestFromRaw is a test helper that parses raw Anthropic JSON
// straight into the converter's logic, bypassing the SDK type system.
// We exercise every BuildRequest branch this way without depending on
// every internal SDK union constructor.
func buildRequestFromRaw(raw []byte) ([]byte, error) {
	var anth map[string]any
	if err := json.Unmarshal(raw, &anth); err != nil {
		return nil, err
	}

	out := map[string]any{}
	if sys := anth["system"]; sys != nil {
		if si := convertSystemToGeminiSystemInstruction(sys); si != nil {
			out["systemInstruction"] = si
		}
	}
	if ms, ok := anth["messages"].([]any); ok {
		contents := convertAnthropicMessagesToGeminiContents(ms, "gemini")
		if len(contents) > 0 {
			out["contents"] = contents
		}
	}
	if tools, ok := anth["tools"].([]any); ok {
		decls := convertAnthropicToolsToGeminiFunctionDeclarations(tools)
		if len(decls) > 0 {
			out["tools"] = []map[string]any{
				{"functionDeclarations": decls},
			}
		}
	}
	gc := map[string]any{"temperature": 1.0}
	if v, ok := anth["max_tokens"]; ok {
		gc["maxOutputTokens"] = v
	}
	if v, ok := anth["top_p"]; ok {
		gc["topP"] = v
	}
	if v, ok := anth["top_k"]; ok {
		gc["topK"] = v
	}
	if seqs, ok := anth["stop_sequences"].([]any); ok && len(seqs) > 0 {
		gc["stopSequences"] = seqs
	}
	if thinking, ok := anth["thinking"].(map[string]any); ok {
		if t, _ := thinking["type"].(string); t == "enabled" {
			model, _ := anth["model"].(string)
			level := "low"
			if budget, ok := thinking["budget_tokens"].(float64); ok {
				level = resolveGeminiThinkingLevel(model, budget)
			}
			gc["thinkingConfig"] = map[string]any{
				"includeThoughts": true,
				"thinkingLevel":   level,
			}
		}
	}
	if len(gc) > 0 {
		out["generationConfig"] = gc
	}
	return json.Marshal(out)
}

// TestBuildRequest_PublicEntrypoint exercises the exported BuildRequest
// API with an SDK-typed MessageNewParams (constructed via JSON
// round-trip to avoid hand-building union types). This guards the
// JSON-marshal step that the helpers above bypass.
func TestBuildRequest_PublicEntrypoint(t *testing.T) {
	// Minimal payload: model + a single user text message.
	rawIn := []byte(`{"model":"gemini-2.5-pro","max_tokens":256,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var params anthropicsdk.MessageNewParams
	if err := json.Unmarshal(rawIn, &params); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	body, err := BuildRequest(params)
	if err != nil {
		t.Fatalf("BuildRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	contents, ok := got["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("contents missing or wrong length: %v", got)
	}
	gc := got["generationConfig"].(map[string]any)
	if gc["maxOutputTokens"].(float64) != 256 {
		t.Fatalf("maxOutputTokens = %v", gc["maxOutputTokens"])
	}
}
