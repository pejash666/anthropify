package openai_responses

import (
	"context"
	"encoding/json"
	"fmt"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/anthropify/internal/schema"
)

// BuildRequest lowers an Anthropic MessageNewParams into the JSON body
// expected by POST /v1/responses.
//
// This is a deliberately conservative port: it handles text messages,
// system prompts, tools, tool_choice, temperature, top_p, max_tokens and
// thinking budget. More exotic features (image inputs, server-side
// tools beyond apply_patch, structured output) are forwarded verbatim
// when present and passed through to the upstream API.
//
// ctx is consulted for the active schema.Policy via schema.PolicyFrom;
// the default PolicyStrict makes incompatible tool input_schema fields
// surface as ErrSchemaIncompatible instead of being silently passed
// through and rejected by the upstream API. This fixes the latent bug
// where convertTools used to forward the input_schema verbatim, so
// $ref / additionalProperties:true caused opaque 400s from the
// Responses API.
func BuildRequest(ctx context.Context, req anthropicsdk.MessageNewParams, stream bool) ([]byte, error) {
	return BuildRequestForAzure(ctx, req, stream, "")
}

// BuildRequestForAzure mirrors BuildRequest but, when effectiveModel is
// non-empty, overrides the wire-side `model` field with it. Used by
// the Azure adapter to surface cfg.Deployment in the request body —
// Azure expects the deployment name in `model` regardless of which URL
// form is used. Passing "" gives byte-identical output to BuildRequest.
func BuildRequestForAzure(ctx context.Context, req anthropicsdk.MessageNewParams, stream bool, effectiveModel string) ([]byte, error) {
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("openai_responses: marshal anthropic request: %w", err)
	}
	var src map[string]any
	if err := json.Unmarshal(raw, &src); err != nil {
		return nil, err
	}

	out := map[string]any{
		"model":  src["model"],
		"stream": stream,
	}
	if effectiveModel != "" {
		out["model"] = effectiveModel
	}

	// Map system prompt -> instructions.
	if sys, ok := src["system"]; ok {
		if instr := flattenSystem(sys); instr != "" {
			out["instructions"] = instr
		}
	}

	// Map messages -> input (Responses API accepts a flattened list).
	if msgs, ok := src["messages"].([]any); ok {
		out["input"] = convertMessages(msgs)
	}

	// Tools.
	if tools, ok := src["tools"].([]any); ok && len(tools) > 0 {
		converted, err := convertTools(ctx, tools)
		if err != nil {
			return nil, err
		}
		out["tools"] = converted
	}
	if tc, ok := src["tool_choice"]; ok {
		out["tool_choice"] = tc
	}

	// Sampling params.
	if v, ok := src["temperature"]; ok {
		out["temperature"] = v
	}
	if v, ok := src["top_p"]; ok {
		out["top_p"] = v
	}
	if v, ok := src["max_tokens"]; ok {
		out["max_output_tokens"] = v
	}

	// Thinking budget.
	if th, ok := src["thinking"].(map[string]any); ok {
		if budget, ok := th["budget_tokens"]; ok {
			out["reasoning"] = map[string]any{
				"effort":        "medium",
				"budget_tokens": budget,
			}
		} else if t, _ := th["type"].(string); t == "enabled" {
			out["reasoning"] = map[string]any{"effort": "medium"}
		}
	}

	return json.Marshal(out)
}

// flattenSystem normalises Anthropic's polymorphic system field to a
// plain string. It accepts either a string or an array of {type:"text",
// text:"..."} blocks.
func flattenSystem(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []any:
		var parts []byte
		for _, b := range t {
			if m, ok := b.(map[string]any); ok {
				if s, _ := m["text"].(string); s != "" {
					parts = append(parts, s...)
					parts = append(parts, '\n')
				}
			}
		}
		return string(parts)
	}
	return ""
}

// convertMessages turns Anthropic Messages into Responses-API input items.
// The mapping is intentionally lossy-but-sound: Anthropic content blocks
// become OpenAI input_text / input_image / tool_result entries and
// tool_use blocks become function_call items.
func convertMessages(msgs []any) []any {
	var out []any
	for _, mi := range msgs {
		m, ok := mi.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		content := m["content"]

		switch c := content.(type) {
		case string:
			out = append(out, map[string]any{
				"role":    role,
				"content": []any{map[string]any{"type": textFieldForRole(role), "text": c}},
			})
		case []any:
			var parts []any
			for _, bi := range c {
				b, ok := bi.(map[string]any)
				if !ok {
					continue
				}
				blockType, _ := b["type"].(string)
				switch blockType {
				case "text":
					parts = append(parts, map[string]any{
						"type": textFieldForRole(role),
						"text": b["text"],
					})
				case "image":
					// Responses API expects input_image with image_url.
					if src, ok := b["source"].(map[string]any); ok {
						url := ""
						if u, ok := src["url"].(string); ok {
							url = u
						} else if data, ok := src["data"].(string); ok {
							mediaType, _ := src["media_type"].(string)
							url = "data:" + mediaType + ";base64," + data
						}
						if url != "" {
							parts = append(parts, map[string]any{
								"type":      "input_image",
								"image_url": url,
							})
						}
					}
				case "tool_use":
					// Emit a separate function_call item; tool_use is
					// always an assistant action.
					call := map[string]any{
						"type":      "function_call",
						"call_id":   b["id"],
						"name":      b["name"],
						"arguments": toJSONString(b["input"]),
					}
					out = append(out, call)
				case "tool_result":
					call := map[string]any{
						"type":    "function_call_output",
						"call_id": b["tool_use_id"],
						"output":  toJSONString(b["content"]),
					}
					out = append(out, call)
				}
			}
			if len(parts) > 0 {
				out = append(out, map[string]any{"role": role, "content": parts})
			}
		}
	}
	return out
}

// textFieldForRole returns "input_text" for user/tool content and
// "output_text" for assistant content, as required by the Responses API.
func textFieldForRole(role string) string {
	if role == "assistant" {
		return "output_text"
	}
	return "input_text"
}

// convertTools maps Anthropic tool schemas to OpenAI function tools.
// Each tool's input_schema is run through schema.Normalize against
// DialectOpenAIStrict so that fields the upstream rejects
// (e.g. $ref, additionalProperties:true under structured outputs) are
// surfaced as ErrSchemaIncompatible under PolicyStrict (default) or
// rewritten under PolicyLossy/PolicyBestEffort with slog.Warn breadcrumbs.
//
// This fixes the latent bug noted in docs/design/v0.2.0-...md §1: prior
// versions forwarded the schema verbatim, so dialect-incompatible
// fields produced opaque upstream 400s instead of a localised error.
func convertTools(ctx context.Context, tools []any) ([]any, error) {
	policy := schema.PolicyFrom(ctx)
	out := make([]any, 0, len(tools))
	for _, ti := range tools {
		t, ok := ti.(map[string]any)
		if !ok {
			continue
		}
		// Anthropic built-in or custom tools both carry name+input_schema.
		name, _ := t["name"].(string)
		if name == "" {
			continue
		}
		fn := map[string]any{
			"type": "function",
			"name": name,
		}
		if desc, _ := t["description"].(string); desc != "" {
			fn["description"] = desc
		}
		if rawSchema, ok := t["input_schema"]; ok {
			canonical, _ := rawSchema.(map[string]any)
			normalised, _, err := schema.Normalize(ctx, canonical, schema.DialectOpenAIStrict, policy)
			if err != nil {
				return nil, fmt.Errorf("openai_responses: tool %q: %w", name, err)
			}
			if normalised != nil {
				fn["parameters"] = normalised
			} else {
				fn["parameters"] = rawSchema
			}
		}
		out = append(out, fn)
	}
	return out, nil
}

// toJSONString serialises arbitrary content to a JSON string, matching
// the Responses API's function_call.arguments convention.
func toJSONString(v any) string {
	if v == nil {
		return "{}"
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}
