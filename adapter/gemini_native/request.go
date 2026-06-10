package gemini_native

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/pejash666/anthropify/internal/schema"
)

// BuildRequest converts an Anthropic MessageNewParams payload into a
// Gemini :streamGenerateContent JSON body.
//
// The implementation is a direct port of llm-proxy's
// ConvertAnthropicToGeminiNative (service/llm/converter.go). Like the
// reference, it round-trips the SDK request through map[string]any so
// that we never have to enumerate every Anthropic field by hand.
//
// Server-side logging, the third-party-proxy context detection, and
// fallbacks specific to the proxy's logging plane have been dropped;
// otherwise the produced body matches the reference field-for-field
// for the supported subset of Anthropic features (text/image content,
// tool_use, tool_result, thinking, system, tools, max_tokens, top_p,
// top_k, stop_sequences, thinking config).
//
// ctx is consulted for the active schema.Policy via schema.PolicyFrom;
// the default PolicyStrict makes incompatible tool input_schema fields
// (oneOf, $ref, multipleOf, etc.) surface as ErrSchemaIncompatible
// instead of silently disappearing.
func BuildRequest(ctx context.Context, req anthropicsdk.MessageNewParams) ([]byte, error) {
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("anthropify/gemini_native: marshal request: %w", err)
	}

	var anth map[string]any
	if err := json.Unmarshal(raw, &anth); err != nil {
		return nil, fmt.Errorf("anthropify/gemini_native: unmarshal request: %w", err)
	}

	out := map[string]any{}

	// 1. system -> systemInstruction
	if sys := anth["system"]; sys != nil {
		if si := convertSystemToGeminiSystemInstruction(sys); si != nil {
			out["systemInstruction"] = si
		}
	}

	// 2. messages -> contents
	if ms, ok := anth["messages"].([]any); ok {
		contents := convertAnthropicMessagesToGeminiContents(ms, "gemini")
		if len(contents) > 0 {
			out["contents"] = contents
		}
	}

	// 3. tools -> tools[0].functionDeclarations
	if tools, ok := anth["tools"].([]any); ok {
		decls, err := convertAnthropicToolsToGeminiFunctionDeclarations(ctx, tools)
		if err != nil {
			return nil, err
		}
		if len(decls) > 0 {
			out["tools"] = []map[string]any{
				{"functionDeclarations": decls},
			}
		}
	}

	// 4. generationConfig
	gc := map[string]any{}
	gc["temperature"] = 1.0 // matches the reference's hard-coded default

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

	// 5. thinking config
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

func convertSystemToGeminiSystemInstruction(sys any) map[string]any {
	if sys == nil {
		return nil
	}
	var parts []map[string]any
	switch s := sys.(type) {
	case string:
		if s != "" {
			parts = append(parts, map[string]any{"text": s})
		}
	case []any:
		for _, block := range s {
			m, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if t, _ := m["type"].(string); t == "text" {
				if text, _ := m["text"].(string); text != "" {
					parts = append(parts, map[string]any{"text": text})
				}
			}
		}
	}
	if len(parts) == 0 {
		return nil
	}
	return map[string]any{"parts": parts}
}

// convertAnthropicMessagesToGeminiContents mirrors the reference at
// service/llm/converter.go:1455. The reference filters thinking blocks
// based on whether the message-level "model" field starts with the
// target prefix ("gemini"), and falls back to a third-party-proxy
// context flag for older client formats. We retain the model-prefix
// rule but simplify the fallback: when no message.model is present we
// keep the thinking block (Anthropic SDK requests will not include the
// proxy's custom field, so dropping them would silently break the
// thought-signature replay loop).
func convertAnthropicMessagesToGeminiContents(messages []any, targetModel string) []map[string]any {
	contents := make([]map[string]any, 0)

	for _, mi := range messages {
		mm, ok := mi.(map[string]any)
		if !ok {
			continue
		}

		role, _ := mm["role"].(string)
		blocks, _ := mm["content"].([]any)
		messageModel, _ := mm["model"].(string)

		// First-pass: collect a fallback thinking signature for tool_use
		// blocks that don't carry their own thoughtSignature.
		var thinkingSignature string
		if role == "assistant" && messageModel != "" {
			for _, ci := range blocks {
				cm, ok := ci.(map[string]any)
				if !ok {
					continue
				}
				if t, _ := cm["type"].(string); t == "thinking" {
					if sig, _ := cm["signature"].(string); sig != "" && strings.HasPrefix(messageModel, targetModel) {
						thinkingSignature = sig
						break
					}
				}
			}
		}

		geminiRole := role
		if role == "assistant" {
			geminiRole = "model"
		}
		// "system" role messages should not appear here; the reference
		// only handles user/assistant in this path. Skip anything else
		// to avoid producing a contents[] entry the API will reject.
		if geminiRole != "user" && geminiRole != "model" {
			continue
		}

		parts := make([]map[string]any, 0, len(blocks))

		// Anthropic also allows `content` to be a plain string.
		if blocks == nil {
			if text, ok := mm["content"].(string); ok && text != "" {
				parts = append(parts, map[string]any{"text": text})
			}
		}

		for _, ci := range blocks {
			cm, ok := ci.(map[string]any)
			if !ok {
				continue
			}
			blockType, _ := cm["type"].(string)
			switch blockType {
			case "text":
				if text, _ := cm["text"].(string); text != "" {
					parts = append(parts, map[string]any{"text": text})
				}

			case "image":
				src, ok := cm["source"].(map[string]any)
				if !ok {
					continue
				}
				switch t, _ := src["type"].(string); t {
				case "base64":
					data, _ := src["data"].(string)
					mimeType, _ := src["media_type"].(string)
					if mimeType == "" {
						mimeType = "image/png"
					}
					if data != "" {
						parts = append(parts, map[string]any{
							"inlineData": map[string]any{
								"mimeType": mimeType,
								"data":     data,
							},
						})
					}
				case "url":
					url, _ := src["url"].(string)
					mimeType, _ := src["media_type"].(string)
					if mimeType == "" {
						mimeType = "image/png"
					}
					if url != "" {
						parts = append(parts, map[string]any{
							"fileData": map[string]any{
								"fileUri":  url,
								"mimeType": mimeType,
							},
						})
					}
				}

			case "tool_use":
				if role != "assistant" {
					continue
				}
				name, _ := cm["name"].(string)
				if name == "" {
					continue
				}
				fc := map[string]any{"name": name}
				if input, ok := cm["input"]; ok && input != nil {
					fc["args"] = input
				}
				part := map[string]any{"functionCall": fc}

				signature, _ := cm["thought_signature"].(string)
				if signature == "" && thinkingSignature != "" {
					signature = thinkingSignature
				}
				if signature != "" {
					part["thoughtSignature"] = signature
				}
				parts = append(parts, part)

			case "tool_result":
				if role != "user" {
					continue
				}
				toolUseID, _ := cm["tool_use_id"].(string)
				if toolUseID == "" {
					continue
				}
				var resultText string
				var imageParts []map[string]any
				switch tc := cm["content"].(type) {
				case []any:
					var texts []string
					for _, b := range tc {
						bm, ok := b.(map[string]any)
						if !ok {
							continue
						}
						switch bt, _ := bm["type"].(string); bt {
						case "text":
							if t, _ := bm["text"].(string); t != "" {
								texts = append(texts, t)
							}
						case "image":
							src, ok := bm["source"].(map[string]any)
							if !ok {
								continue
							}
							mimeType, _ := src["media_type"].(string)
							data, _ := src["data"].(string)
							if mimeType != "" && data != "" {
								imageParts = append(imageParts, map[string]any{
									"inlineData": map[string]any{
										"mimeType": mimeType,
										"data":     data,
									},
								})
							}
						}
					}
					resultText = strings.Join(texts, "\n")
				case string:
					resultText = tc
				}
				parts = append(parts, map[string]any{
					"functionResponse": map[string]any{
						"name": toolUseID,
						"response": map[string]any{
							"result": resultText,
						},
					},
				})
				parts = append(parts, imageParts...)

			case "thinking":
				if role != "assistant" {
					continue
				}
				thinkingText, _ := cm["thinking"].(string)
				if thinkingText == "" {
					continue
				}
				// Keep when (a) message.model matches target, OR (b) no
				// message.model field is present (SDK callers).
				keep := messageModel == "" || strings.HasPrefix(messageModel, targetModel)
				if !keep {
					continue
				}
				part := map[string]any{
					"text":    thinkingText,
					"thought": true,
				}
				if sig, _ := cm["signature"].(string); sig != "" {
					part["thoughtSignature"] = sig
				}
				parts = append(parts, part)

			default:
				// Unknown block types are dropped silently; the proxy
				// logs a warning but we have no logger here.
			}
		}

		if len(parts) > 0 {
			contents = append(contents, map[string]any{
				"role":  geminiRole,
				"parts": parts,
			})
		}
	}

	return contents
}

func convertAnthropicToolsToGeminiFunctionDeclarations(ctx context.Context, tools []any) ([]map[string]any, error) {
	policy := schema.PolicyFrom(ctx)
	out := make([]map[string]any, 0, len(tools))
	for _, ti := range tools {
		tm, ok := ti.(map[string]any)
		if !ok {
			continue
		}
		name, _ := tm["name"].(string)
		if name == "" {
			continue
		}
		decl := map[string]any{"name": name}
		if desc, _ := tm["description"].(string); desc != "" {
			decl["description"] = desc
		}
		if rawSchema, ok := tm["input_schema"].(map[string]any); ok {
			normalised, _, err := schema.Normalize(ctx, rawSchema, schema.DialectGemini, policy)
			if err != nil {
				return nil, fmt.Errorf("gemini_native: tool %q: %w", name, err)
			}
			decl["parameters"] = normalised
		}
		out = append(out, decl)
	}
	return out, nil
}

// sanitizeSchemaForGemini is retained as a thin wrapper around
// schema.Normalize for backwards compatibility with existing tests; new
// call sites should use schema.Normalize(ctx, ..., DialectGemini, ...)
// directly so they pick up the active SchemaPolicy.
//
// The wrapper uses PolicyLossy so it preserves the v0.1.x silent-drop
// behaviour the original tests assert.
func sanitizeSchemaForGemini(in map[string]any) map[string]any {
	out, _, _ := schema.Normalize(context.Background(), in, schema.DialectGemini, schema.PolicyLossy)
	return out
}

// resolveGeminiThinkingLevel maps an Anthropic thinking budget to one
// of Gemini's thinkingLevel buckets. 3.1+ models support a "medium"
// level for budgets up to 20k tokens; older models only have low / high.
func resolveGeminiThinkingLevel(model string, budgetTokens float64) string {
	switch {
	case budgetTokens <= 10000:
		return "low"
	case isGemini31Model(model) && budgetTokens <= 20000:
		return "medium"
	default:
		return "high"
	}
}

func isGemini31Model(model string) bool {
	model = strings.TrimSpace(strings.ToLower(model))
	return strings.HasPrefix(model, "gemini-3.1-")
}
