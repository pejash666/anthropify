package chat_completions

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// BuildRequest translates an Anthropic MessageNewParams into an OpenAI
// Chat Completions request body. The boolean stream argument selects
// streaming vs non-streaming mode (the adapter sets it to true; tests
// may flip it for inspection).
//
// The translation follows ConvertAnthropicToChatCompletions in
// service/llm/converter.go (line 883 onwards), with two simplifications:
//
//  1. The proxy logs through zap on several branches; those calls are
//     dropped, only the behaviour they observe is preserved.
//  2. Self-deployed-model lookups (GetSelfDeployedModelsConfigService)
//     are dropped: anthropify does not have a model registry, so the
//     decision to emit reasoning_content is made strictly on the model
//     string passed in.
//
// We round-trip through map[string]any rather than building OpenAI
// SDK structs so that we do not need to take a dependency on
// openai-go / sashabaranov/go-openai.
func BuildRequest(req anthropicsdk.MessageNewParams, stream bool) ([]byte, error) {
	// Convert MessageNewParams to map[string]any via JSON; the SDK
	// types are not introspectable enough to walk directly.
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("anthropify/chat_completions: marshal MessageNewParams: %w", err)
	}
	var anth map[string]any
	if err := json.Unmarshal(raw, &anth); err != nil {
		return nil, fmt.Errorf("anthropify/chat_completions: unmarshal MessageNewParams: %w", err)
	}

	out := map[string]any{}

	// Model: passthrough.
	model := ""
	if m, ok := anth["model"].(string); ok {
		out["model"] = m
		model = m
	}

	// System content.
	systemContent := convertSystemContent(anth["system"])

	// Messages.
	var messages []map[string]any
	if ms, ok := anth["messages"].([]any); ok {
		messages = convertAnthropicMessages(ms, systemContent, model)
	}
	out["messages"] = messages

	// kimi-k2 family special: if history assistant carries tool_calls
	// but no reasoning_content, thinking must be disabled or Kimi will
	// reject the request.
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

	// Tools.
	if tools, ok := anth["tools"].([]any); ok {
		outTools := convertAnthropicTools(tools)
		if len(outTools) > 0 {
			out["tools"] = outTools
		}
	}

	// Tool choice.
	if toolChoice := convertToolChoice(anth["tool_choice"]); toolChoice != nil {
		out["tool_choice"] = toolChoice
	}

	// Sampling: kimi-k2 family cannot accept temperature overrides.
	if !isKimiK2Family(model) {
		if v, ok := anth["temperature"]; ok {
			out["temperature"] = v
		}
	}

	// Thinking config.
	isGLMModel := strings.HasPrefix(strings.ToLower(model), "glm-")
	isDeepSeekPassthroughThinking := isDeepSeekPassthrough(model)

	switch {
	case needDisableThinking && isKimiK2Family(model):
		out["thinking"] = map[string]any{"type": "disabled"}
	case anth["thinking"] != nil:
		if thinkingMap, ok := anth["thinking"].(map[string]any); ok {
			if isGLMModel || isDeepSeekPassthroughThinking {
				out["thinking"] = thinkingMap
			} else if enabled, ok := thinkingMap["type"].(string); ok && enabled == "enabled" {
				// Map budget to reasoning_effort.
				budgetValue := thinkingMap["budget_tokens"]
				if budgetValue == nil {
					budgetValue = thinkingMap["thinking_budget"]
				}
				var budget int64
				ok := false
				switch v := budgetValue.(type) {
				case float64:
					budget = int64(v)
					ok = true
				case int64:
					budget = v
					ok = true
				case int:
					budget = int64(v)
					ok = true
				}
				if ok {
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
	case isGLMModel:
		// GLM defaults to thinking enabled; close it explicitly when
		// the upstream did not opt in.
		out["thinking"] = map[string]any{"type": "disabled"}
	}

	if v, ok := anth["max_tokens"]; ok {
		out["max_completion_tokens"] = v
	}

	out["stream"] = stream

	return json.Marshal(out)
}

// convertSystemContent mirrors service/llm/converter.go:409.
func convertSystemContent(sys any) any {
	if sys == nil {
		return nil
	}
	switch s := sys.(type) {
	case string:
		return s
	case []any:
		systemParts := make([]map[string]any, 0)
		for _, it := range s {
			if m, ok := it.(map[string]any); ok {
				if t, ok := m["text"].(string); ok && t != "" {
					systemParts = append(systemParts, map[string]any{
						"type": "text",
						"text": t,
					})
				}
			}
		}
		if len(systemParts) == 1 {
			return systemParts[0]["text"].(string)
		} else if len(systemParts) > 1 {
			return systemParts
		}
	}
	return nil
}

// isKimiK2Family flags the Kimi k2.5 / k2.6 models that share Moonshot's
// "no temperature override" + "reasoning_content required with tool_calls" rules.
//
// Mirrors service/llm/converter.go:445. We hard-code the model
// identifiers because anthropify has no shared constants package.
//
// SoT canonical names (library/constants/llm.go: KimiK25, KimiK26) only
// use the dot form ("kimi-k2.5", "kimi-k2.6"). The hyphenated aliases
// ("kimi-k2-5", "kimi-k2-6") are a anthropify extension to tolerate
// callers that normalise dots to hyphens; do not remove them without
// auditing downstream callers.
func isKimiK2Family(model string) bool {
	switch strings.ToLower(model) {
	case "kimi-k2-5", "kimi-k2.5", "kimi-k2-6", "kimi-k2.6":
		return true
	}
	return false
}

// isReasoningModel: models that support and require reasoning_content
// when sending historical assistant messages. Mirrors the SoT list at
// service/llm/converter.go:450.
//
// SoT canonical names (library/constants/llm.go) only use the dot form
// for GLM ("glm-4.6", "glm-4.7", "glm-5.1"). The hyphenated GLM aliases
// ("glm-4-6", "glm-4-7", "glm-5-1") are a anthropify extension and
// are kept on purpose to tolerate hyphen-normalised model names.
func isReasoningModel(model string) bool {
	ml := strings.ToLower(model)
	switch ml {
	case "kimi-k2-thinking", "kimi-k2-thinking-turbo":
		return true
	case "deepseek-reasoner", "deepseek-v4-pro", "deepseek-v4-flash":
		return true
	case "glm-4.6", "glm-4-6", "glm-4.7", "glm-4-7", "glm-5", "glm-5-turbo", "glm-5.1", "glm-5-1":
		return true
	}
	if isKimiK2Family(model) {
		return true
	}
	return false
}

// isDeepSeekPassthrough flags DeepSeek variants that need the thinking
// block forwarded verbatim instead of mapped to reasoning_effort.
//
// SoT canonical name for V3.2 (library/constants/llm.go:DeepSeekV32) is
// "deepseek/deepseek-v3.2" - the slash form. The bare "deepseek-v3.2"
// and "deepseek-v3-2" aliases are anthropify extensions covering
// callers that strip the provider prefix or normalise the dot.
func isDeepSeekPassthrough(model string) bool {
	ml := strings.ToLower(model)
	switch ml {
	case "deepseek-v3.2", "deepseek-v3-2", "deepseek/deepseek-v3.2", "deepseek-v4-pro", "deepseek-v4-flash":
		return true
	}
	return false
}

// convertAnthropicMessages ports service/llm/converter.go:466. The
// ctx-bound logging is dropped, otherwise the behaviour is preserved.
func convertAnthropicMessages(anthropicMessages []any, systemContent any, model string) []map[string]any {
	messages := make([]map[string]any, 0)
	supportsReasoningContent := isReasoningModel(model)

	for _, mi := range anthropicMessages {
		mm, ok := mi.(map[string]any)
		if !ok {
			continue
		}
		role, _ := mm["role"].(string)
		contents, _ := mm["content"].([]any)

		parts := make([]map[string]any, 0)
		assistantToolCalls := make([]map[string]any, 0)
		thinkingContents := make([]string, 0)
		toolResultMessages := make([]map[string]any, 0)
		toolResultImageMessages := make([]map[string]any, 0)

		for _, ci := range contents {
			cm, ok := ci.(map[string]any)
			if !ok {
				continue
			}
			t, _ := cm["type"].(string)
			switch t {
			case "text":
				if v, _ := cm["text"].(string); v != "" {
					parts = append(parts, map[string]any{"type": "text", "text": v})
				}
			case "thinking":
				if thinkingText, ok := cm["thinking"].(string); ok && thinkingText != "" {
					thinkingContents = append(thinkingContents, thinkingText)
				}
			case "image":
				if src, ok := cm["source"].(map[string]any); ok {
					if media, _ := src["media_type"].(string); media != "" {
						if data, _ := src["data"].(string); data != "" {
							dataURL := fmt.Sprintf("data:%s;base64,%s", media, data)
							parts = append(parts, map[string]any{
								"type":      "image_url",
								"image_url": map[string]any{"url": dataURL},
							})
							break
						}
					}
					if url, _ := src["url"].(string); url != "" {
						parts = append(parts, map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": url},
						})
					}
				}
			case "tool_use":
				if role == "assistant" {
					name, _ := cm["name"].(string)
					var argsStr string
					if input, ok := cm["input"].(map[string]any); ok {
						if b, err := json.Marshal(input); err == nil {
							argsStr = string(b)
						}
					}
					toolCall := map[string]any{
						"type":     "function",
						"function": map[string]any{"name": name, "arguments": argsStr},
					}
					if idStr, _ := cm["id"].(string); idStr != "" {
						toolCall["id"] = idStr
					}
					assistantToolCalls = append(assistantToolCalls, toolCall)
				}
			case "tool_result":
				if role == "user" {
					toolCallID, _ := cm["tool_use_id"].(string)
					var outputStr string
					if rawArr, ok := cm["content"].([]any); ok {
						if b, err := json.Marshal(rawArr); err == nil {
							outputStr = string(b)
						}
						imgParts := make([]map[string]any, 0)
						for _, it := range rawArr {
							if m, ok := it.(map[string]any); ok {
								if tt, _ := m["type"].(string); tt == "image" {
									if src, ok := m["source"].(map[string]any); ok {
										media := fmt.Sprint(src["media_type"])
										data := fmt.Sprint(src["data"])
										if media != "" && data != "" {
											dataURL := fmt.Sprintf("data:%s;base64,%s", media, data)
											imgParts = append(imgParts, map[string]any{
												"type":      "image_url",
												"image_url": map[string]any{"url": dataURL},
											})
										}
									}
								}
							}
						}
						if len(imgParts) > 0 {
							toolResultImageMessages = append(toolResultImageMessages, map[string]any{"role": "user", "content": imgParts})
						}
					} else if s, ok := cm["content"].(string); ok {
						outputStr = s
					}
					toolMsg := map[string]any{"role": "tool", "content": outputStr}
					if toolCallID != "" {
						toolMsg["tool_call_id"] = toolCallID
					}
					toolResultMessages = append(toolResultMessages, toolMsg)
				}
			}
		}

		switch role {
		case "assistant":
			msg := map[string]any{"role": "assistant"}
			if len(assistantToolCalls) > 0 {
				msg["tool_calls"] = assistantToolCalls
			}
			if len(parts) > 0 {
				if len(parts) == 1 {
					if textPart, ok := parts[0]["text"].(string); ok && parts[0]["type"] == "text" {
						msg["content"] = textPart
					} else {
						msg["content"] = parts
					}
				} else {
					msg["content"] = parts
				}
			}
			if supportsReasoningContent && len(thinkingContents) > 0 {
				reasoningContent := ""
				for i, thinking := range thinkingContents {
					if i > 0 {
						reasoningContent += "\n\n"
					}
					reasoningContent += thinking
				}
				msg["reasoning_content"] = reasoningContent
			}
			if msg["content"] != nil || len(assistantToolCalls) > 0 || msg["reasoning_content"] != nil {
				messages = append(messages, msg)
			}
		case "user":
			messages = append(messages, toolResultMessages...)
			messages = append(messages, toolResultImageMessages...)
			if len(parts) > 0 {
				if len(parts) == 1 {
					if textPart, ok := parts[0]["text"].(string); ok && parts[0]["type"] == "text" {
						messages = append(messages, map[string]any{"role": "user", "content": textPart})
					} else {
						messages = append(messages, map[string]any{"role": "user", "content": parts})
					}
				} else {
					messages = append(messages, map[string]any{"role": "user", "content": parts})
				}
			}
		default:
			if len(parts) > 0 {
				if len(parts) == 1 {
					if textPart, ok := parts[0]["text"].(string); ok && parts[0]["type"] == "text" {
						messages = append(messages, map[string]any{"role": role, "content": textPart})
					} else {
						messages = append(messages, map[string]any{"role": role, "content": parts})
					}
				} else {
					messages = append(messages, map[string]any{"role": role, "content": parts})
				}
			}
		}
	}

	if systemContent != nil {
		messages = append([]map[string]any{{"role": "system", "content": systemContent}}, messages...)
	}

	return messages
}

// convertAnthropicTools ports service/llm/converter.go:697.
func convertAnthropicTools(anthropicTools []any) []any {
	outTools := make([]any, 0, len(anthropicTools))
	for _, ti := range anthropicTools {
		if tm, ok := ti.(map[string]any); ok {
			fn := map[string]any{}
			if v, ok := tm["name"]; ok {
				fn["name"] = v
			}
			if v, ok := tm["description"]; ok {
				fn["description"] = v
			}

			var parameters map[string]any
			if v, ok := tm["input_schema"]; ok && v != nil {
				if schema, ok := v.(map[string]any); ok && schema != nil {
					parameters = cleanSchemaForAzure(schema)
				}
			}
			if parameters == nil {
				parameters = map[string]any{
					"type":       "object",
					"properties": map[string]any{},
				}
			}
			fn["parameters"] = parameters
			outTools = append(outTools, map[string]any{"type": "function", "function": fn})
		}
	}
	return outTools
}

// cleanSchemaForAzure recursively normalises a JSON schema so it is
// accepted by Azure / OpenAI-compatible providers. Mirrors the SoT at
// service/llm/converter.go:736.
func cleanSchemaForAzure(schema map[string]any) map[string]any {
	cleaned := make(map[string]any)
	for key, value := range schema {
		switch key {
		case "type":
			if typeValue, ok := value.(string); ok {
				cleaned[key] = strings.ToLower(typeValue)
			} else {
				cleaned[key] = value
			}
		case "additionalProperties":
			if value == nil {
				cleaned[key] = false
			} else {
				cleaned[key] = value
			}
		case "$schema":
			// drop
			continue
		case "properties":
			if props, ok := value.(map[string]any); ok {
				names := make([]string, 0, len(props))
				for n := range props {
					names = append(names, n)
				}
				sort.Strings(names)
				cleanedProps := make(map[string]any)
				for _, n := range names {
					if sm, ok := props[n].(map[string]any); ok {
						cleanedProps[n] = cleanSchemaForAzure(sm)
					} else {
						cleanedProps[n] = props[n]
					}
				}
				cleaned[key] = cleanedProps
			} else {
				cleaned[key] = value
			}
		case "items":
			if itemSchema, ok := value.(map[string]any); ok {
				cleaned[key] = cleanSchemaForAzure(itemSchema)
			} else {
				cleaned[key] = value
			}
		case "required":
			if reqArray, ok := value.([]any); ok {
				reqStrings := make([]string, 0, len(reqArray))
				for _, ri := range reqArray {
					if name, ok := ri.(string); ok {
						reqStrings = append(reqStrings, name)
					}
				}
				if len(reqStrings) > 0 {
					sort.Strings(reqStrings)
					cleaned[key] = reqStrings
				}
			} else if reqStrings, ok := value.([]string); ok {
				sorted := make([]string, len(reqStrings))
				copy(sorted, reqStrings)
				sort.Strings(sorted)
				cleaned[key] = sorted
			} else {
				cleaned[key] = value
			}
		default:
			cleaned[key] = value
		}
	}
	return cleaned
}

// convertToolChoice ports service/llm/converter.go:853.
func convertToolChoice(anthropicToolChoice any) any {
	if anthropicToolChoice == nil {
		return nil
	}
	switch tc := anthropicToolChoice.(type) {
	case string:
		return tc
	case map[string]any:
		if t, ok := tc["type"].(string); ok {
			switch strings.ToLower(t) {
			case "auto":
				return nil
			case "any":
				return "required"
			case "tool":
				if name, ok := tc["name"].(string); ok && name != "" {
					return map[string]any{
						"type":     "function",
						"function": map[string]any{"name": name},
					}
				}
			}
		}
	}
	return nil
}
