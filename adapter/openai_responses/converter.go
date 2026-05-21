// Package openai_responses converts between the Anthropic Messages
// protocol and OpenAI's GPT-5 Response API. Only the streaming direction
// is fully implemented today; non-streaming calls run the stream and
// assemble the Message in the outer Client.
//
// The bulk of this file is a port of the ResponseAPIToAnthropicConverter
// found in the llm-proxy repository (commit 10c8a40), stripped of its
// internal logging dependencies.
package openai_responses

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Converter is a stateful translator that accepts Response-API JSON
// events (one at a time) and emits the Anthropic event stream needed to
// reproduce the same response.
//
// A Converter MUST be used from a single goroutine. Create a fresh
// instance per request.
type Converter struct {
	messageID    string
	currentIndex int
	hasStarted   bool

	// Thinking-block state.
	thinkingStarted    bool
	thinkingCompleted  bool
	thinkingBlockIndex int
	reasoningSignature string
	textStarted        bool

	// Tool-call state.
	toolCallStarted     bool
	lastBlockWasToolUse bool
	sawPostToolText     bool

	// GPT-5 apply_patch built-in tool state.
	applyPatchStarted bool
	applyPatchDiff    string
	applyPatchCallID  string
	applyPatchPath    string
	applyPatchOpType  string

	accumulatedUsage map[string]any
	stopReason       string
}

// NewConverter returns a Converter ready to receive events.
func NewConverter(messageID string) *Converter {
	return &Converter{
		messageID:        messageID,
		accumulatedUsage: map[string]any{},
	}
}

// ConvertEvent accepts a single Response-API event (serialised as JSON)
// and returns zero or more Anthropic-shaped events (likewise JSON).
// Unrecognised event types are silently dropped.
func (c *Converter) ConvertEvent(raw string) []string {
	var eventData map[string]any
	if err := json.Unmarshal([]byte(raw), &eventData); err != nil {
		return nil
	}
	eventType, _ := eventData["type"].(string)
	if eventType == "" {
		return nil
	}

	switch eventType {
	case "response.created":
		return c.onResponseCreated(eventData)
	case "response.reasoning_summary_text.delta",
		"response.reasoning_text.delta",
		"response.reasoning.delta":
		return c.handleReasoningDelta(eventData)
	case "response.output_text.delta":
		return c.handleTextDelta(eventData)
	case "response.function_call_arguments.delta":
		return c.handleToolCallDelta(eventData)
	case "response.output_item.added":
		return c.handleOutputItemAdded(eventData)
	case "response.function_call_arguments.done":
		return c.handleFunctionCallDone()
	case "response.apply_patch_call_operation_diff.delta":
		return c.handleApplyPatchDelta(eventData)
	case "response.output_item.done":
		return c.handleOutputItemDone(eventData)
	case "response.completed":
		return c.handleResponseCompleted(eventData)
	case "response.incomplete":
		return c.handleResponseIncomplete(eventData)
	case "response.failed":
		return c.handleResponseFailed(eventData)
	}
	return nil
}

func (c *Converter) onResponseCreated(eventData map[string]any) []string {
	if c.hasStarted {
		return nil
	}
	modelName := ""
	if response, ok := eventData["response"].(map[string]any); ok {
		if m, ok := response["model"].(string); ok {
			modelName = m
		}
	}
	c.hasStarted = true
	return []string{c.createMessageStart(modelName)}
}

func (c *Converter) handleReasoningDelta(eventData map[string]any) []string {
	var events []string
	if !c.thinkingStarted {
		// Handle OpenRouter quirk: a stray text delta before reasoning
		// must be closed first.
		if c.textStarted {
			events = append(events, c.createContentBlockStop(c.currentIndex))
			c.textStarted = false
			c.currentIndex++
		}
		events = append(events, c.createContentBlockStart("thinking"))
		c.thinkingBlockIndex = c.currentIndex
		c.thinkingStarted = true
	}
	if delta, ok := eventData["delta"].(string); ok {
		events = append(events, c.createContentBlockDelta("thinking", delta))
	}
	return events
}

func (c *Converter) handleTextDelta(eventData map[string]any) []string {
	var events []string
	if c.thinkingStarted && !c.thinkingCompleted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
		c.thinkingCompleted = true
		c.currentIndex++
	}
	if !c.textStarted {
		events = append(events, c.createContentBlockStart("text"))
		c.textStarted = true
	}
	if delta, ok := eventData["delta"].(string); ok {
		events = append(events, c.createContentBlockDelta("text", delta))
	}
	c.sawPostToolText = true
	return events
}

func (c *Converter) handleToolCallDelta(eventData map[string]any) []string {
	delta, ok := eventData["delta"].(string)
	if !ok {
		return nil
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": c.currentIndex,
		"delta": map[string]any{
			"type":         "input_json_delta",
			"partial_json": delta,
		},
	}
	data, _ := json.Marshal(event)
	return []string{string(data)}
}

func (c *Converter) handleOutputItemAdded(eventData map[string]any) []string {
	item, ok := eventData["item"].(map[string]any)
	if !ok {
		return nil
	}
	itemType, _ := item["type"].(string)
	if itemType != "function_call" && itemType != "apply_patch_call" {
		return nil
	}

	var events []string
	if c.textStarted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
		c.currentIndex++
		c.textStarted = false
	} else if c.thinkingStarted && !c.thinkingCompleted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
		c.thinkingCompleted = true
		c.currentIndex++
	}

	switch itemType {
	case "function_call":
		toolName, _ := item["name"].(string)
		toolID, _ := item["id"].(string)
		event := map[string]any{
			"type":  "content_block_start",
			"index": c.currentIndex,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    toolID,
				"name":  toolName,
				"input": map[string]any{},
			},
		}
		data, _ := json.Marshal(event)
		events = append(events, string(data))
		c.toolCallStarted = true
		c.lastBlockWasToolUse = true
		c.sawPostToolText = false
	case "apply_patch_call":
		c.applyPatchCallID, _ = item["call_id"].(string)
		c.applyPatchDiff = ""
		c.applyPatchStarted = true
		if operation, ok := item["operation"].(map[string]any); ok {
			c.applyPatchPath, _ = operation["path"].(string)
			c.applyPatchOpType, _ = operation["type"].(string)
		}
		event := map[string]any{
			"type":  "content_block_start",
			"index": c.currentIndex,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    c.applyPatchCallID,
				"name":  "apply_patch",
				"input": map[string]any{},
			},
		}
		data, _ := json.Marshal(event)
		events = append(events, string(data))
		c.lastBlockWasToolUse = true
		c.sawPostToolText = false
	}
	return events
}

func (c *Converter) handleFunctionCallDone() []string {
	if !c.toolCallStarted {
		return nil
	}
	c.toolCallStarted = false
	events := []string{c.createContentBlockStop(c.currentIndex)}
	c.currentIndex++
	return events
}

func (c *Converter) handleApplyPatchDelta(eventData map[string]any) []string {
	if c.applyPatchStarted {
		if delta, ok := eventData["delta"].(string); ok {
			c.applyPatchDiff += delta
		}
	}
	return nil
}

func (c *Converter) handleOutputItemDone(eventData map[string]any) []string {
	item, ok := eventData["item"].(map[string]any)
	if !ok {
		return nil
	}
	itemType, _ := item["type"].(string)
	if itemType != "apply_patch_call" || !c.applyPatchStarted {
		return nil
	}
	inputObj := map[string]any{
		"type": c.applyPatchOpType,
		"path": c.applyPatchPath,
		"diff": c.applyPatchDiff,
	}
	inputJSON, _ := json.Marshal(inputObj)

	deltaEvent := map[string]any{
		"type":  "content_block_delta",
		"index": c.currentIndex,
		"delta": map[string]any{
			"type":         "input_json_delta",
			"partial_json": string(inputJSON),
		},
	}
	deltaData, _ := json.Marshal(deltaEvent)
	events := []string{string(deltaData), c.createContentBlockStop(c.currentIndex)}
	c.currentIndex++
	c.applyPatchStarted = false
	c.lastBlockWasToolUse = true
	return events
}

func (c *Converter) handleResponseCompleted(eventData map[string]any) []string {
	var events []string
	if resp, ok := eventData["response"].(map[string]any); ok {
		if output, ok := resp["output"].([]any); ok {
			for _, item := range output {
				itemMap, ok := item.(map[string]any)
				if !ok {
					continue
				}
				if t, _ := itemMap["type"].(string); t == "reasoning" {
					if sig, _ := itemMap["signature"].(string); sig != "" {
						c.reasoningSignature = sig
					}
				}
			}
		}
		if usage, ok := resp["usage"].(map[string]any); ok {
			c.accumulatedUsage = usage
		}
	}

	if c.reasoningSignature != "" && c.thinkingStarted {
		sigEvent := map[string]any{
			"type":  "content_block_delta",
			"index": c.thinkingBlockIndex,
			"delta": map[string]any{
				"type":      "signature_delta",
				"signature": c.reasoningSignature,
			},
		}
		sigData, _ := json.Marshal(sigEvent)
		events = append(events, string(sigData))
	}

	if c.textStarted || c.toolCallStarted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	} else if c.thinkingStarted && !c.thinkingCompleted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	}

	if c.lastBlockWasToolUse && !c.sawPostToolText {
		c.stopReason = "tool_use"
	} else {
		c.stopReason = "end_turn"
	}

	events = append(events, c.createMessageDelta())
	events = append(events, c.createMessageStop())
	return events
}

func (c *Converter) handleResponseIncomplete(eventData map[string]any) []string {
	var events []string
	if c.textStarted || c.toolCallStarted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	} else if c.thinkingStarted && !c.thinkingCompleted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	}

	if resp, ok := eventData["response"].(map[string]any); ok {
		if usage, ok := resp["usage"].(map[string]any); ok {
			c.accumulatedUsage = usage
		}
		if !(c.lastBlockWasToolUse && !c.sawPostToolText) {
			if inc, ok := resp["incomplete_details"].(map[string]any); ok {
				if reason, ok := inc["reason"].(string); ok {
					c.stopReason = mapIncompleteReason(reason)
				}
			}
		}
	}
	if c.lastBlockWasToolUse && !c.sawPostToolText {
		c.stopReason = "tool_use"
	}

	events = append(events, c.createMessageDelta())
	events = append(events, c.createMessageStop())
	return events
}

func (c *Converter) handleResponseFailed(eventData map[string]any) []string {
	errorMessage := "unknown error"
	if resp, ok := eventData["response"].(map[string]any); ok {
		if errObj, ok := resp["error"].(map[string]any); ok {
			if msg, ok := errObj["message"].(string); ok && msg != "" {
				errorMessage = msg
			}
		}
	}
	payload := map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "upstream_error", "message": errorMessage},
	}
	data, _ := json.Marshal(payload)
	return []string{string(data)}
}

func mapIncompleteReason(reason string) string {
	switch strings.ToLower(reason) {
	case "max_output_tokens":
		return "max_tokens"
	case "content_filter":
		return "refusal"
	default:
		return reason
	}
}

// ---- Event builders ----------------------------------------------------------

func (c *Converter) createMessageStart(modelName string) string {
	event := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            c.messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         modelName,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	}
	data, _ := json.Marshal(event)
	return string(data)
}

func (c *Converter) createContentBlockStart(blockType string) string {
	block := map[string]any{}
	switch blockType {
	case "thinking":
		block["type"] = "thinking"
		block["thinking"] = ""
		block["signature"] = ""
	case "text":
		block["type"] = "text"
		block["text"] = ""
	}
	event := map[string]any{
		"type":          "content_block_start",
		"index":         c.currentIndex,
		"content_block": block,
	}
	data, _ := json.Marshal(event)
	return string(data)
}

func (c *Converter) createContentBlockDelta(blockType, delta string) string {
	d := map[string]any{}
	switch blockType {
	case "thinking":
		d["type"] = "thinking_delta"
		d["thinking"] = delta
	case "text":
		d["type"] = "text_delta"
		d["text"] = delta
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": c.currentIndex,
		"delta": d,
	}
	data, _ := json.Marshal(event)
	return string(data)
}

func (c *Converter) createContentBlockStop(index int) string {
	event := map[string]any{"type": "content_block_stop", "index": index}
	data, _ := json.Marshal(event)
	return string(data)
}

func (c *Converter) createMessageDelta() string {
	delta := map[string]any{
		"stop_reason":   strings.ToLower(c.stopReason),
		"stop_sequence": nil,
	}
	event := map[string]any{"type": "message_delta", "delta": delta}
	if u := c.convertUsage(); u != nil {
		event["usage"] = u
	}
	data, _ := json.Marshal(event)
	return string(data)
}

func (c *Converter) createMessageStop() string {
	event := map[string]any{"type": "message_stop"}
	if u := c.convertUsage(); u != nil {
		event["usage"] = u
	}
	data, _ := json.Marshal(event)
	return string(data)
}

// convertUsage maps the GPT-5 usage block to Anthropic field names.
// Returns nil when no usage has been captured.
func (c *Converter) convertUsage() map[string]any {
	if len(c.accumulatedUsage) == 0 {
		return nil
	}
	toFloat := func(v any) float64 {
		switch t := v.(type) {
		case float64:
			return t
		case float32:
			return float64(t)
		case int:
			return float64(t)
		case int64:
			return float64(t)
		}
		return 0
	}
	out := map[string]any{}
	var cached float64
	if details, ok := c.accumulatedUsage["input_tokens_details"].(map[string]any); ok {
		if c, ok := details["cached_tokens"]; ok {
			cached = toFloat(c)
			out["cache_read_input_tokens"] = int(cached)
		}
	}
	if v, ok := c.accumulatedUsage["input_tokens"]; ok {
		out["input_tokens"] = int(toFloat(v) - cached)
	}
	if v, ok := c.accumulatedUsage["output_tokens"]; ok {
		out["output_tokens"] = int(toFloat(v))
	}
	if details, ok := c.accumulatedUsage["output_tokens_details"].(map[string]any); ok {
		if v, ok := details["reasoning_tokens"]; ok {
			out["reasoning_tokens"] = int(toFloat(v))
		}
	}
	if _, ok := out["cache_creation_input_tokens"]; !ok {
		out["cache_creation_input_tokens"] = 0
	}
	return out
}

// AccumulatedUsage returns the last usage snapshot observed from the
// upstream response, or nil if none was seen.
func (c *Converter) AccumulatedUsage() map[string]any { return c.accumulatedUsage }

// StopReason returns the Anthropic-mapped stop reason, once known.
func (c *Converter) StopReason() string { return c.stopReason }

// ensure fmt is always imported to avoid "unused" errors when the file
// is edited; used by future error formatting helpers.
var _ = fmt.Sprint
