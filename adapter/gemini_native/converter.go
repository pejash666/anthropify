// Package gemini_native: streaming conversion from Google's Vertex/AI
// Studio streamGenerateContent SSE protocol to Anthropic Messages
// events.
//
// The bulk of this file is a port of the GeminiToAnthropicConverter
// found in the llm-proxy repository (service/llm/converter.go), stripped
// of its server-side dependencies (zap logging, request-scoped context
// metadata, uuid generation). Behaviour for the documented event shapes
// is intended to match that reference; differences are limited to:
//
//   - tool_use ids are generated with crypto/rand instead of google/uuid
//     to avoid pulling a third-party dependency into hybridstream;
//   - log lines are dropped; non-fatal parse problems are silently
//     ignored, matching the openai_responses converter style;
//   - retry_prompt / stop_msg fields (which the proxy adds for its own
//     client) are NOT emitted, since the canonical Anthropic schema
//     does not include them.
package gemini_native

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// Converter is a stateful translator that accepts Gemini
// streamGenerateContent JSON events (one at a time) and emits the
// Anthropic event stream needed to reproduce the same response.
//
// A Converter MUST be used from a single goroutine. Create a fresh
// instance per request.
type Converter struct {
	model     string
	messageID string

	currentIndex     int
	textStarted      bool
	thinkingStarted  bool
	signatureSent    bool
	toolCallStarted  bool
	accumulatedUsage map[string]any

	lastBlockWasToolUse bool
	sawPostToolText     bool
	stopReason          string
	finished            bool

	// hasStarted records whether message_start has been emitted so the
	// converter is idempotent across reconnects/retries that replay the
	// first event.
	hasStarted bool
}

// NewConverter returns a Converter ready to accept events. model is
// echoed back inside the message_start event (Anthropic clients use it
// to display the assistant model badge).
func NewConverter(model, messageID string) *Converter {
	return &Converter{
		model:            model,
		messageID:        messageID,
		accumulatedUsage: map[string]any{},
	}
}

// ConvertEvent accepts a single Gemini SSE data payload (the JSON object
// after the "data:" prefix) and returns zero or more Anthropic-shaped
// events (likewise serialised as JSON strings). Unrecognised events,
// candidate-less frames, and parse errors are silently dropped.
func (c *Converter) ConvertEvent(eventData string) []string {
	var geminiEvent map[string]any
	if err := json.Unmarshal([]byte(eventData), &geminiEvent); err != nil {
		return nil
	}

	var events []string

	// First event seen: emit message_start. The reference uses a
	// composite condition; we collapse it to a single hasStarted flag
	// because re-checking the running indices is brittle.
	if !c.hasStarted {
		events = append(events, c.createMessageStart())
		c.hasStarted = true
	}

	candidates, ok := geminiEvent["candidates"].([]any)
	if !ok || len(candidates) == 0 {
		// usageMetadata frames may arrive without candidates at the very
		// end. Capture them so the final message_stop carries usage.
		if usage, ok := geminiEvent["usageMetadata"].(map[string]any); ok {
			c.accumulatedUsage = usage
		}
		return events
	}

	candidate, ok := candidates[0].(map[string]any)
	if !ok {
		return events
	}

	// finishReason and usageMetadata are checked BEFORE parts because
	// the terminal Gemini event may carry only finishReason / usage
	// without any content.parts.
	if finishReason, ok := candidate["finishReason"].(string); ok && finishReason != "" {
		c.stopReason = mapFinishReason(finishReason)
	}
	if usage, ok := geminiEvent["usageMetadata"].(map[string]any); ok {
		c.accumulatedUsage = usage
	}

	content, ok := candidate["content"].(map[string]any)
	if !ok {
		return events
	}
	parts, ok := content["parts"].([]any)
	if !ok || len(parts) == 0 {
		return events
	}

	for _, partInterface := range parts {
		part, ok := partInterface.(map[string]any)
		if !ok {
			continue
		}

		// 1. Text part. Gemini distinguishes thinking vs final text via
		//    the "thought": true flag (only set when includeThoughts is
		//    enabled in the request).
		if text, ok := part["text"].(string); ok && text != "" {
			if isThought, _ := part["thought"].(bool); isThought {
				// Switching from text -> thinking: close the open text
				// block first.
				if c.textStarted {
					events = append(events, c.createContentBlockStop(c.currentIndex))
					c.currentIndex++
					c.textStarted = false
				}
				if !c.thinkingStarted {
					events = append(events, c.createContentBlockStart("thinking"))
					c.thinkingStarted = true
					c.signatureSent = false
				}
				events = append(events, c.createContentBlockDelta("thinking", text))
				if !c.signatureSent {
					if sig, ok := part["thoughtSignature"].(string); ok && sig != "" {
						events = append(events, c.createSignatureDelta(sig))
						c.signatureSent = true
					}
				}
			} else {
				// Plain text. If we were inside a thinking block, flush
				// the signature_delta (Gemini sometimes attaches the
				// thoughtSignature to the first non-thought text part)
				// and close the thinking block.
				if c.thinkingStarted {
					if !c.signatureSent {
						if sig, ok := part["thoughtSignature"].(string); ok && sig != "" {
							events = append(events, c.createSignatureDelta(sig))
							c.signatureSent = true
						}
					}
					events = append(events, c.createContentBlockStop(c.currentIndex))
					c.currentIndex++
					c.thinkingStarted = false
				}
				if !c.textStarted {
					events = append(events, c.createContentBlockStart("text"))
					c.textStarted = true
				}
				events = append(events, c.createContentBlockDelta("text", text))
				c.sawPostToolText = true
			}
		}

		// 2. functionCall part -> tool_use block. Gemini's native
		//    functionCall has no id; we synthesise one with random
		//    bytes so parallel tool calls remain unique.
		if functionCall, ok := part["functionCall"].(map[string]any); ok {
			toolName, _ := functionCall["name"].(string)
			toolArgs := functionCall["args"]
			thoughtSignature, _ := part["thoughtSignature"].(string)

			// Close any currently open block.
			if c.textStarted {
				events = append(events, c.createContentBlockStop(c.currentIndex))
				c.currentIndex++
				c.textStarted = false
			} else if c.thinkingStarted {
				// thinking + tool_use: NOT signature_delta; the signature
				// rides inside the tool_use content block instead.
				events = append(events, c.createContentBlockStop(c.currentIndex))
				c.currentIndex++
				c.thinkingStarted = false
			}

			toolID := "gemini_" + toolName + "_" + shortRandomHex(6)
			events = append(events, c.createToolUseWithArgs(toolName, toolID, toolArgs, thoughtSignature))
			// Tool use blocks are atomic: emit stop immediately so the
			// reader can dispatch the call without waiting.
			events = append(events, c.createContentBlockStop(c.currentIndex))
			c.currentIndex++
			c.toolCallStarted = false
			c.lastBlockWasToolUse = true
			c.sawPostToolText = false
		}

		// 3. functionResponse parts only appear in user messages on the
		//    request side; ignore them here.
	}

	return events
}

// OnStreamDone flushes any open block and emits the terminal
// message_delta + message_stop pair. Callers MUST invoke it exactly
// once after the upstream stream closes (regardless of whether the
// final Gemini event carried finishReason).
func (c *Converter) OnStreamDone() []string {
	if c.finished {
		return nil
	}
	c.finished = true

	var events []string

	// Close whichever block is still open. tool_use blocks are closed
	// at creation time, so the toolCallStarted branch is purely defensive.
	if c.textStarted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	} else if c.thinkingStarted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	} else if c.toolCallStarted {
		events = append(events, c.createContentBlockStop(c.currentIndex))
	}

	events = append(events, c.createMessageDelta())
	events = append(events, c.createMessageStop())
	return events
}

// AccumulatedUsage returns the most recent usageMetadata payload seen
// from the upstream, or nil if none was observed.
func (c *Converter) AccumulatedUsage() map[string]any { return c.accumulatedUsage }

// StopReason returns the Anthropic-mapped stop reason, once known.
func (c *Converter) StopReason() string { return c.stopReason }

// ---- Event builders ---------------------------------------------------------

func (c *Converter) createMessageStart() string {
	event := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            c.messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         c.model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createContentBlockStart(blockType string) string {
	block := map[string]any{"type": blockType}
	switch blockType {
	case "text":
		block["text"] = ""
	case "thinking":
		block["thinking"] = ""
		block["signature"] = ""
	}
	event := map[string]any{
		"type":          "content_block_start",
		"index":         c.currentIndex,
		"content_block": block,
	}
	b, _ := json.Marshal(event)
	return string(b)
}

// createToolUseWithArgs emits a content_block_start carrying the full
// args payload (Gemini delivers function arguments as a single object,
// unlike OpenAI which streams them as JSON deltas).
func (c *Converter) createToolUseWithArgs(name, id string, args any, thoughtSignature string) string {
	var input any
	if args == nil {
		input = map[string]any{}
	} else {
		input = args
	}
	contentBlock := map[string]any{
		"type":  "tool_use",
		"id":    id,
		"name":  name,
		"input": input,
	}
	if thoughtSignature != "" {
		contentBlock["thought_signature"] = thoughtSignature
	}
	event := map[string]any{
		"type":          "content_block_start",
		"index":         c.currentIndex,
		"content_block": contentBlock,
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createContentBlockDelta(blockType, delta string) string {
	d := map[string]any{}
	switch blockType {
	case "text":
		d["type"] = "text_delta"
		d["text"] = delta
	case "thinking":
		d["type"] = "thinking_delta"
		d["thinking"] = delta
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": c.currentIndex,
		"delta": d,
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createSignatureDelta(signature string) string {
	event := map[string]any{
		"type":  "content_block_delta",
		"index": c.currentIndex,
		"delta": map[string]any{
			"type":      "signature_delta",
			"signature": signature,
		},
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createContentBlockStop(index int) string {
	event := map[string]any{"type": "content_block_stop", "index": index}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createMessageDelta() string {
	// Gemini's finishReason does not distinguish tool_use; if the last
	// emitted block was a tool_use AND no plain text followed, override
	// the stop reason.
	stopReason := c.stopReason
	if c.lastBlockWasToolUse && !c.sawPostToolText {
		stopReason = "tool_use"
	}
	stopReason = strings.ToLower(stopReason)

	event := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"output_tokens": 0,
		},
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createMessageStop() string {
	event := map[string]any{"type": "message_stop"}
	if u := c.convertUsage(); u != nil {
		event["usage"] = u
	}
	b, _ := json.Marshal(event)
	return string(b)
}

// convertUsage maps Gemini's usageMetadata block to Anthropic field
// names. Returns nil when no usage has been captured.
func (c *Converter) convertUsage() map[string]any {
	if len(c.accumulatedUsage) == 0 {
		return nil
	}
	toInt := func(v any) int {
		switch t := v.(type) {
		case float64:
			return int(t)
		case float32:
			return int(t)
		case int:
			return t
		case int64:
			return int(t)
		}
		return 0
	}

	prompt := toInt(c.accumulatedUsage["promptTokenCount"])
	candidates := toInt(c.accumulatedUsage["candidatesTokenCount"])
	cached := toInt(c.accumulatedUsage["cachedContentTokenCount"])
	thoughts := toInt(c.accumulatedUsage["thoughtsTokenCount"])

	out := map[string]any{
		"input_tokens":                prompt - cached,
		"output_tokens":               candidates,
		"cache_read_input_tokens":     cached,
		"cache_creation_input_tokens": 0,
	}
	if thoughts > 0 {
		out["thoughts_tokens"] = thoughts
	}
	return out
}

// mapFinishReason maps Gemini's finishReason to Anthropic's stop_reason.
//
// Mapping table:
//
//   - STOP                                                    -> end_turn
//   - MAX_TOKENS                                              -> max_tokens
//   - SAFETY / BLOCKLIST / PROHIBITED_CONTENT / SPII /
//     RECITATION / LANGUAGE                                   -> refusal
//   - everything else (OTHER, MALFORMED_FUNCTION_CALL, ...)   -> passthrough
//
// The tool_use override is applied later in createMessageDelta when the
// last emitted block was a tool_use without trailing text.
func mapFinishReason(geminiReason string) string {
	switch strings.ToUpper(geminiReason) {
	case "STOP":
		return "end_turn"
	case "MAX_TOKENS":
		return "max_tokens"
	case "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "LANGUAGE":
		return "refusal"
	default:
		return geminiReason
	}
}

// shortRandomHex returns a hex-encoded random string of the given byte
// length (so output length is 2*n). It is used to mint synthetic
// tool_use ids; collisions within a single request are astronomically
// unlikely at n=6.
func shortRandomHex(n int) string {
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		return strings.Repeat("0", n*2)
	}
	return hex.EncodeToString(buf)
}
