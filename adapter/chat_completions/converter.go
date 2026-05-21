package chat_completions

import (
	"encoding/json"
	"sort"
	"strings"
)

// Converter translates an upstream OpenAI Chat Completions SSE stream
// into Anthropic Messages events.
//
// This is a port of the ChatCompletionsToAnthropicConverter state
// machine that ships in the llm-proxy repository at
// service/llm/converter.go (see lines 1903–2457). The port intentionally
// drops the following server-side concerns:
//
//   - SetSessionID / [Kimi DEBUG] structured logging: these are
//     observability hooks specific to the proxy's request lifecycle
//     and have no place inside an SDK adapter.
//   - retry_prompt / stop_msg fields on message_delta: those are
//     product-UX strings emitted by the proxy, not part of the
//     Anthropic wire protocol.
//
// Everything else — including the model behaviour quirks that callers
// actually encounter (reasoning_content support for Kimi/DeepSeek/GLM,
// Gemini-via-Chat-Completions thinking signaled by extra_content, and
// the tool_call ordering rules) — is preserved verbatim.
//
// Inputs to ConvertEvent are the raw JSON bodies of OpenAI SSE `data:`
// lines (the `data: ` prefix and `[DONE]` sentinel are stripped by the
// caller). Outputs are JSON strings carrying Anthropic events; callers
// usually wrap each one in adapter.RawEvent before delivering it to
// the client.
type Converter struct {
	messageID string
	model     string

	currentIndex        int
	hasStarted          bool
	thinkingStarted     bool
	thinkingCompleted   bool
	textStarted         bool
	lastBlockWasToolUse bool
	sawPostToolText     bool
	accumulatedUsage    map[string]any
	stopReason          string

	// Multiple tool calls: state keyed by the OpenAI tool_calls index.
	toolCallStates    map[int]*toolCallPending
	activeToolOAIndex int  // currently open OpenAI tool_call index, -1 for none
	anyToolStarted    bool // any tool block was ever opened
	finished          bool // OnStreamDone has already run
}

// toolCallPending tracks one in-flight OpenAI tool_call between deltas.
type toolCallPending struct {
	name       string
	id         string
	args       []string // arguments fragments buffered while id is unknown
	blockIndex int      // Anthropic content-block index once started
	started    bool
}

// NewConverter constructs a converter for the given model. The model
// name is echoed back inside message_start so that Anthropic clients
// can identify the responding model.
func NewConverter(model, messageID string) *Converter {
	if messageID == "" {
		messageID = genMessageID()
	}
	return &Converter{
		messageID:         messageID,
		model:             model,
		accumulatedUsage:  make(map[string]any),
		toolCallStates:    make(map[int]*toolCallPending),
		activeToolOAIndex: -1,
	}
}

// ConvertEvent consumes one upstream Chat Completions JSON payload and
// returns the Anthropic events triggered by it.
func (c *Converter) ConvertEvent(openaiEvent string) []string {
	var events []string
	var m map[string]any
	if err := json.Unmarshal([]byte(openaiEvent), &m); err != nil {
		// Mirror the SoT: silently skip malformed frames so that one
		// glitchy line doesn't take down the whole stream.
		return events
	}
	if !c.hasStarted {
		events = append(events, c.createMessageStart())
		c.hasStarted = true
	}
	if usage, ok := m["usage"].(map[string]any); ok {
		c.accumulatedUsage = usage
	}
	choices, _ := m["choices"].([]any)
	for _, ch := range choices {
		choice, _ := ch.(map[string]any)
		if delta, ok := choice["delta"].(map[string]any); ok {
			// reasoning_content: Kimi k2-thinking / DeepSeek-reasoner /
			// GLM-4.7 etc.
			if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
				events = append(events, c.onReasoningDelta(rc)...)
			}

			// Gemini-via-Chat-Completions signals thinking through
			// extra_content.google.thought=true on text deltas.
			isGeminiThinking := false
			if extraContent, ok := delta["extra_content"].(map[string]any); ok {
				if google, ok := extraContent["google"].(map[string]any); ok {
					if thought, ok := google["thought"].(bool); ok && thought {
						isGeminiThinking = true
					}
				}
			}

			if txt, ok := delta["content"].(string); ok && txt != "" {
				if isGeminiThinking {
					events = append(events, c.onReasoningDelta(txt)...)
				} else {
					events = append(events, c.onTextDelta(txt)...)
				}
			}

			if tcs, ok := delta["tool_calls"].([]any); ok && len(tcs) > 0 {
				for _, tci := range tcs {
					if tcm, ok := tci.(map[string]any); ok {
						events = append(events, c.onToolCallDelta(tcm)...)
					}
				}
			}
		}
		if fr, _ := choice["finish_reason"].(string); fr != "" {
			switch fr {
			case "stop":
				c.stopReason = "end_turn"
			case "length":
				c.stopReason = "max_tokens"
			case "tool_calls":
				// Resolved in OnStreamDone based on lastBlockWasToolUse.
			case "content_filter":
				c.stopReason = "refusal"
			default:
				c.stopReason = fr
			}
		}
	}
	return events
}

func (c *Converter) onReasoningDelta(delta string) []string {
	var ev []string
	if !c.thinkingStarted {
		// OpenRouter and some compatibles emit a leading text delta
		// (often "\n\n") before reasoning_content. Close any open text
		// block before starting the thinking block.
		if c.textStarted {
			ev = append(ev, c.createContentBlockStop(c.currentIndex))
			c.textStarted = false
			c.currentIndex++
		}
		ev = append(ev, c.createContentBlockStart("thinking"))
		c.thinkingStarted = true
	}
	ev = append(ev, c.createContentBlockDelta("thinking", delta))
	return ev
}

func (c *Converter) onTextDelta(delta string) []string {
	var ev []string
	if c.thinkingStarted && !c.thinkingCompleted {
		ev = append(ev, c.createContentBlockStop(c.currentIndex))
		c.thinkingCompleted = true
		c.currentIndex++
	}
	// Text after a tool_use: close the active tool block first.
	if c.activeToolOAIndex >= 0 {
		if state := c.toolCallStates[c.activeToolOAIndex]; state != nil && state.started {
			ev = append(ev, c.createContentBlockStop(state.blockIndex))
			c.currentIndex = state.blockIndex + 1
			c.activeToolOAIndex = -1
		}
	}
	if !c.textStarted {
		ev = append(ev, c.createContentBlockStart("text"))
		c.textStarted = true
	}
	ev = append(ev, c.createContentBlockDelta("text", delta))
	c.sawPostToolText = true
	return ev
}

func (c *Converter) onToolCallDelta(tc map[string]any) []string {
	var ev []string

	// 1. OpenAI tool_calls index (defaults to 0 to tolerate providers
	// that omit the field for the first call).
	oaIndex := 0
	if idx, ok := tc["index"].(float64); ok {
		oaIndex = int(idx)
	}

	// 2. function.name / function.arguments
	var name, arguments string
	if fn, ok := tc["function"].(map[string]any); ok {
		if n, ok := fn["name"].(string); ok {
			name = n
		}
		if args, ok := fn["arguments"].(string); ok {
			arguments = args
		}
	}

	// 3. tool_call id. We forward it verbatim; the SoT used to wrap
	// Kimi IDs but Plan B removed that, see service/llm/converter.go:2100-2134.
	var idVal string
	if id, ok := tc["id"].(string); ok {
		idVal = id
	}

	// 4. Find or create per-OA-index state.
	state, exists := c.toolCallStates[oaIndex]
	if !exists {
		state = &toolCallPending{blockIndex: -1}
		c.toolCallStates[oaIndex] = state
	}

	// 5. Accumulate name and id.
	if name != "" {
		state.name = name
	}
	if idVal != "" {
		state.id = idVal
	}

	// 6. No id yet: buffer the arguments and wait.
	if state.id == "" {
		if arguments != "" {
			state.args = append(state.args, arguments)
		}
		return ev
	}

	// 7. First time we can open the content block.
	if !state.started {
		if c.textStarted {
			ev = append(ev, c.createContentBlockStop(c.currentIndex))
			c.currentIndex++
			c.textStarted = false
		}
		if c.thinkingStarted && !c.thinkingCompleted {
			ev = append(ev, c.createContentBlockStop(c.currentIndex))
			c.thinkingCompleted = true
			c.currentIndex++
		}
		// Close the previous tool block, if any, when switching to a
		// new OA index.
		if c.activeToolOAIndex >= 0 && c.activeToolOAIndex != oaIndex {
			if prev := c.toolCallStates[c.activeToolOAIndex]; prev != nil && prev.started {
				ev = append(ev, c.createContentBlockStop(prev.blockIndex))
				c.currentIndex = prev.blockIndex + 1
			}
		}

		state.blockIndex = c.currentIndex
		ev = append(ev, c.createToolUseStart(state.name, state.id))
		state.started = true
		c.activeToolOAIndex = oaIndex
		c.anyToolStarted = true
		c.lastBlockWasToolUse = true
		c.sawPostToolText = false

		// Flush buffered arguments first.
		for _, a := range state.args {
			ev = append(ev, c.createToolUseArgsDeltaForBlock(a, state.blockIndex))
		}
		state.args = nil

		if arguments != "" {
			ev = append(ev, c.createToolUseArgsDeltaForBlock(arguments, state.blockIndex))
		}
		return ev
	}

	// 8. Block already open: just stream more arguments.
	if arguments != "" {
		ev = append(ev, c.createToolUseArgsDeltaForBlock(arguments, state.blockIndex))
	}
	return ev
}

// OnStreamDone is invoked once the upstream stream terminates without
// error and emits the final message_delta + message_stop pair. Callers
// MUST invoke it exactly once; calling it again is a no-op for the
// header but harmless (the block-close logic guards on state.started).
func (c *Converter) OnStreamDone() []string {
	if c.finished {
		return nil
	}
	c.finished = true
	var ev []string

	// 1. If we have any tool calls that buffered content but never got
	// an id (and therefore never opened a block), open them now with a
	// synthetic fallback id. This mirrors the SoT.
	var unstartedIndices []int
	for idx, state := range c.toolCallStates {
		if !state.started && (state.name != "" || len(state.args) > 0) {
			unstartedIndices = append(unstartedIndices, idx)
		}
	}
	sort.Ints(unstartedIndices)

	for _, idx := range unstartedIndices {
		state := c.toolCallStates[idx]

		if c.textStarted {
			ev = append(ev, c.createContentBlockStop(c.currentIndex))
			c.currentIndex++
			c.textStarted = false
		}
		if c.thinkingStarted && !c.thinkingCompleted {
			ev = append(ev, c.createContentBlockStop(c.currentIndex))
			c.thinkingCompleted = true
			c.currentIndex++
		}
		if c.activeToolOAIndex >= 0 {
			if prev := c.toolCallStates[c.activeToolOAIndex]; prev != nil && prev.started {
				ev = append(ev, c.createContentBlockStop(prev.blockIndex))
				c.currentIndex = prev.blockIndex + 1
			}
		}

		fallbackID := state.id
		if fallbackID == "" {
			fallbackID = fallbackToolCallID(c.currentIndex)
		}
		state.blockIndex = c.currentIndex
		ev = append(ev, c.createToolUseStart(state.name, fallbackID))
		state.started = true
		c.activeToolOAIndex = idx
		c.anyToolStarted = true
		c.lastBlockWasToolUse = true
		c.sawPostToolText = false

		for _, a := range state.args {
			ev = append(ev, c.createToolUseArgsDeltaForBlock(a, state.blockIndex))
		}
		state.args = nil
	}

	// 2. Close the last active block.
	if c.activeToolOAIndex >= 0 {
		if state := c.toolCallStates[c.activeToolOAIndex]; state != nil && state.started {
			ev = append(ev, c.createContentBlockStop(state.blockIndex))
		}
	} else if c.textStarted {
		ev = append(ev, c.createContentBlockStop(c.currentIndex))
	} else if c.thinkingStarted && !c.thinkingCompleted {
		ev = append(ev, c.createContentBlockStop(c.currentIndex))
	}

	// 3. Set stop_reason.
	if c.lastBlockWasToolUse && !c.sawPostToolText {
		c.stopReason = "tool_use"
	}

	ev = append(ev, c.createMessageDelta())
	ev = append(ev, c.createMessageStop())
	return ev
}

// —— Anthropic event builders ——

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
			"usage":         map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createContentBlockStart(blockType string) string {
	contentBlock := map[string]any{}
	switch blockType {
	case "thinking":
		contentBlock["type"] = "thinking"
		contentBlock["thinking"] = ""
	case "text":
		contentBlock["type"] = "text"
		contentBlock["text"] = ""
	case "tool_use":
		contentBlock["type"] = "tool_use"
		contentBlock["id"] = fallbackToolCallID(c.currentIndex)
		contentBlock["name"] = ""
		contentBlock["input"] = map[string]any{}
	}
	event := map[string]any{
		"type":          "content_block_start",
		"index":         c.currentIndex,
		"content_block": contentBlock,
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createToolUseStart(name, id string) string {
	event := map[string]any{
		"type":  "content_block_start",
		"index": c.currentIndex,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    id,
			"name":  name,
			"input": map[string]any{},
		},
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createToolUseArgsDeltaForBlock(args string, blockIndex int) string {
	event := map[string]any{
		"type":  "content_block_delta",
		"index": blockIndex,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": args},
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createContentBlockDelta(blockType, delta string) string {
	deltaObj := map[string]any{}
	switch blockType {
	case "thinking":
		deltaObj["type"] = "thinking_delta"
		deltaObj["thinking"] = delta
	case "text":
		deltaObj["type"] = "text_delta"
		deltaObj["text"] = delta
	}
	event := map[string]any{
		"type":  "content_block_delta",
		"index": c.currentIndex,
		"delta": deltaObj,
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
	stopReason := strings.ToLower(c.stopReason)

	delta := map[string]any{
		"stop_reason":   stopReason,
		"stop_sequence": nil,
	}

	event := map[string]any{
		"type":  "message_delta",
		"delta": delta,
	}

	// Anthropic's message_delta carries usage.output_tokens.
	if len(c.accumulatedUsage) > 0 {
		var outputTokens int
		if v, ok := c.accumulatedUsage["completion_tokens"].(float64); ok {
			outputTokens = int(v)
		} else if v, ok := c.accumulatedUsage["completion_tokens"].(int); ok {
			outputTokens = v
		}
		event["usage"] = map[string]any{
			"output_tokens": outputTokens,
		}
	}

	b, _ := json.Marshal(event)
	return string(b)
}

func (c *Converter) createMessageStop() string {
	event := map[string]any{"type": "message_stop"}
	if len(c.accumulatedUsage) > 0 {
		anthropicUsage := map[string]any{}

		var promptTokens int
		if v, ok := c.accumulatedUsage["prompt_tokens"].(float64); ok {
			promptTokens = int(v)
		} else if v, ok := c.accumulatedUsage["prompt_tokens"].(int); ok {
			promptTokens = v
		}

		var completionTokens int
		if v, ok := c.accumulatedUsage["completion_tokens"].(float64); ok {
			completionTokens = int(v)
		} else if v, ok := c.accumulatedUsage["completion_tokens"].(int); ok {
			completionTokens = v
		}

		// cached_tokens: support both nested (OpenAI / GLM) and flat
		// (Gemini-via-OpenAI) layouts.
		var cachedTokens int
		if details, ok := c.accumulatedUsage["prompt_tokens_details"].(map[string]any); ok {
			if cached, ok := details["cached_tokens"].(float64); ok {
				cachedTokens = int(cached)
			} else if cached, ok := details["cached_tokens"].(int); ok {
				cachedTokens = cached
			}
		}
		if cachedTokens == 0 {
			if v, ok := c.accumulatedUsage["cached_tokens"].(float64); ok {
				cachedTokens = int(v)
			} else if v, ok := c.accumulatedUsage["cached_tokens"].(int); ok {
				cachedTokens = v
			}
		}

		// OpenAI/GLM/Gemini: prompt_tokens already includes cached.
		// Anthropic: input_tokens excludes cache_read_input_tokens.
		inputTokens := promptTokens - cachedTokens

		anthropicUsage["input_tokens"] = inputTokens
		anthropicUsage["output_tokens"] = completionTokens
		// Always emit cache fields, even if zero, per Anthropic spec.
		anthropicUsage["cache_read_input_tokens"] = cachedTokens
		anthropicUsage["cache_creation_input_tokens"] = 0

		event["usage"] = anthropicUsage
	}
	b, _ := json.Marshal(event)
	return string(b)
}

func fallbackToolCallID(idx int) string {
	// Matches the SoT fallback in service/llm/converter.go:2248.
	return "call_" + itoa(idx)
}

// itoa avoids the (small) overhead of strconv for the common case.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
