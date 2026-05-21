// Package hybridstream provides a unified Anthropic-protocol facade over
// multiple LLM providers. All public request and response types are aliases
// to github.com/anthropics/anthropic-sdk-go so callers need only learn the
// Anthropic Messages API.
package hybridstream

import (
	"encoding/json"
	"errors"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/hybridstream/adapter"
)

// ProviderKind identifies one of the supported backends.
type ProviderKind string

const (
	// ProviderAnthropic is the native Anthropic Messages API (passthrough).
	ProviderAnthropic ProviderKind = "anthropic"
	// ProviderOpenAIResponses is the OpenAI GPT-5 Response API.
	ProviderOpenAIResponses ProviderKind = "openai_responses"
	// ProviderGeminiNative is the Google Vertex/AI Studio streamGenerateContent API.
	ProviderGeminiNative ProviderKind = "gemini_native"
	// ProviderChatCompletions is the OpenAI-compatible ChatCompletion API
	// (Kimi/GLM/DeepSeek/OpenAI ChatGPT etc.).
	ProviderChatCompletions ProviderKind = "chat_completions"
)

// MessageNewParams is the canonical request type (Anthropic Messages API).
// Re-exported so callers only need to depend on this package.
type MessageNewParams = anthropic.MessageNewParams

// Message is the canonical non-streaming response.
type Message = anthropic.Message

// MessageStreamEventUnion is the canonical streaming event emitted by the
// StreamReader. All adapters normalise their upstream protocol into this
// shape.
type MessageStreamEventUnion = anthropic.MessageStreamEventUnion

// Sentinel errors exposed to callers. Wrap using fmt.Errorf("%w: ...", err).
var (
	// ErrProviderNotConfigured is returned when routing selects a provider
	// that was never registered via the functional options.
	ErrProviderNotConfigured = errors.New("hybridstream: provider not configured")
	// ErrUnknownModel is returned when no route rule matches the request
	// model and no override is installed.
	ErrUnknownModel = errors.New("hybridstream: unable to route model")
	// ErrStreamClosed is returned by StreamReader when the upstream stream
	// has been fully consumed.
	ErrStreamClosed = errors.New("hybridstream: stream closed")
	// ErrUnsupported is returned by adapters that lack a native
	// non-streaming endpoint. The top-level Client uses errors.Is to
	// detect this sentinel and falls back to draining Stream. It is an
	// alias of adapter.ErrUnsupported so every adapter and the parent
	// package share one comparable value.
	ErrUnsupported = adapter.ErrUnsupported
)

// DecodeEvent parses a raw Anthropic-JSON event payload into a typed event.
// Exposed for callers that consume SSE themselves (e.g. testing).
func DecodeEvent(data []byte) (MessageStreamEventUnion, error) {
	var evt MessageStreamEventUnion
	if err := json.Unmarshal(data, &evt); err != nil {
		return evt, err
	}
	return evt, nil
}
