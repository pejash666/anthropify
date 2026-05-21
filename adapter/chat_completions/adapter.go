// Package chat_completions is scaffolded for OpenAI-compatible
// ChatCompletion backends (Kimi, GLM, DeepSeek, plain OpenAI, ...).
// Only the constructor and wiring are stable today; the SSE converter
// is a TODO tracked against the reference port in
// service/llm/converter.go (ChatCompletionsToAnthropicConverter).
package chat_completions

import (
	"context"
	"errors"
	"net/http"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/hybridstream/adapter"
)

// Config captures the subset of OpenAICompatConfig consumed here.
type Config struct {
	BaseURL      string
	APIKey       string
	ExtraHeaders http.Header
	DefaultModel string
	HTTPClient   *http.Client
}

// Adapter drives a single OpenAI-compatible ChatCompletion backend.
type Adapter struct {
	name string
	cfg  Config
}

// New validates cfg and returns an Adapter labelled with the given name.
// The label is surfaced via Adapter.Name for diagnostic routing.
func New(name string, cfg Config) (*Adapter, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("hybridstream/chat_completions: BaseURL is required")
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &Adapter{name: name, cfg: cfg}, nil
}

// Name satisfies adapter.Adapter.
func (a *Adapter) Name() string { return "chat_completions:" + a.name }

// Invoke returns ErrNotImplemented.
func (a *Adapter) Invoke(_ context.Context, _ anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	return nil, ErrNotImplemented
}

// Stream returns ErrNotImplemented.
func (a *Adapter) Stream(_ context.Context, _ anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	return nil, ErrNotImplemented
}

// ErrNotImplemented indicates the adapter has not been fully ported yet.
var ErrNotImplemented = errors.New("hybridstream/chat_completions: not implemented in MVP; scaffold only")
