// Package chat_completions implements the hybridstream adapter for any
// OpenAI Chat Completions compatible upstream (Kimi / DeepSeek / Qwen /
// GLM / OpenRouter / etc.), translating its SSE stream into the
// Anthropic Messages event stream.
//
// The protocol port follows the converter that ships in the llm-proxy
// repository (service/llm/converter.go); see converter.go and
// request.go in this directory for details on which behaviours were
// intentionally simplified for the SDK surface.
//
// One Adapter instance serves one upstream backend. Multiple backends
// are routed to from the top-level Client via WithChatCompletion(name,
// cfg) / WithModelRoute, allowing model strings like "kimi-k2-thinking"
// to land on the "kimi" backend, "deepseek-v4-pro" on the "deepseek"
// backend, and so on.
package chat_completions

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/hybridstream/adapter"
	"github.com/shahao/hybridstream/internal/ssehelper"
)

// Config captures the per-backend OpenAI-compatible settings consumed
// by this adapter. Settings come from the top-level options package's
// OpenAICompatConfig and are validated again here so a misconfigured
// backend fails at construction time rather than on first request.
type Config struct {
	// BaseURL is the upstream root, e.g. "https://api.moonshot.cn/v1".
	// The adapter appends "/chat/completions"; trailing slashes are
	// tolerated.
	BaseURL string

	// APIKey is sent as Authorization: Bearer <APIKey> when non-empty.
	APIKey string

	// ExtraHeaders are added to every request, useful for tenant
	// identifiers and feature flags.
	ExtraHeaders http.Header

	// DefaultModel is used when the caller's req.Model is empty. Most
	// integrations leave this blank and rely on the routed model.
	DefaultModel string

	HTTPClient *http.Client
}

// Adapter drives one OpenAI-compatible ChatCompletion backend and
// rewrites its SSE stream as Anthropic events.
type Adapter struct {
	name string
	cfg  Config
}

// New validates cfg and returns an Adapter labelled with the given
// name. The label is surfaced via Adapter.Name for diagnostic routing.
func New(name string, cfg Config) (*Adapter, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("hybridstream/chat_completions: BaseURL is required")
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &Adapter{name: name, cfg: cfg}, nil
}

// Name satisfies adapter.Adapter and identifies the configured backend.
func (a *Adapter) Name() string { return "chat_completions:" + a.name }

// Invoke returns ErrUnsupported; the outer Client drains Stream() for
// non-streaming calls.
func (a *Adapter) Invoke(_ context.Context, _ anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	return nil, ErrUnsupported
}

// Stream issues the request and launches a goroutine that converts
// upstream SSE frames to Anthropic events.
func (a *Adapter) Stream(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	model := string(req.Model)
	if model == "" {
		model = a.cfg.DefaultModel
		req.Model = anthropicsdk.Model(model)
	}
	if model == "" {
		return nil, errors.New("hybridstream/chat_completions: model is required")
	}

	payload, err := BuildRequest(req, true)
	if err != nil {
		return nil, err
	}

	endpoint := a.buildEndpoint()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if a.cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+a.cfg.APIKey)
	}
	for k, vs := range a.cfg.ExtraHeaders {
		for _, v := range vs {
			httpReq.Header.Add(k, v)
		}
	}

	resp, err := a.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("hybridstream/chat_completions[%s]: upstream status %d: %s", a.name, resp.StatusCode, string(body))
	}

	ch := make(chan adapter.RawEvent, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		conv := NewConverter(model, genMessageID())
		dec := ssehelper.NewDecoder(resp.Body)
		for dec.Next() {
			if ctx.Err() != nil {
				ch <- adapter.RawEvent{Err: ctx.Err()}
				return
			}
			evt := dec.Event()
			if evt.Data == "" || evt.Data == "[DONE]" {
				continue
			}
			for _, out := range conv.ConvertEvent(evt.Data) {
				ch <- adapter.RawEvent{Data: []byte(out)}
			}
		}
		if err := dec.Err(); err != nil {
			ch <- adapter.RawEvent{Err: err}
			return
		}
		for _, out := range conv.OnStreamDone() {
			ch <- adapter.RawEvent{Data: []byte(out)}
		}
	}()
	return ch, nil
}

// buildEndpoint joins BaseURL with the chat/completions path while
// tolerating callers that either include or omit the version suffix.
func (a *Adapter) buildEndpoint() string {
	base := strings.TrimRight(a.cfg.BaseURL, "/")
	// If the caller already pointed at the endpoint, use it as-is.
	if strings.HasSuffix(base, "/chat/completions") {
		return base
	}
	return base + "/chat/completions"
}

// errUnsupported mirrors the parent package's sentinel; we duplicate it
// here to avoid an import cycle.
var errUnsupported = errors.New("hybridstream: feature not yet supported")

// ErrUnsupported is exposed so callers can errors.Is against it.
var ErrUnsupported = errUnsupported

// ErrNotImplemented is retained for backwards-compat with callers that
// referenced the scaffold's sentinel; new code should use ErrUnsupported.
var ErrNotImplemented = errUnsupported
