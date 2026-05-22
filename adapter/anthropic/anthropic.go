// Package anthropic implements the Anthropic passthrough adapter. Because
// the canonical request/response format already matches Anthropic's API,
// this adapter mostly forwards bytes and decodes SSE into normalised
// events.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/anthropify/adapter"
	"github.com/shahao/anthropify/internal/ssehelper"
)

// Config captures the subset of AnthropicConfig that the adapter cares
// about. All fields mirror the top-level option type.
type Config struct {
	APIKey       string
	BaseURL      string
	Version      string
	ExtraHeaders http.Header
	HTTPClient   *http.Client
}

// Adapter forwards requests to the Anthropic Messages API.
type Adapter struct {
	cfg Config
}

// New constructs an Adapter and validates required fields.
func New(cfg Config) (*Adapter, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("anthropify/anthropic: APIKey is required")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com"
	}
	if cfg.Version == "" {
		cfg.Version = "2023-06-01"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &Adapter{cfg: cfg}, nil
}

// Name satisfies adapter.Adapter.
func (a *Adapter) Name() string { return "anthropic" }

// Invoke performs a non-streaming call. It POSTs the request to
// /v1/messages with stream=false and returns the full JSON Message.
func (a *Adapter) Invoke(ctx context.Context, req anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	payload, err := buildPayload(req, false)
	if err != nil {
		return nil, err
	}
	httpReq, err := a.newHTTPRequest(ctx, payload, false)
	if err != nil {
		return nil, err
	}
	resp, err := a.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("anthropify/anthropic: upstream status %d: %s", resp.StatusCode, string(body))
	}
	var msg anthropicsdk.Message
	if err := json.Unmarshal(body, &msg); err != nil {
		return nil, fmt.Errorf("anthropify/anthropic: decode response: %w", err)
	}
	return &msg, nil
}

// Stream performs a streaming call. The returned channel emits each SSE
// frame as a RawEvent and is closed when the stream ends.
func (a *Adapter) Stream(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	payload, err := buildPayload(req, true)
	if err != nil {
		return nil, err
	}
	httpReq, err := a.newHTTPRequest(ctx, payload, true)
	if err != nil {
		return nil, err
	}
	resp, err := a.cfg.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("anthropify/anthropic: upstream status %d: %s", resp.StatusCode, string(body))
	}

	ch := make(chan adapter.RawEvent, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		dec := ssehelper.NewDecoder(resp.Body)
		for dec.Next() {
			if ctx.Err() != nil {
				ch <- adapter.RawEvent{Err: ctx.Err()}
				return
			}
			evt := dec.Event()
			if evt.Data == "" {
				continue
			}
			ch <- adapter.RawEvent{Data: []byte(evt.Data)}
		}
		if err := dec.Err(); err != nil {
			ch <- adapter.RawEvent{Err: err}
		}
	}()
	return ch, nil
}

func (a *Adapter) newHTTPRequest(ctx context.Context, payload []byte, stream bool) (*http.Request, error) {
	url := strings.TrimRight(a.cfg.BaseURL, "/") + "/v1/messages"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.cfg.APIKey)
	req.Header.Set("anthropic-version", a.cfg.Version)
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	for k, vs := range a.cfg.ExtraHeaders {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	return req, nil
}

// buildPayload serialises the MessageNewParams and injects the "stream"
// flag. We round-trip through a map because MessageNewParams is an
// opaque union type.
func buildPayload(req anthropicsdk.MessageNewParams, stream bool) ([]byte, error) {
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("anthropify/anthropic: marshal request: %w", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	if stream {
		m["stream"] = true
	} else {
		delete(m, "stream")
	}
	return json.Marshal(m)
}
