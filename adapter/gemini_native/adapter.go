// Package gemini_native implements the hybridstream adapter for
// Google's Vertex AI / AI Studio streamGenerateContent JSON+SSE API,
// translating its events into the Anthropic Messages event stream.
//
// The protocol port is a stripped-down version of the converter that
// ships in the llm-proxy repository (service/llm/converter.go). See
// converter.go and request.go in this directory for details on which
// behaviours were intentionally simplified.
package gemini_native

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/hybridstream/adapter"
	"github.com/shahao/hybridstream/internal/ssehelper"
)

// Config captures the subset of GeminiConfig that the adapter consumes.
//
// Two upstream modes are supported:
//
//  1. Vertex AI: set Project, Location, and APIKey (which is treated
//     as a bearer access token, e.g. produced by `gcloud auth
//     print-access-token`). Publisher defaults to "google".
//  2. AI Studio: set BaseURL=https://generativelanguage.googleapis.com
//     and APIKey to your AI Studio key. Project/Location are ignored.
//     The key is appended as ?key=... per Google's public API rules.
type Config struct {
	Project      string
	Location     string
	Publisher    string
	APIKey       string
	BaseURL      string
	ExtraHeaders http.Header
	HTTPClient   *http.Client
}

// Adapter drives the Gemini streamGenerateContent endpoint and rewrites
// its SSE stream as Anthropic events.
type Adapter struct {
	cfg Config
}

// New validates cfg and returns a ready-to-use Adapter.
func New(cfg Config) (*Adapter, error) {
	if cfg.APIKey == "" && (cfg.Project == "" || cfg.Location == "") {
		return nil, errors.New("hybridstream/gemini_native: either APIKey or (Project + Location) must be set")
	}
	if cfg.Publisher == "" {
		cfg.Publisher = "google"
	}
	if cfg.BaseURL == "" {
		// Default to Vertex AI; AI-Studio callers must set it
		// explicitly.
		cfg.BaseURL = "https://aiplatform.googleapis.com"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &Adapter{cfg: cfg}, nil
}

// Name satisfies adapter.Adapter.
func (a *Adapter) Name() string { return "gemini_native" }

// Invoke returns ErrUnsupported; the outer Client drains Stream() for
// non-streaming calls.
func (a *Adapter) Invoke(_ context.Context, _ anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	return nil, errUnsupported
}

// Stream issues the request and launches a goroutine that converts
// Gemini SSE frames to Anthropic events.
func (a *Adapter) Stream(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	model := string(req.Model)
	if model == "" {
		return nil, errors.New("hybridstream/gemini_native: model is required")
	}

	payload, err := BuildRequest(req)
	if err != nil {
		return nil, err
	}

	endpoint, useBearer := a.buildEndpoint(model)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if useBearer {
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
		return nil, fmt.Errorf("hybridstream/gemini_native: upstream status %d: %s", resp.StatusCode, string(body))
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
		// Stream closed cleanly: flush message_delta + message_stop.
		for _, out := range conv.OnStreamDone() {
			ch <- adapter.RawEvent{Data: []byte(out)}
		}
	}()
	return ch, nil
}

// buildEndpoint returns the fully-qualified upstream URL plus a flag
// indicating whether to send the API key as a Bearer header (Vertex
// mode) or as a `key=` query string (AI Studio mode).
//
// Heuristic: if Project and Location are both set we use the Vertex
// path; otherwise we assume AI Studio.
func (a *Adapter) buildEndpoint(model string) (string, bool) {
	base := strings.TrimRight(a.cfg.BaseURL, "/")
	if a.cfg.Project != "" && a.cfg.Location != "" {
		// Vertex AI form (matches the reference at
		// service/llm/converter.go:2785).
		path := fmt.Sprintf(
			"%s/v1/projects/%s/locations/%s/publishers/%s/models/%s:streamGenerateContent",
			base, a.cfg.Project, a.cfg.Location, a.cfg.Publisher, model,
		)
		return path + "?alt=sse", true
	}
	// AI Studio form: /v1beta/models/{model}:streamGenerateContent?alt=sse&key=...
	path := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent", base, model)
	q := url.Values{}
	q.Set("alt", "sse")
	if a.cfg.APIKey != "" {
		q.Set("key", a.cfg.APIKey)
	}
	return path + "?" + q.Encode(), false
}

// errUnsupported mirrors the parent package's sentinel; we duplicate it
// here to avoid an import cycle.
var errUnsupported = errors.New("hybridstream: feature not yet supported")

// ErrUnsupported is exposed so callers can errors.Is against it.
var ErrUnsupported = errUnsupported

// ErrNotImplemented is retained for backwards-compat with callers that
// referenced the scaffold's sentinel; new code should use ErrUnsupported.
var ErrNotImplemented = errUnsupported
