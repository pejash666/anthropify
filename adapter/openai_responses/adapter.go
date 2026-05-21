package openai_responses

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

// Config is the subset of OpenAIConfig this adapter consumes.
type Config struct {
	APIKey       string
	BaseURL      string
	Organization string
	Project      string
	ExtraHeaders http.Header
	HTTPClient   *http.Client
}

// Adapter drives the OpenAI /v1/responses endpoint and rewrites its SSE
// stream as Anthropic events.
type Adapter struct {
	cfg Config
}

// New validates cfg and returns a ready-to-use Adapter.
func New(cfg Config) (*Adapter, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("hybridstream/openai_responses: APIKey is required")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &Adapter{cfg: cfg}, nil
}

// Name satisfies adapter.Adapter.
func (a *Adapter) Name() string { return "openai_responses" }

// Invoke returns ErrUnsupported; the outer Client drains Stream() for
// non-streaming calls. A native path may be added later.
func (a *Adapter) Invoke(_ context.Context, _ anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	return nil, errUnsupported
}

// Stream issues the request and launches a goroutine that converts the
// upstream SSE frames to Anthropic events.
func (a *Adapter) Stream(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	payload, err := BuildRequest(req, true)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(a.cfg.BaseURL, "/") + "/v1/responses"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+a.cfg.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")
	if a.cfg.Organization != "" {
		httpReq.Header.Set("OpenAI-Organization", a.cfg.Organization)
	}
	if a.cfg.Project != "" {
		httpReq.Header.Set("OpenAI-Project", a.cfg.Project)
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
		return nil, fmt.Errorf("hybridstream/openai_responses: upstream status %d: %s", resp.StatusCode, string(body))
	}

	ch := make(chan adapter.RawEvent, 16)
	go func() {
		defer close(ch)
		defer resp.Body.Close()
		conv := NewConverter(genMessageID())
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
		}
	}()
	return ch, nil
}

// errUnsupported is returned by Invoke; the sentinel lives in the parent
// package, so we re-export it through a small shim to avoid an import
// cycle (parent imports this package).
var errUnsupported = errors.New("hybridstream: feature not yet supported")

// ErrUnsupported is exposed so callers can errors.Is against it.
var ErrUnsupported = errUnsupported
