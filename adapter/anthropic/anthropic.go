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
	"github.com/shahao/anthropify/internal/schema"
	"github.com/shahao/anthropify/internal/ssehelper"
)

// AnthropicMode selects which upstream the adapter dispatches to. The
// zero value (AnthropicModeDirect) preserves v0.1.x behaviour: the
// adapter speaks raw HTTPS to api.anthropic.com (or any
// AnthropicConfig.BaseURL such as MiniMax). AnthropicModeBedrock
// switches to a delegated path through anthropic-sdk-go's bedrock
// subpackage, which handles SigV4 signing, URL rewriting,
// anthropic_version body injection, anthropic-beta header to body
// translation, and AWS event-stream decoding. Set internally by the
// top-level WithAnthropicBedrock option; not a public knob users
// toggle on AnthropicConfig.
type AnthropicMode int

const (
	// AnthropicModeDirect speaks raw HTTPS to an Anthropic-protocol
	// endpoint (api.anthropic.com, MiniMax, or any drop-in compat
	// host). This is the zero value and matches legacy behaviour.
	AnthropicModeDirect AnthropicMode = iota
	// AnthropicModeBedrock dispatches through anthropic-sdk-go's
	// bedrock subpackage; AWS-flavoured fields on Config are required.
	AnthropicModeBedrock
)

// Config captures the subset of AnthropicConfig (plus Bedrock-only
// extensions) that the adapter cares about.
type Config struct {
	// Mode selects between Direct (zero value, used for both Anthropic
	// Inc. and any drop-in compat host such as MiniMax) and Bedrock.
	Mode AnthropicMode

	// Direct-mode fields. Required when Mode == AnthropicModeDirect.
	APIKey       string
	BaseURL      string
	Version      string
	ExtraHeaders http.Header
	HTTPClient   *http.Client

	// Bedrock-mode fields. Required when Mode == AnthropicModeBedrock.
	AWSAccessKeyID     string
	AWSSecretAccessKey string
	AWSSessionToken    string // optional; only for STS / SSO temporary credentials
	AWSRegion          string
}

// Adapter forwards requests to the Anthropic Messages API. When
// cfg.Mode == AnthropicModeBedrock, requests are routed through an
// anthropic-sdk-go client wired with bedrock.WithConfig; otherwise
// requests are sent as raw HTTPS to cfg.BaseURL.
type Adapter struct {
	name string
	cfg  Config

	// bedrockClient is non-nil iff cfg.Mode == AnthropicModeBedrock.
	// It is the SDK client wired through bedrock.WithConfig that
	// handles SigV4 + URL rewrite + body injection + event-stream
	// decoding. See bedrock.go for construction details.
	bedrockClient *anthropicsdk.Client
}

// New constructs an Adapter and validates required fields. The name
// labels the configured backend (e.g. "anthropic", "minimax",
// "bedrock") and is surfaced via Adapter.Name as "anthropic:<name>".
func New(name string, cfg Config) (*Adapter, error) {
	if cfg.Mode == AnthropicModeBedrock {
		return newBedrock(name, cfg)
	}
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
	return &Adapter{name: name, cfg: cfg}, nil
}

// Name satisfies adapter.Adapter and identifies the configured backend.
func (a *Adapter) Name() string { return "anthropic:" + a.name }

// Invoke performs a non-streaming call. It POSTs the request to
// /v1/messages with stream=false and returns the full JSON Message.
// In Bedrock mode the SDK rewrites the URL and signs the request.
func (a *Adapter) Invoke(ctx context.Context, req anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	if a.cfg.Mode == AnthropicModeBedrock {
		return a.invokeBedrock(ctx, req)
	}
	payload, err := buildPayload(ctx, req, false)
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
// frame as a RawEvent and is closed when the stream ends. In Bedrock
// mode the SDK decodes AWS event-stream frames back into canonical
// Anthropic SSE event JSON before they reach the channel.
func (a *Adapter) Stream(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	if a.cfg.Mode == AnthropicModeBedrock {
		return a.streamBedrock(ctx, req)
	}
	payload, err := buildPayload(ctx, req, true)
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
func buildPayload(ctx context.Context, req anthropicsdk.MessageNewParams, stream bool) ([]byte, error) {
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
	// Even though the Anthropic dialect is identity, we run every
	// tool's input_schema through schema.Normalize so the policy
	// pipeline is uniform across adapters and so stale validation
	// errors (cycles, depth caps) trigger here too if the dialect
	// configuration ever tightens.
	if tools, ok := m["tools"].([]any); ok {
		policy := schema.PolicyFrom(ctx)
		for i, ti := range tools {
			tm, ok := ti.(map[string]any)
			if !ok {
				continue
			}
			if rawSchema, ok := tm["input_schema"].(map[string]any); ok {
				normalised, _, err := schema.Normalize(ctx, rawSchema, schema.DialectAnthropic, policy)
				if err != nil {
					name, _ := tm["name"].(string)
					return nil, fmt.Errorf("anthropify/anthropic: tool %q: %w", name, err)
				}
				tm["input_schema"] = normalised
			}
			tools[i] = tm
		}
		m["tools"] = tools
	}
	return json.Marshal(m)
}
