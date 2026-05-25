// Package gemini_native implements the anthropify adapter for
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

	"github.com/shahao/anthropify/adapter"
	"github.com/shahao/anthropify/internal/ssehelper"
)

// GeminiMode selects which Google endpoint family + auth shape the
// adapter targets. The three production modes are exposed as named
// constants; GeminiModeAuto preserves the original heuristic so
// existing callers do not need to touch their config.
type GeminiMode int

const (
	// GeminiModeAuto picks a mode from the populated config fields:
	// project+location wins (Vertex), otherwise an APIKey routes to
	// AI Studio. Auto never selects Express; that mode must be
	// requested explicitly to avoid breaking callers that historically
	// used AI Studio keys against generativelanguage.googleapis.com.
	GeminiModeAuto GeminiMode = iota
	// GeminiModeStudio targets generativelanguage.googleapis.com with
	// an AI Studio key passed via ?key=.
	GeminiModeStudio
	// GeminiModeExpress targets aiplatform.googleapis.com with the
	// simplified /v1/publishers/google path and ?key= auth (Vertex
	// Express Mode, used with AQ.-prefixed keys).
	GeminiModeExpress
	// GeminiModeVertex targets aiplatform.googleapis.com with the
	// full /v1/projects/.../locations/... path and Bearer auth.
	GeminiModeVertex
)

const (
	defaultVertexBaseURL  = "https://aiplatform.googleapis.com"
	defaultStudioBaseURL  = "https://generativelanguage.googleapis.com"
	defaultExpressBaseURL = "https://aiplatform.googleapis.com"
)

// Config captures the subset of GeminiConfig that the adapter consumes.
//
// Three upstream modes are supported, selected by Mode:
//
//  1. Vertex AI (GeminiModeVertex): Project + Location + APIKey
//     (treated as a Bearer access token, e.g. `gcloud auth
//     print-access-token`). Publisher defaults to "google".
//  2. AI Studio (GeminiModeStudio): APIKey only, sent as ?key=
//     against generativelanguage.googleapis.com.
//  3. Vertex Express (GeminiModeExpress): APIKey only (typically
//     AQ.-prefixed), sent as ?key= against aiplatform.googleapis.com
//     using the simplified /v1/publishers/google path.
//
// GeminiModeAuto (the zero value) keeps the legacy heuristic:
// project+location => Vertex, else APIKey => Studio.
type Config struct {
	Mode         GeminiMode
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
	cfg      Config
	resolved GeminiMode // post-resolution; never GeminiModeAuto
}

// New validates cfg and returns a ready-to-use Adapter.
func New(cfg Config) (*Adapter, error) {
	if cfg.Publisher == "" {
		cfg.Publisher = "google"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}

	resolved, err := resolveMode(cfg)
	if err != nil {
		return nil, err
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL(resolved)
	}

	return &Adapter{cfg: cfg, resolved: resolved}, nil
}

// resolveMode collapses Auto into a concrete mode and validates that
// the requested mode has the fields it needs.
func resolveMode(cfg Config) (GeminiMode, error) {
	switch cfg.Mode {
	case GeminiModeAuto:
		if cfg.Project != "" && cfg.Location != "" {
			return GeminiModeVertex, nil
		}
		if cfg.APIKey != "" {
			return GeminiModeStudio, nil
		}
		return 0, errors.New("anthropify/gemini_native: either APIKey or (Project + Location) must be set")
	case GeminiModeStudio:
		if cfg.APIKey == "" {
			return 0, errors.New("anthropify/gemini_native: Studio mode requires APIKey")
		}
		return GeminiModeStudio, nil
	case GeminiModeExpress:
		if cfg.APIKey == "" {
			return 0, errors.New("anthropify/gemini_native: Express mode requires APIKey")
		}
		return GeminiModeExpress, nil
	case GeminiModeVertex:
		if cfg.Project == "" || cfg.Location == "" {
			return 0, errors.New("anthropify/gemini_native: Vertex mode requires Project and Location")
		}
		if cfg.APIKey == "" {
			return 0, errors.New("anthropify/gemini_native: Vertex mode requires APIKey (Bearer access token)")
		}
		return GeminiModeVertex, nil
	default:
		return 0, fmt.Errorf("anthropify/gemini_native: unknown Mode %d", cfg.Mode)
	}
}

func defaultBaseURL(mode GeminiMode) string {
	switch mode {
	case GeminiModeStudio:
		return defaultStudioBaseURL
	case GeminiModeExpress:
		return defaultExpressBaseURL
	default:
		return defaultVertexBaseURL
	}
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
		return nil, errors.New("anthropify/gemini_native: model is required")
	}

	payload, err := BuildRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	endpoint, useBearer := a.buildEndpoint(model, true)
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
		return nil, fmt.Errorf("anthropify/gemini_native: upstream status %d: %s", resp.StatusCode, string(body))
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
// mode) or as a `key=` query string (Studio / Express modes).
//
// The stream flag toggles between :streamGenerateContent (with
// alt=sse) and :generateContent. The non-stream branch is provided
// for forward-compatibility with a future Invoke implementation; the
// current Stream-only flow always passes stream=true.
func (a *Adapter) buildEndpoint(model string, stream bool) (string, bool) {
	base := strings.TrimRight(a.cfg.BaseURL, "/")
	action := ":generateContent"
	if stream {
		action = ":streamGenerateContent"
	}

	switch a.resolved {
	case GeminiModeVertex:
		path := fmt.Sprintf(
			"%s/v1/projects/%s/locations/%s/publishers/%s/models/%s%s",
			base, a.cfg.Project, a.cfg.Location, a.cfg.Publisher, model, action,
		)
		if stream {
			return path + "?alt=sse", true
		}
		return path, true

	case GeminiModeExpress:
		path := fmt.Sprintf(
			"%s/v1/publishers/%s/models/%s%s",
			base, a.cfg.Publisher, model, action,
		)
		q := url.Values{}
		if stream {
			q.Set("alt", "sse")
		}
		q.Set("key", a.cfg.APIKey)
		return path + "?" + q.Encode(), false

	case GeminiModeStudio:
		fallthrough
	default:
		path := fmt.Sprintf("%s/v1beta/models/%s%s", base, model, action)
		q := url.Values{}
		if stream {
			q.Set("alt", "sse")
		}
		if a.cfg.APIKey != "" {
			q.Set("key", a.cfg.APIKey)
		}
		return path + "?" + q.Encode(), false
	}
}

// errUnsupported mirrors the parent package's sentinel; we duplicate it
// here to avoid an import cycle.
var errUnsupported = adapter.ErrUnsupported

// ErrUnsupported is exposed so callers can errors.Is against it.
var ErrUnsupported = errUnsupported

// ErrNotImplemented is retained for backwards-compat with callers that
// referenced the scaffold's sentinel; new code should use ErrUnsupported.
var ErrNotImplemented = errUnsupported
