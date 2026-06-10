package openai_responses

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

	"github.com/pejash666/anthropify/adapter"
	"github.com/pejash666/anthropify/internal/ssehelper"
)

// Config is the subset of OpenAIConfig this adapter consumes.
//
// The first six fields cover vanilla OpenAI Responses. The trailing
// Azure-* fields are zero in vanilla mode; setting Azure=true switches
// the adapter into Azure OpenAI Responses mode. See the Azure track
// design doc (docs/design/v0.2.0-azure-responses.md) for the URL
// composition rules and the dual-auth header decision.
type Config struct {
	APIKey       string
	BaseURL      string
	Organization string
	Project      string
	ExtraHeaders http.Header
	HTTPClient   *http.Client

	// Azure, when true, makes Stream issue requests against an Azure
	// OpenAI resource instead of api.openai.com. Vanilla OpenAI
	// callers leave this false.
	Azure bool
	// APIVersion is the Azure ?api-version=... query string value
	// (e.g. "2025-03-01-preview"). Required when Azure==true.
	APIVersion string
	// Deployment, when non-empty, pins requests to one Azure
	// deployment (URL form: /openai/deployments/<dep>/responses) and
	// overrides the wire-side body model field. Empty falls back to
	// the deployment-less form (/openai/responses) — the body model
	// field then carries the deployment name verbatim, mirroring the
	// llm-proxy reference implementation.
	Deployment string
}

// Adapter drives the OpenAI /v1/responses endpoint and rewrites its SSE
// stream as Anthropic events.
type Adapter struct {
	name string
	cfg  Config
}

// New validates cfg and returns a ready-to-use Adapter labelled with
// the given name. The label (e.g. "openai") is surfaced via
// Adapter.Name as "openai_responses:<name>".
func New(name string, cfg Config) (*Adapter, error) {
	if cfg.APIKey == "" {
		return nil, errors.New("anthropify/openai_responses: APIKey is required")
	}
	if cfg.Azure {
		if cfg.BaseURL == "" {
			return nil, errors.New("anthropify/openai_responses: BaseURL is required in Azure mode")
		}
		if cfg.APIVersion == "" {
			return nil, errors.New("anthropify/openai_responses: APIVersion is required in Azure mode")
		}
	} else if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = http.DefaultClient
	}
	return &Adapter{name: name, cfg: cfg}, nil
}

// Name satisfies adapter.Adapter and identifies the configured backend.
func (a *Adapter) Name() string { return "openai_responses:" + a.name }

// buildEndpoint returns the full request URL according to the active
// mode:
//   - vanilla OpenAI:           <baseURL>/v1/responses
//   - Azure deployment-less:    <baseURL>/openai/responses?api-version=<ver>
//   - Azure deployment-bound:   <baseURL>/openai/deployments/<dep>/responses?api-version=<ver>
//
// Trailing slashes on BaseURL are trimmed exactly once. The query
// string is composed via url.Values to keep encoding correct.
func (a *Adapter) buildEndpoint() string {
	base := strings.TrimRight(a.cfg.BaseURL, "/")
	if !a.cfg.Azure {
		return base + "/v1/responses"
	}
	q := url.Values{}
	q.Set("api-version", a.cfg.APIVersion)
	if a.cfg.Deployment != "" {
		return base + "/openai/deployments/" + a.cfg.Deployment + "/responses?" + q.Encode()
	}
	return base + "/openai/responses?" + q.Encode()
}

// effectiveModel applies the layered fallback documented in the Azure
// design doc: cfg.Deployment > req.Model. The middle Route.UpstreamModel
// layer is already realised upstream by Client.dispatch (which rewrites
// req.Model before the adapter sees it), so the adapter only needs to
// consider the deployment override here. Returns "" when no override
// is needed (caller keeps the canonical req.Model).
func (a *Adapter) effectiveModel() string {
	if a.cfg.Azure && a.cfg.Deployment != "" {
		return a.cfg.Deployment
	}
	return ""
}

// Invoke returns ErrUnsupported; the outer Client drains Stream() for
// non-streaming calls. A native path may be added later.
func (a *Adapter) Invoke(_ context.Context, _ anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	return nil, errUnsupported
}

// Stream issues the request and launches a goroutine that converts the
// upstream SSE frames to Anthropic events.
func (a *Adapter) Stream(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	payload, err := BuildRequestForAzure(ctx, req, true, a.effectiveModel())
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, a.buildEndpoint(), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if a.cfg.Azure {
		// Belt-and-suspenders: api-key is Azure's canonical key
		// header; Authorization: Bearer also works for resource keys
		// (per the llm-proxy reference) and is required for Entra/AAD
		// access tokens. Sending both is robust against APIM gateways
		// that strip one or the other.
		httpReq.Header.Set("api-key", a.cfg.APIKey)
		httpReq.Header.Set("Authorization", "Bearer "+a.cfg.APIKey)
	} else {
		httpReq.Header.Set("Authorization", "Bearer "+a.cfg.APIKey)
		if a.cfg.Organization != "" {
			httpReq.Header.Set("OpenAI-Organization", a.cfg.Organization)
		}
		if a.cfg.Project != "" {
			httpReq.Header.Set("OpenAI-Project", a.cfg.Project)
		}
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
		return nil, fmt.Errorf("anthropify/openai_responses: upstream status %d: %s", resp.StatusCode, string(body))
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

// errUnsupported mirrors the parent package's sentinel; we duplicate it
// here to avoid an import cycle.
var errUnsupported = adapter.ErrUnsupported

// ErrUnsupported is exposed so callers can errors.Is against it.
var ErrUnsupported = errUnsupported
