package openai_responses

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/anthropify/adapter"
)

// captured holds what the fake upstream observed on a single request.
// All fields are written before Close() returns the channel, so reads
// after the Stream goroutine drains are race-free.
type captured struct {
	mu      sync.Mutex
	method  string
	path    string
	rawQuery string
	headers http.Header
	body    []byte
}

func newRecordingServer(t *testing.T, c *captured) *httptest.Server {
	t.Helper()
	const sse = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_test\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c.mu.Lock()
		c.method = r.Method
		c.path = r.URL.Path
		c.rawQuery = r.URL.RawQuery
		c.headers = r.Header.Clone()
		c.body, _ = io.ReadAll(r.Body)
		c.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		_, _ = io.WriteString(w, sse)
	}))
}

func sampleReq(t *testing.T, model string) anthropicsdk.MessageNewParams {
	t.Helper()
	src := []byte(`{"model":"` + model + `","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}
	return req
}

func drain(t *testing.T, ch <-chan adapter.RawEvent) {
	t.Helper()
	for ev := range ch {
		if ev.Err != nil {
			// SSE parse errors at the tail of a tiny canned stream are
			// fine; we only need the upstream HTTP request to have
			// been observed.
			_ = ev.Err
		}
	}
}

// TestStream_AzureSendsBothAuthHeaders proves the dual-auth decision
// from docs/design/v0.2.0-azure-responses.md §5.3: Azure mode sends
// BOTH api-key and Authorization: Bearer on every request, and does
// NOT leak the OpenAI-Organization / OpenAI-Project headers.
func TestStream_AzureSendsBothAuthHeaders(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("azure", Config{
		APIKey:     "secret-key",
		BaseURL:    srv.URL,
		Azure:      true,
		APIVersion: "2025-03-01-preview",
		Deployment: "gpt-5-PTU",
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if got := cap.headers.Get("api-key"); got != "secret-key" {
		t.Errorf("api-key header = %q, want %q", got, "secret-key")
	}
	if got := cap.headers.Get("Authorization"); got != "Bearer secret-key" {
		t.Errorf("Authorization = %q, want %q", got, "Bearer secret-key")
	}
	if got := cap.headers.Get("OpenAI-Organization"); got != "" {
		t.Errorf("OpenAI-Organization should be absent in Azure mode, got %q", got)
	}
	if got := cap.headers.Get("OpenAI-Project"); got != "" {
		t.Errorf("OpenAI-Project should be absent in Azure mode, got %q", got)
	}
}

// TestStream_VanillaSendsOnlyBearer is the negative half of the auth
// contract: vanilla OpenAI must NOT send the api-key header (which
// would confuse logs of api.openai.com itself).
func TestStream_VanillaSendsOnlyBearer(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("openai", Config{
		APIKey:       "sk-vanilla",
		BaseURL:      srv.URL,
		Organization: "org-x",
		Project:      "proj-y",
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if got := cap.headers.Get("Authorization"); got != "Bearer sk-vanilla" {
		t.Errorf("Authorization = %q", got)
	}
	if got := cap.headers.Get("api-key"); got != "" {
		t.Errorf("vanilla mode must not send api-key, got %q", got)
	}
	if got := cap.headers.Get("OpenAI-Organization"); got != "org-x" {
		t.Errorf("OpenAI-Organization = %q", got)
	}
	if got := cap.headers.Get("OpenAI-Project"); got != "proj-y" {
		t.Errorf("OpenAI-Project = %q", got)
	}
}

// TestStream_AzureURLDeploymentBound proves the wire-side URL form
// (path + query) when Deployment is set. This is the deployment-bound
// form documented in §1 of the Azure design doc.
func TestStream_AzureURLDeploymentBound(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    srv.URL,
		Azure:      true,
		APIVersion: "2025-03-01-preview",
		Deployment: "gpt-5-PTU",
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if got, want := cap.path, "/openai/deployments/gpt-5-PTU/responses"; got != want {
		t.Errorf("path = %q, want %q", got, want)
	}
	if got, want := cap.rawQuery, "api-version=2025-03-01-preview"; got != want {
		t.Errorf("query = %q, want %q", got, want)
	}
	if got := cap.method; got != http.MethodPost {
		t.Errorf("method = %q, want POST", got)
	}
}

// TestStream_AzureURLDeploymentless proves the deployment-less URL
// form, which mirrors the llm-proxy reference implementation. The
// deployment name is expected to be carried in the body's `model`
// field — covered separately by request_test.go.
func TestStream_AzureURLDeploymentless(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    srv.URL,
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if got, want := cap.path, "/openai/responses"; got != want {
		t.Errorf("path = %q, want %q", got, want)
	}
	if got, want := cap.rawQuery, "api-version=2025-03-01-preview"; got != want {
		t.Errorf("query = %q, want %q", got, want)
	}
}

// TestStream_AzureExtraHeadersMerged confirms the ExtraHeaders config
// is merged into Azure-mode requests just like vanilla mode (Track A
// expectation).
func TestStream_AzureExtraHeadersMerged(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    srv.URL,
		Azure:      true,
		APIVersion: "2025-03-01-preview",
		ExtraHeaders: http.Header{
			"X-Trace-Id": []string{"abc-123"},
			"X-Custom":   []string{"v1", "v2"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if got := cap.headers.Get("X-Trace-Id"); got != "abc-123" {
		t.Errorf("X-Trace-Id = %q", got)
	}
	if got := cap.headers.Values("X-Custom"); len(got) != 2 || got[0] != "v1" || got[1] != "v2" {
		t.Errorf("X-Custom values = %v", got)
	}
}

// TestStream_AzureBodyModelOverridden checks the wire-side behaviour
// of the layered model fallback: when Deployment is set, the request
// body's `model` field is the Azure deployment name regardless of
// what the caller put in req.Model.
func TestStream_AzureBodyModelOverridden(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    srv.URL,
		Azure:      true,
		APIVersion: "2025-03-01-preview",
		Deployment: "my-azure-deployment",
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "gpt-5-canonical"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if !strings.Contains(string(cap.body), `"model":"my-azure-deployment"`) {
		t.Errorf("body should carry deployment as model, got: %s", cap.body)
	}
	if strings.Contains(string(cap.body), `"model":"gpt-5-canonical"`) {
		t.Errorf("body should NOT carry canonical model when Deployment is set, got: %s", cap.body)
	}
}

// TestStream_AzureBodyModelPassthroughWhenNoDeployment proves the
// deployment-less form's contract: body.model is whatever the caller
// supplied (post-Route.UpstreamModel rewrite, which happens upstream
// in Client.dispatch).
func TestStream_AzureBodyModelPassthroughWhenNoDeployment(t *testing.T) {
	cap := &captured{}
	srv := newRecordingServer(t, cap)
	defer srv.Close()

	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    srv.URL,
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err != nil {
		t.Fatal(err)
	}
	ch, err := a.Stream(context.Background(), sampleReq(t, "deployment-from-route"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	drain(t, ch)

	if !strings.Contains(string(cap.body), `"model":"deployment-from-route"`) {
		t.Errorf("body should pass req.Model through when Deployment unset, got: %s", cap.body)
	}
}
