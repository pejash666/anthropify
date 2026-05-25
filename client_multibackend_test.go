package anthropify

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

// minimalAnthropicSSE is a tiny but well-formed Anthropic Messages SSE
// reply that exercises message_start through message_stop. The payload
// is intentionally short so each test stub stays self-contained.
const minimalAnthropicSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant","model":"claude-x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
	"event: content_block_start\n" +
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
	"event: content_block_delta\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n" +
	"event: content_block_stop\n" +
	`data: {"type":"content_block_stop","index":0}` + "\n\n" +
	"event: message_delta\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":""},"usage":{"input_tokens":1,"output_tokens":2}}` + "\n\n" +
	"event: message_stop\n" +
	`data: {"type":"message_stop"}` + "\n\n"

// minimalOpenAIResponsesSSE mirrors the canonical /v1/responses stream
// using the same shape as adapter/openai_responses/testdata/text_stream.sse
// so the converter produces a complete Anthropic event sequence.
const minimalOpenAIResponsesSSE = "event: response.created\n" +
	`data: {"type":"response.created","response":{"id":"resp_x","model":"gpt-x"}}` + "\n\n" +
	"event: response.output_text.delta\n" +
	`data: {"type":"response.output_text.delta","delta":"hi"}` + "\n\n" +
	"event: response.completed\n" +
	`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2}}}` + "\n\n"

// recordingStub captures the most recent inbound request headers so
// tests can assert on per-backend isolation. The body is consumed and
// the configured SSE fixture is replayed.
type recordingStub struct {
	mu       sync.Mutex
	hits     int32
	lastAuth string
	lastKey  string
	lastHdr  http.Header
	lastPath string

	statusCode int
	body       string
}

func newRecordingStub(body string) *recordingStub {
	return &recordingStub{statusCode: http.StatusOK, body: body}
}

func (s *recordingStub) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&s.hits, 1)
		s.mu.Lock()
		s.lastAuth = r.Header.Get("Authorization")
		s.lastKey = r.Header.Get("x-api-key")
		s.lastHdr = r.Header.Clone()
		s.lastPath = r.URL.Path
		_, _ = io.Copy(io.Discard, r.Body)
		code := s.statusCode
		body := s.body
		s.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(code)
		_, _ = io.WriteString(w, body)
	}
}

func (s *recordingStub) Hits() int { return int(atomic.LoadInt32(&s.hits)) }
func (s *recordingStub) LastAPIKey() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastKey
}
func (s *recordingStub) LastAuth() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastAuth
}
func (s *recordingStub) LastHeader(key string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastHdr.Get(key)
}
func (s *recordingStub) SetStatus(code int, body string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.statusCode = code
	s.body = body
}

// makeAnthropicReq constructs a minimal MessageNewParams with the given
// model. It is the test-side analogue of `client.CreateMessageStream`'s
// caller side.
func makeAnthropicReq(t *testing.T, model string) anthropicsdk.MessageNewParams {
	t.Helper()
	src := []byte(fmt.Sprintf(`{"model":%q,"max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"."}]}]}`, model))
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return req
}

// drainStream pulls the entire stream and returns concatenated text.
func drainStream(t *testing.T, sr *StreamReader) string {
	t.Helper()
	var sb strings.Builder
	for sr.Next() {
		evt := sr.Current()
		if evt.Type == "content_block_delta" && evt.Delta.Type == "text_delta" {
			sb.WriteString(evt.Delta.Text)
		}
	}
	if err := sr.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	return sb.String()
}

// ----------------------------------------------------------------------
// Anthropic multi-backend tests (§3.7 cases 1-5).
// ----------------------------------------------------------------------

// TestAnthropicMultiBackend_RoutingIsolation registers two anthropic
// backends with distinct BaseURL+APIKey. A request routed to "minimax"
// must hit only that stub and carry MiniMax's x-api-key.
func TestAnthropicMultiBackend_RoutingIsolation(t *testing.T) {
	stubA := newRecordingStub(minimalAnthropicSSE)
	stubM := newRecordingStub(minimalAnthropicSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvM := httptest.NewServer(stubM.handler())
	defer srvM.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{APIKey: "key-claude", BaseURL: srvA.URL}),
		WithAnthropicCompat("minimax", AnthropicConfig{APIKey: "key-mm", BaseURL: srvM.URL}),
		WithModelRoute("minimax-", Route{Provider: ProviderAnthropic, Backend: "minimax"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := makeAnthropicReq(t, "MiniMax-M2.7")
	stream, err := client.CreateMessageStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q", got)
	}
	if stubM.Hits() != 1 {
		t.Fatalf("minimax hits = %d", stubM.Hits())
	}
	if stubA.Hits() != 0 {
		t.Fatalf("anthropic hits = %d (expected 0; cross-bleed)", stubA.Hits())
	}
	if stubM.LastAPIKey() != "key-mm" {
		t.Fatalf("minimax x-api-key = %q (expected key-mm)", stubM.LastAPIKey())
	}
}

// TestAnthropicMultiBackend_HeaderIsolation confirms ExtraHeaders
// declared on one backend do not leak onto another.
func TestAnthropicMultiBackend_HeaderIsolation(t *testing.T) {
	stubA := newRecordingStub(minimalAnthropicSSE)
	stubB := newRecordingStub(minimalAnthropicSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvB := httptest.NewServer(stubB.handler())
	defer srvB.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{
			APIKey:       "k1",
			BaseURL:      srvA.URL,
			ExtraHeaders: http.Header{"X-Tenant": []string{"a"}},
		}),
		WithAnthropicCompat("other", AnthropicConfig{
			APIKey:       "k2",
			BaseURL:      srvB.URL,
			ExtraHeaders: http.Header{"X-Tenant": []string{"b"}},
		}),
		WithModelRoute("other-", Route{Provider: ProviderAnthropic, Backend: "other"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Hit "anthropic" backend.
	streamA, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "claude-test"))
	if err != nil {
		t.Fatalf("stream A: %v", err)
	}
	_ = drainStream(t, streamA)
	// Hit "other" backend.
	streamB, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "other-foo"))
	if err != nil {
		t.Fatalf("stream B: %v", err)
	}
	_ = drainStream(t, streamB)

	if got := stubA.LastHeader("X-Tenant"); got != "a" {
		t.Fatalf("anthropic X-Tenant = %q", got)
	}
	if got := stubB.LastHeader("X-Tenant"); got != "b" {
		t.Fatalf("other X-Tenant = %q", got)
	}
}

// TestAnthropicMultiBackend_ErrorIsolation verifies a 500 on one
// backend does not poison subsequent calls routed to another.
func TestAnthropicMultiBackend_ErrorIsolation(t *testing.T) {
	stubA := newRecordingStub(minimalAnthropicSSE)
	stubM := newRecordingStub(minimalAnthropicSSE)
	stubM.SetStatus(http.StatusInternalServerError, "boom")
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvM := httptest.NewServer(stubM.handler())
	defer srvM.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{APIKey: "k1", BaseURL: srvA.URL}),
		WithAnthropicCompat("minimax", AnthropicConfig{APIKey: "k2", BaseURL: srvM.URL}),
		WithModelRoute("minimax-", Route{Provider: ProviderAnthropic, Backend: "minimax"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// First, a doomed call to "minimax" — must fail.
	if _, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "MiniMax-")); err == nil {
		t.Fatalf("expected error from minimax 500")
	}
	// Second, a call to "anthropic" — must succeed and produce text.
	stream, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "claude-test"))
	if err != nil {
		t.Fatalf("anthropic stream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q", got)
	}
}

// TestAnthropicMultiBackend_BackwardCompat verifies the v0.1.x sugar
// path: a single WithAnthropic registration plus the built-in
// claude-* default route resolves end-to-end without any explicit
// WithModelRoute call.
func TestAnthropicMultiBackend_BackwardCompat(t *testing.T) {
	stubA := newRecordingStub(minimalAnthropicSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{APIKey: "k1", BaseURL: srvA.URL}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "claude-3-5-sonnet"))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q", got)
	}
	if stubA.Hits() != 1 {
		t.Fatalf("hits = %d", stubA.Hits())
	}
}

// TestAnthropicMultiBackend_UnknownBackend asserts ErrProviderNotConfigured
// wraps the backend name when the route points at a missing slot.
func TestAnthropicMultiBackend_UnknownBackend(t *testing.T) {
	stubA := newRecordingStub(minimalAnthropicSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{APIKey: "k1", BaseURL: srvA.URL}),
		WithModelRoute("ghost-", Route{Provider: ProviderAnthropic, Backend: "missing"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "ghost-1"))
	if err == nil {
		t.Fatalf("expected error")
	}
	if !errors.Is(err, ErrProviderNotConfigured) {
		t.Fatalf("err = %v (not ErrProviderNotConfigured)", err)
	}
	if !strings.Contains(err.Error(), "missing") {
		t.Fatalf("err = %v (does not name backend)", err)
	}
}

// ----------------------------------------------------------------------
// OpenAI Responses parity tests (§3.7 case 6 — the same five against
// WithOpenAIResponsesCompat).
// ----------------------------------------------------------------------

// TestOpenAIResponsesMultiBackend_RoutingIsolation routes a non-default
// model to a second openai_responses backend and asserts isolation.
func TestOpenAIResponsesMultiBackend_RoutingIsolation(t *testing.T) {
	stubA := newRecordingStub(minimalOpenAIResponsesSSE)
	stubB := newRecordingStub(minimalOpenAIResponsesSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvB := httptest.NewServer(stubB.handler())
	defer srvB.Close()

	client, err := New(
		WithOpenAI(OpenAIConfig{APIKey: "sk-a", BaseURL: srvA.URL}),
		WithOpenAIResponsesCompat("alt", OpenAIResponsesConfig{APIKey: "sk-b", BaseURL: srvB.URL}),
		WithModelRoute("alt-", Route{Provider: ProviderOpenAIResponses, Backend: "alt"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "alt-foo"))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q", got)
	}
	if stubB.Hits() != 1 || stubA.Hits() != 0 {
		t.Fatalf("hits A=%d B=%d (expected 0/1)", stubA.Hits(), stubB.Hits())
	}
	if stubB.LastAuth() != "Bearer sk-b" {
		t.Fatalf("alt Authorization = %q", stubB.LastAuth())
	}
}

// TestOpenAIResponsesMultiBackend_HeaderIsolation confirms per-backend
// ExtraHeaders do not cross-bleed.
func TestOpenAIResponsesMultiBackend_HeaderIsolation(t *testing.T) {
	stubA := newRecordingStub(minimalOpenAIResponsesSSE)
	stubB := newRecordingStub(minimalOpenAIResponsesSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvB := httptest.NewServer(stubB.handler())
	defer srvB.Close()

	client, err := New(
		WithOpenAI(OpenAIConfig{
			APIKey:       "sk-a",
			BaseURL:      srvA.URL,
			ExtraHeaders: http.Header{"X-Tenant": []string{"a"}},
		}),
		WithOpenAIResponsesCompat("alt", OpenAIResponsesConfig{
			APIKey:       "sk-b",
			BaseURL:      srvB.URL,
			ExtraHeaders: http.Header{"X-Tenant": []string{"b"}},
		}),
		WithModelRoute("alt-", Route{Provider: ProviderOpenAIResponses, Backend: "alt"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	streamA, _ := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "gpt-5-mini"))
	_ = drainStream(t, streamA)
	streamB, _ := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "alt-foo"))
	_ = drainStream(t, streamB)

	if got := stubA.LastHeader("X-Tenant"); got != "a" {
		t.Fatalf("openai X-Tenant = %q", got)
	}
	if got := stubB.LastHeader("X-Tenant"); got != "b" {
		t.Fatalf("alt X-Tenant = %q", got)
	}
}

// TestOpenAIResponsesMultiBackend_ErrorIsolation parallels the
// anthropic error-isolation case.
func TestOpenAIResponsesMultiBackend_ErrorIsolation(t *testing.T) {
	stubA := newRecordingStub(minimalOpenAIResponsesSSE)
	stubB := newRecordingStub(minimalOpenAIResponsesSSE)
	stubB.SetStatus(http.StatusInternalServerError, "boom")
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvB := httptest.NewServer(stubB.handler())
	defer srvB.Close()

	client, err := New(
		WithOpenAI(OpenAIConfig{APIKey: "sk-a", BaseURL: srvA.URL}),
		WithOpenAIResponsesCompat("alt", OpenAIResponsesConfig{APIKey: "sk-b", BaseURL: srvB.URL}),
		WithModelRoute("alt-", Route{Provider: ProviderOpenAIResponses, Backend: "alt"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "alt-foo")); err == nil {
		t.Fatalf("expected error from alt 500")
	}
	stream, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "gpt-5-mini"))
	if err != nil {
		t.Fatalf("openai stream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q", got)
	}
}

// TestOpenAIResponsesMultiBackend_BackwardCompat — same shape as the
// anthropic case: WithOpenAI alone is enough for built-in gpt-* routes.
func TestOpenAIResponsesMultiBackend_BackwardCompat(t *testing.T) {
	stubA := newRecordingStub(minimalOpenAIResponsesSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()

	client, err := New(
		WithOpenAI(OpenAIConfig{APIKey: "sk-a", BaseURL: srvA.URL}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "gpt-5-mini"))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q", got)
	}
	if stubA.Hits() != 1 {
		t.Fatalf("hits = %d", stubA.Hits())
	}
}

// TestOpenAIResponsesMultiBackend_UnknownBackend mirrors the anthropic
// missing-slot case.
func TestOpenAIResponsesMultiBackend_UnknownBackend(t *testing.T) {
	stubA := newRecordingStub(minimalOpenAIResponsesSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()

	client, err := New(
		WithOpenAI(OpenAIConfig{APIKey: "sk-a", BaseURL: srvA.URL}),
		WithModelRoute("ghost-", Route{Provider: ProviderOpenAIResponses, Backend: "missing"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "ghost-1"))
	if err == nil {
		t.Fatalf("expected error")
	}
	if !errors.Is(err, ErrProviderNotConfigured) {
		t.Fatalf("err = %v (not ErrProviderNotConfigured)", err)
	}
	if !strings.Contains(err.Error(), "missing") {
		t.Fatalf("err = %v", err)
	}
}

// TestAdapterName_DefaultSugarKeepsSuffix verifies the design decision
// for Q-OPEN-3: the default-named slot returns the suffixed form
// "anthropic:anthropic" / "openai_responses:openai".
func TestAdapterName_DefaultSugarKeepsSuffix(t *testing.T) {
	stubA := newRecordingStub(minimalAnthropicSSE)
	stubO := newRecordingStub(minimalOpenAIResponsesSSE)
	srvA := httptest.NewServer(stubA.handler())
	defer srvA.Close()
	srvO := httptest.NewServer(stubO.handler())
	defer srvO.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{APIKey: "k1", BaseURL: srvA.URL}),
		WithOpenAI(OpenAIConfig{APIKey: "sk-a", BaseURL: srvO.URL}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := client.anthropics["anthropic"].Name(); got != "anthropic:anthropic" {
		t.Fatalf("anthropic name = %q (expected suffix preserved)", got)
	}
	if got := client.openais["openai"].Name(); got != "openai_responses:openai" {
		t.Fatalf("openai name = %q (expected suffix preserved)", got)
	}
}

// ----------------------------------------------------------------------
// Azure OpenAI Responses (v0.2.0 Track E) — covers the registration,
// duplicate-name guard, dispatch path, and dual-auth headers
// end-to-end through Client.New + Client.dispatch.
// ----------------------------------------------------------------------

// TestAzureOpenAI_BackendRegistered verifies the Azure backend lands in
// the same `openais` map as vanilla OpenAI, with the configured name
// surfacing through Adapter.Name. This is the symmetry decision from
// docs/design/v0.2.0-azure-responses.md §4.2.
func TestAzureOpenAI_BackendRegistered(t *testing.T) {
	stub := newRecordingStub(minimalOpenAIResponsesSSE)
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	client, err := New(
		WithAzureOpenAI("azure", AzureOpenAIConfig{
			BaseURL:    srv.URL,
			APIKey:     "k",
			APIVersion: "2025-03-01-preview",
			Deployment: "gpt-5-PTU",
		}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	a, ok := client.openais["azure"]
	if !ok {
		t.Fatalf("azure backend missing; openais=%v", keysOf(client.openais))
	}
	if got := a.Name(); got != "openai_responses:azure" {
		t.Fatalf("name = %q, want openai_responses:azure", got)
	}
}

// TestAzureOpenAI_DuplicateNameRejected enforces the namespace-sharing
// rule from docs/design/v0.2.0-azure-responses.md §4.3: registering
// the same backend name under both helpers must fail at construction
// time, not silently overwrite.
func TestAzureOpenAI_DuplicateNameRejected(t *testing.T) {
	stub := newRecordingStub(minimalOpenAIResponsesSSE)
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	_, err := New(
		WithOpenAIResponsesCompat("shared", OpenAIResponsesConfig{
			APIKey:  "sk",
			BaseURL: srv.URL,
		}),
		WithAzureOpenAI("shared", AzureOpenAIConfig{
			BaseURL:    srv.URL,
			APIKey:     "k",
			APIVersion: "2025-03-01-preview",
		}),
	)
	if err == nil {
		t.Fatal("expected duplicate-name error, got nil")
	}
	if !strings.Contains(err.Error(), "shared") || !strings.Contains(err.Error(), "Azure") {
		t.Fatalf("error = %q, expected mention of name + Azure", err.Error())
	}
}

// TestAzureOpenAI_DispatchAndDualAuth is the end-to-end happy path:
// register an Azure backend, route a model prefix to it, send a
// streamed request, drain it, and assert the upstream observed BOTH
// auth headers and the deployment-bound URL form.
func TestAzureOpenAI_DispatchAndDualAuth(t *testing.T) {
	stub := newRecordingStub(minimalOpenAIResponsesSSE)
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	client, err := New(
		WithAzureOpenAI("azure", AzureOpenAIConfig{
			BaseURL:    srv.URL,
			APIKey:     "azure-secret",
			APIVersion: "2025-03-01-preview",
			Deployment: "gpt-5-PTU",
		}),
		WithModelRoute("gpt-5", Route{
			Provider: ProviderOpenAIResponses,
			Backend:  "azure",
		}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sr, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	_ = drainStream(t, sr)

	if stub.Hits() != 1 {
		t.Fatalf("upstream hits = %d, want 1", stub.Hits())
	}
	if got := stub.LastAuth(); got != "Bearer azure-secret" {
		t.Errorf("Authorization = %q", got)
	}
	if got := stub.LastHeader("api-key"); got != "azure-secret" {
		t.Errorf("api-key header = %q", got)
	}
	// httptest.Server strips its own host so r.URL.Path is the wire
	// path. The deployment-bound URL form puts the deployment in the
	// path; api-version goes in the query (covered by the headers
	// test in adapter_headers_test.go — here we just confirm dispatch
	// arrived at the Azure-shaped path, not /v1/responses).
	if !strings.Contains(stub.lastPath, "/openai/deployments/gpt-5-PTU/responses") {
		t.Errorf("path = %q, want Azure deployment-bound shape", stub.lastPath)
	}
}

// TestAzureOpenAI_RouteUpstreamModelFallback exercises the middle
// layer of the layered model fallback from
// docs/design/v0.2.0-azure-responses.md §5.4: when cfg.Deployment is
// empty, Route.UpstreamModel rewrites req.Model upstream in
// Client.dispatch and the Azure adapter's deployment-less URL carries
// that rewritten value in the body.
func TestAzureOpenAI_RouteUpstreamModelFallback(t *testing.T) {
	// Capture the wire body to verify the rewritten model field.
	bodies := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		select {
		case bodies <- b:
		default:
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, minimalOpenAIResponsesSSE)
	}))
	defer srv.Close()

	client, err := New(
		WithAzureOpenAI("azure", AzureOpenAIConfig{
			BaseURL:    srv.URL,
			APIKey:     "k",
			APIVersion: "2025-03-01-preview",
			// Deployment intentionally unset — we rely on the
			// Route.UpstreamModel layer.
		}),
		WithModelRoute("gpt-5", Route{
			Provider:      ProviderOpenAIResponses,
			Backend:       "azure",
			UpstreamModel: "azure-deployment-name",
		}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	sr, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "gpt-5"))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	_ = drainStream(t, sr)

	select {
	case body := <-bodies:
		if !strings.Contains(string(body), `"model":"azure-deployment-name"`) {
			t.Errorf("body should carry rewritten model, got: %s", body)
		}
	default:
		t.Fatal("upstream not hit")
	}
}

// keysOf is a tiny helper for diagnostics in failure messages.
func keysOf[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// errors / io / context imports already declared above; this package
// stub keeps go vet happy if the file ever ends without using them.
var _ = errors.New
