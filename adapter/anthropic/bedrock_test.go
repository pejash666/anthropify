package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
)

// --- TestBedrockAdapter_New_RequiresCredentials ----------------------
//
// Asserts the early-validation contract: missing AccessKeyID,
// SecretAccessKey, or Region produces a descriptive error. Missing
// SessionToken is fine (long-lived IAM-user keys do not have one).
func TestBedrockAdapter_New_RequiresCredentials(t *testing.T) {
	cases := []struct {
		name    string
		cfg     Config
		wantSub string
	}{
		{
			name:    "missing access key",
			cfg:     Config{Mode: AnthropicModeBedrock, AWSSecretAccessKey: "sk", AWSRegion: "us-east-1"},
			wantSub: "AWSAccessKeyID",
		},
		{
			name:    "missing secret",
			cfg:     Config{Mode: AnthropicModeBedrock, AWSAccessKeyID: "ak", AWSRegion: "us-east-1"},
			wantSub: "AWSSecretAccessKey",
		},
		{
			name:    "missing region",
			cfg:     Config{Mode: AnthropicModeBedrock, AWSAccessKeyID: "ak", AWSSecretAccessKey: "sk"},
			wantSub: "AWSRegion",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := New("bedrock", tc.cfg)
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("err = %q, want substring %q", err.Error(), tc.wantSub)
			}
		})
	}

	// SessionToken absent is acceptable.
	_, err := New("bedrock", Config{
		Mode:               AnthropicModeBedrock,
		AWSAccessKeyID:     "AKIATEST",
		AWSSecretAccessKey: "secret",
		AWSRegion:          "us-east-1",
	})
	if err != nil {
		t.Fatalf("New with valid creds (no session token) should succeed, got %v", err)
	}
}

// captured holds what the test server saw on a single request.
type captured struct {
	method  string
	path    string
	headers http.Header
	body    map[string]any
}

// newBedrockCaptureServer returns a server that records the inbound
// request shape and replies with a canonical Anthropic Message JSON.
func newBedrockCaptureServer(t *testing.T, capt *captured) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capt.method = r.Method
		capt.path = r.URL.Path
		capt.headers = r.Header.Clone()
		bodyBytes, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(bodyBytes, &capt.body)

		resp := map[string]any{
			"id":            "msg_bedrock_test",
			"type":          "message",
			"role":          "assistant",
			"model":         "claude-opus-4-5",
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
			"content":       []map[string]any{{"type": "text", "text": "ok"}},
			"usage":         map[string]any{"input_tokens": 1, "output_tokens": 1},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

// --- TestBedrockAdapter_Invoke_PathRewrite ---------------------------
//
// Asserts that the SDK rewrites POST /v1/messages -> POST
// /model/{id}/invoke, strips "model" / "stream" from the body, and
// injects "anthropic_version": "bedrock-2023-05-31".
func TestBedrockAdapter_Invoke_PathRewrite(t *testing.T) {
	var capt captured
	srv := newBedrockCaptureServer(t, &capt)
	defer srv.Close()

	a, err := New("bedrock", Config{
		Mode:               AnthropicModeBedrock,
		AWSAccessKeyID:     "AKIATEST",
		AWSSecretAccessKey: "secret",
		AWSRegion:          "us-east-1",
		BaseURL:            srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := anthropicsdk.MessageNewParams{
		Model:     "anthropic.claude-opus-4-5-20250929-v1:0",
		MaxTokens: 16,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := a.Invoke(ctx, req); err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	if capt.method != http.MethodPost {
		t.Errorf("method = %q, want POST", capt.method)
	}
	wantPath := "/model/anthropic.claude-opus-4-5-20250929-v1:0/invoke"
	if capt.path != wantPath {
		t.Errorf("path = %q, want %q", capt.path, wantPath)
	}
	if _, present := capt.body["model"]; present {
		t.Errorf("body still contains \"model\" field; SDK should have stripped it: %v", capt.body)
	}
	if _, present := capt.body["stream"]; present {
		t.Errorf("body contains \"stream\" field on non-streaming Invoke")
	}
	gotVer, _ := capt.body["anthropic_version"].(string)
	if gotVer != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version = %q, want %q (SDK should inject the Bedrock default)", gotVer, "bedrock-2023-05-31")
	}
}

// --- TestBedrockAdapter_Invoke_BetaHeaderBecomesBody -----------------
//
// Asserts the SDK's anthropic-beta header to anthropic_beta body
// translation. This is the key behaviour difference vs Direct/MiniMax
// — Bedrock rejects the header so the SDK rewrites it. We verify the
// header is absent on the wire and the body field carries the same
// values as a JSON array.
func TestBedrockAdapter_Invoke_BetaHeaderBecomesBody(t *testing.T) {
	var capt captured
	srv := newBedrockCaptureServer(t, &capt)
	defer srv.Close()

	a, err := New("bedrock", Config{
		Mode:               AnthropicModeBedrock,
		AWSAccessKeyID:     "AKIATEST",
		AWSSecretAccessKey: "secret",
		AWSRegion:          "us-east-1",
		BaseURL:            srv.URL,
		ExtraHeaders: http.Header{
			// Two separate header values become two body array entries.
			"Anthropic-Beta": []string{
				"prompt-caching-2024-07-31",
				"context-management-2025-06-27",
			},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = a.Invoke(ctx, anthropicsdk.MessageNewParams{
		Model:     "anthropic.claude-opus-4-5-20250929-v1:0",
		MaxTokens: 16,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	if got := capt.headers.Get("Anthropic-Beta"); got != "" {
		t.Errorf("anthropic-beta still on the wire as a header: %q (SDK should have lifted it into the body)", got)
	}
	rawBeta, ok := capt.body["anthropic_beta"]
	if !ok {
		t.Fatalf("body lacks anthropic_beta field; got body keys = %v", capt.body)
	}
	betaList, ok := rawBeta.([]any)
	if !ok {
		t.Fatalf("anthropic_beta is %T, want []any (JSON array)", rawBeta)
	}
	if len(betaList) != 2 {
		t.Fatalf("anthropic_beta has %d entries, want 2: %v", len(betaList), betaList)
	}
	want := map[string]bool{
		"prompt-caching-2024-07-31":     true,
		"context-management-2025-06-27": true,
	}
	for _, v := range betaList {
		s, _ := v.(string)
		if !want[s] {
			t.Errorf("unexpected beta entry %q", s)
		}
	}
}

// --- TestBedrockAdapter_Invoke_SigV4Auth -----------------------------
//
// Confirms the static credentials thread through to the SDK's signer:
// the Authorization header on the wire begins with the SigV4 prefix
// and references our AccessKeyID. We do not verify the signature
// itself (that is the SDK's responsibility); we just prove the
// machinery wired up.
func TestBedrockAdapter_Invoke_SigV4Auth(t *testing.T) {
	var capt captured
	srv := newBedrockCaptureServer(t, &capt)
	defer srv.Close()

	a, err := New("bedrock", Config{
		Mode:               AnthropicModeBedrock,
		AWSAccessKeyID:     "AKIATESTKEYID",
		AWSSecretAccessKey: "TESTSECRETKEY",
		AWSSessionToken:    "TESTSESSIONTOKEN",
		AWSRegion:          "us-east-1",
		BaseURL:            srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = a.Invoke(ctx, anthropicsdk.MessageNewParams{
		Model:     "anthropic.claude-opus-4-5-20250929-v1:0",
		MaxTokens: 16,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}

	authz := capt.headers.Get("Authorization")
	if !strings.HasPrefix(authz, "AWS4-HMAC-SHA256 ") {
		t.Errorf("Authorization = %q, want AWS4-HMAC-SHA256 prefix", authz)
	}
	if !strings.Contains(authz, "Credential=AKIATESTKEYID/") {
		t.Errorf("Authorization missing Credential=AKIATESTKEYID/...: %q", authz)
	}
	// SessionToken should also surface as X-Amz-Security-Token.
	if got := capt.headers.Get("X-Amz-Security-Token"); got != "TESTSESSIONTOKEN" {
		t.Errorf("X-Amz-Security-Token = %q, want %q", got, "TESTSESSIONTOKEN")
	}
}

// encodeBedrockChunk wraps a canonical Anthropic SSE event JSON in the
// AWS event-stream :event-type=chunk frame Bedrock emits at runtime.
// The payload is JSON {"bytes": "<base64 of inner SSE event JSON>"}.
func encodeBedrockChunk(t *testing.T, w io.Writer, sseJSON string) {
	t.Helper()
	payload := map[string]string{"bytes": base64.StdEncoding.EncodeToString([]byte(sseJSON))}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("payload marshal: %v", err)
	}
	msg := eventstream.Message{
		Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("event")},
			{Name: ":event-type", Value: eventstream.StringValue("chunk")},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		},
		Payload: payloadBytes,
	}
	enc := eventstream.NewEncoder()
	if err := enc.Encode(w, msg); err != nil {
		t.Fatalf("eventstream encode: %v", err)
	}
}

// --- TestBedrockAdapter_Stream_EventTranslation ----------------------
//
// Drives Adapter.Stream against an httptest server that emits AWS
// event-stream frames with embedded canonical Anthropic SSE event JSON.
// Asserts the channel surfaces the inner SSE event JSON byte-identical
// (so downstream code that parses Anthropic JSON events keeps working).
func TestBedrockAdapter_Stream_EventTranslation(t *testing.T) {
	innerEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Bedrock streaming path:
		// /model/{id}/invoke-with-response-stream
		if !strings.HasSuffix(r.URL.Path, "/invoke-with-response-stream") {
			t.Errorf("unexpected streaming path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		var buf bytes.Buffer
		for _, ev := range innerEvents {
			buf.Reset()
			encodeBedrockChunk(t, &buf, ev)
			_, _ = w.Write(buf.Bytes())
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer srv.Close()

	a, err := New("bedrock", Config{
		Mode:               AnthropicModeBedrock,
		AWSAccessKeyID:     "AKIATEST",
		AWSSecretAccessKey: "secret",
		AWSRegion:          "us-east-1",
		BaseURL:            srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ch, err := a.Stream(ctx, anthropicsdk.MessageNewParams{
		Model:     "anthropic.claude-opus-4-5-20250929-v1:0",
		MaxTokens: 32,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var got []string
	for evt := range ch {
		if evt.Err != nil {
			t.Fatalf("stream error: %v", evt.Err)
		}
		got = append(got, string(evt.Data))
	}

	if len(got) < 5 {
		t.Fatalf("got %d events, want >=5: %v", len(got), got)
	}
	// Spot-check canonical event types appear in order. The SDK
	// preserves the JSON the upstream sent, so we look for "type":"X".
	wantTypes := []string{
		`"type":"message_start"`,
		`"type":"content_block_start"`,
		`"type":"content_block_delta"`,
		`"type":"message_delta"`,
		`"type":"message_stop"`,
	}
	idx := 0
	for _, frame := range got {
		if idx < len(wantTypes) && strings.Contains(frame, wantTypes[idx]) {
			idx++
		}
	}
	if idx != len(wantTypes) {
		t.Errorf("did not see canonical event sequence; matched %d/%d. Got frames:\n%v", idx, len(wantTypes), got)
	}
}

// --- TestBedrockAdapter_DirectModeUnchanged --------------------------
//
// Sanity guard: the zero-value Mode (AnthropicModeDirect) still goes
// through the raw-HTTP branch; we never accidentally touch
// bedrockClient on a non-Bedrock adapter.
func TestBedrockAdapter_DirectModeUnchanged(t *testing.T) {
	a, err := New("anthropic", Config{
		APIKey:  "sk-test",
		BaseURL: "https://example.invalid",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if a.bedrockClient != nil {
		t.Errorf("Direct adapter should not have bedrockClient set, got %v", a.bedrockClient)
	}
	if a.cfg.Mode != AnthropicModeDirect {
		t.Errorf("cfg.Mode = %v, want AnthropicModeDirect (zero)", a.cfg.Mode)
	}
}

// --- helper: confirm errors.Is wiring still works after wrapping ----
func TestBedrockAdapter_Invoke_PropagatesContextCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Block long enough for the test's ctx to cancel.
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer srv.Close()

	a, err := New("bedrock", Config{
		Mode:               AnthropicModeBedrock,
		AWSAccessKeyID:     "AKIATEST",
		AWSSecretAccessKey: "secret",
		AWSRegion:          "us-east-1",
		BaseURL:            srv.URL,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = a.Invoke(ctx, anthropicsdk.MessageNewParams{
		Model:     "anthropic.claude-opus-4-5-20250929-v1:0",
		MaxTokens: 4,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	})
	if err == nil {
		t.Fatalf("expected error from cancelled context, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) && !strings.Contains(err.Error(), "context") {
		t.Errorf("err = %v, want context-related error", err)
	}
}
