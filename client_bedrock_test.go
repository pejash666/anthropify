package anthropify

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
)

// ----------------------------------------------------------------------
// TestAnthropicMultiBackend_BedrockRouting
// ----------------------------------------------------------------------
//
// Top-level proof that WithAnthropicBedrock wires through the full
// pipeline: WithModelRoute("claude-opus-4-5", ...{Backend:"bedrock",
// UpstreamModel:"anthropic.claude-opus-4-5-20250929-v1:0"}) sends the
// request to the Bedrock stub server (not the Direct stub), with the
// upstream model name rewritten in the URL path and SigV4 credentials
// on the Authorization header.
func TestAnthropicMultiBackend_BedrockRouting(t *testing.T) {
	directStub := newRecordingStub(minimalAnthropicSSE)
	srvDirect := httptest.NewServer(directStub.handler())
	defer srvDirect.Close()

	var bedrockHits int32
	var bedrockPath atomic.Value
	var bedrockAuthz atomic.Value
	srvBedrock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&bedrockHits, 1)
		bedrockPath.Store(r.URL.Path)
		bedrockAuthz.Store(r.Header.Get("Authorization"))
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		var buf bytes.Buffer
		for _, ev := range []string{
			`{"type":"message_start","message":{"id":"msg_b","type":"message","role":"assistant","model":"claude-opus-4-5","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":""},"usage":{"input_tokens":1,"output_tokens":2}}`,
			`{"type":"message_stop"}`,
		} {
			buf.Reset()
			writeBedrockChunk(t, &buf, ev)
			_, _ = w.Write(buf.Bytes())
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer srvBedrock.Close()

	client, err := New(
		WithAnthropic(AnthropicConfig{APIKey: "key-direct", BaseURL: srvDirect.URL}),
		WithAnthropicBedrock("bedrock", BedrockConfig{
			AccessKeyID:     "AKIATESTBEDROCK",
			SecretAccessKey: "secret",
			Region:          "us-east-1",
			BaseURL:         srvBedrock.URL,
		}),
		WithModelRoute("claude-opus-4-5", Route{
			Provider:      ProviderAnthropic,
			Backend:       "bedrock",
			UpstreamModel: "anthropic.claude-opus-4-5-20250929-v1:0",
		}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stream, err := client.CreateMessageStream(context.Background(), makeAnthropicReq(t, "claude-opus-4-5"))
	if err != nil {
		t.Fatalf("CreateMessageStream: %v", err)
	}
	if got := drainStream(t, stream); got != "hi" {
		t.Fatalf("text = %q, want %q", got, "hi")
	}

	if got := atomic.LoadInt32(&bedrockHits); got != 1 {
		t.Fatalf("bedrock hits = %d, want 1", got)
	}
	if directStub.Hits() != 0 {
		t.Fatalf("direct hits = %d, want 0 (cross-bleed)", directStub.Hits())
	}
	if path, _ := bedrockPath.Load().(string); path != "/model/anthropic.claude-opus-4-5-20250929-v1:0/invoke-with-response-stream" {
		t.Fatalf("bedrock path = %q, want /model/.../invoke-with-response-stream", path)
	}
	authz, _ := bedrockAuthz.Load().(string)
	if !strings.HasPrefix(authz, "AWS4-HMAC-SHA256 ") {
		t.Errorf("Authorization = %q, want AWS4-HMAC-SHA256 prefix", authz)
	}
	if !strings.Contains(authz, "Credential=AKIATESTBEDROCK/") {
		t.Errorf("Authorization missing Credential=AKIATESTBEDROCK/...: %q", authz)
	}
}

// ----------------------------------------------------------------------
// TestAnthropicMultiBackend_BedrockNameCollision
// ----------------------------------------------------------------------
//
// Asserts that registering the same backend name through both
// WithAnthropicCompat and WithAnthropicBedrock fails fast at New()
// rather than silently overwriting one with the other.
func TestAnthropicMultiBackend_BedrockNameCollision(t *testing.T) {
	_, err := New(
		WithAnthropicCompat("dup", AnthropicConfig{APIKey: "k", BaseURL: "https://x.invalid"}),
		WithAnthropicBedrock("dup", BedrockConfig{
			AccessKeyID:     "AKIATEST",
			SecretAccessKey: "s",
			Region:          "us-east-1",
		}),
	)
	if err == nil {
		t.Fatalf("expected name collision error, got nil")
	}
	if !strings.Contains(err.Error(), "Direct and a Bedrock") {
		t.Errorf("err = %q, want collision message mentioning Direct/Bedrock", err.Error())
	}
}

// writeBedrockChunk encodes a canonical Anthropic SSE event JSON in
// the AWS event-stream :event-type=chunk frame Bedrock emits at
// runtime. Mirrors helper in adapter/anthropic/bedrock_test.go but is
// duplicated here so this file stays runnable in isolation.
func writeBedrockChunk(t *testing.T, w io.Writer, sseJSON string) {
	t.Helper()
	payload, err := json.Marshal(map[string]string{
		"bytes": base64.StdEncoding.EncodeToString([]byte(sseJSON)),
	})
	if err != nil {
		t.Fatalf("payload marshal: %v", err)
	}
	msg := eventstream.Message{
		Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("event")},
			{Name: ":event-type", Value: eventstream.StringValue("chunk")},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		},
		Payload: payload,
	}
	if err := eventstream.NewEncoder().Encode(w, msg); err != nil {
		t.Fatalf("eventstream encode: %v", err)
	}
}
