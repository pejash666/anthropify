package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
)

func TestAdapter_Invoke(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "secret" {
			t.Errorf("missing api key: %v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"claude-3-5-sonnet-20241022"`) {
			t.Errorf("body missing model: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	a, err := New("anthropic", Config{APIKey: "secret", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}

	src := []byte(`{"model":"claude-3-5-sonnet-20241022","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}

	msg, err := a.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if msg.ID != "msg_1" {
		t.Fatalf("id = %q", msg.ID)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content len = %d", len(msg.Content))
	}
}

func TestAdapter_Stream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		events := []string{
			`event: message_start` + "\n" + `data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}` + "\n\n",
			`event: content_block_start` + "\n" + `data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n",
			`event: content_block_delta` + "\n" + `data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n",
			`event: content_block_stop` + "\n" + `data: {"type":"content_block_stop","index":0}` + "\n\n",
			`event: message_stop` + "\n" + `data: {"type":"message_stop"}` + "\n\n",
		}
		for _, e := range events {
			_, _ = io.WriteString(w, e)
			w.(http.Flusher).Flush()
		}
	}))
	defer srv.Close()

	a, err := New("anthropic", Config{APIKey: "s", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}

	src := []byte(`{"model":"claude-3-5-sonnet-20241022","max_tokens":10,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	var req anthropicsdk.MessageNewParams
	if err := json.Unmarshal(src, &req); err != nil {
		t.Fatal(err)
	}

	ch, err := a.Stream(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	var types []string
	for evt := range ch {
		if evt.Err != nil {
			t.Fatalf("stream err: %v", evt.Err)
		}
		var m map[string]any
		_ = json.Unmarshal(evt.Data, &m)
		types = append(types, m["type"].(string))
	}
	want := "message_start,content_block_start,content_block_delta,content_block_stop,message_stop"
	if strings.Join(types, ",") != want {
		t.Fatalf("types = %v, want %s", types, want)
	}
}
