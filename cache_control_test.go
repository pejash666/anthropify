package anthropify

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// TestSetCacheControl_RoundTrip exercises the happy path: install an
// ephemeral cache_control and verify it lands at the JSON top level
// after json.Marshal.
func TestSetCacheControl_RoundTrip(t *testing.T) {
	req := MessageNewParams{
		Model:     anthropic.Model("claude-3-5-sonnet-20241022"),
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	}
	SetCacheControl(&req, CacheControl{Type: CacheControlEphemeral})

	got, ok := GetCacheControl(req)
	if !ok || got.Type != CacheControlEphemeral {
		t.Fatalf("GetCacheControl after Set: got=%+v ok=%v", got, ok)
	}

	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var top map[string]any
	if err := json.Unmarshal(body, &top); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	cc, ok := top["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("wire body missing cache_control object: %s", body)
	}
	if cc["type"] != "ephemeral" {
		t.Fatalf("cache_control.type = %v, want ephemeral", cc["type"])
	}
	// Sanity: the original required fields must still serialise.
	if top["model"] != "claude-3-5-sonnet-20241022" {
		t.Fatalf("model dropped from body: %s", body)
	}
	if _, ok := top["messages"].([]any); !ok {
		t.Fatalf("messages dropped from body: %s", body)
	}
}

// TestSetCacheControl_Clear shows that passing an empty Type removes
// any previously-installed top-level cache_control.
func TestSetCacheControl_Clear(t *testing.T) {
	req := MessageNewParams{
		Model:     anthropic.Model("claude-3-5-sonnet-20241022"),
		MaxTokens: 32,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	}
	SetCacheControl(&req, CacheControl{Type: CacheControlEphemeral})
	SetCacheControl(&req, CacheControl{}) // clear

	if got, ok := GetCacheControl(req); ok {
		t.Fatalf("expected no cache_control after clear, got %+v", got)
	}
	body, _ := json.Marshal(req)
	var top map[string]any
	_ = json.Unmarshal(body, &top)
	if _, ok := top["cache_control"]; ok {
		t.Fatalf("cache_control still present after clear: %s", body)
	}
}

// TestSetCacheControl_PreservesOtherExtras proves SetCacheControl
// does not clobber unrelated extra fields the caller may have set
// (e.g. via a future metadata injection).
func TestSetCacheControl_PreservesOtherExtras(t *testing.T) {
	req := MessageNewParams{
		Model:     anthropic.Model("claude-3-5-sonnet-20241022"),
		MaxTokens: 32,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
		},
	}
	req.SetExtraFields(map[string]any{"foo": "bar"})
	SetCacheControl(&req, CacheControl{Type: CacheControlEphemeral})

	body, _ := json.Marshal(req)
	var top map[string]any
	_ = json.Unmarshal(body, &top)
	if top["foo"] != "bar" {
		t.Fatalf("unrelated extra foo dropped: %s", body)
	}
	cc, ok := top["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" {
		t.Fatalf("cache_control missing or wrong: %s", body)
	}
}

// TestSetCacheControl_NilSafe protects against accidental nil
// pointer derefs.
func TestSetCacheControl_NilSafe(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("SetCacheControl(nil) panicked: %v", r)
		}
	}()
	SetCacheControl(nil, CacheControl{Type: CacheControlEphemeral})
}

// TestGetCacheControl_AbsentReturnsFalse confirms the read helper
// returns ok=false for a fresh request that never went through
// SetCacheControl.
func TestGetCacheControl_AbsentReturnsFalse(t *testing.T) {
	req := MessageNewParams{
		Model:     anthropic.Model("claude-3-5-sonnet-20241022"),
		MaxTokens: 32,
	}
	if cc, ok := GetCacheControl(req); ok {
		t.Fatalf("expected ok=false on fresh req, got %+v", cc)
	}
}
