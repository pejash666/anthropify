package anthropify

// Top-level Anthropic cache_control support.
//
// Anthropic's Messages API supports two prompt-caching modes
// (https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
//
//  1. Automatic caching - a single cache_control object at the top
//     level of the request. The system applies the cache breakpoint to
//     the last cacheable block and moves it forward as the
//     conversation grows. Best for multi-turn chats where the message
//     history should be cached automatically.
//
//  2. Explicit cache breakpoints - a cache_control object placed
//     directly on individual content blocks (system, messages, tools)
//     for fine-grained control over what gets cached.
//
// This file adds a typed canonical Go API for mode (1). Mode (2) is
// already supported by the underlying SDK via per-block
// SetExtraFields and is unaffected by anything here.
//
// On adapter/anthropic (which also handles MiniMax via
// WithAnthropicCompat) the canonical CacheControl is translated to the
// wire field {"cache_control":{"type":"ephemeral"}}. Every other
// adapter (openai_responses, gemini_native, chat_completions) builds
// its wire payload from typed fields only and intentionally ignores
// CacheControl, because those upstreams cache automatically
// server-side and have no analogous request field.

// CacheControlType discriminates a cache_control entry. Currently the
// only legal value is CacheControlEphemeral.
type CacheControlType string

const (
	// CacheControlEphemeral asks the upstream to cache the longest
	// cacheable prefix for ~5 minutes. Equivalent to the JSON value
	// {"type":"ephemeral"} placed at the top level of the request.
	CacheControlEphemeral CacheControlType = "ephemeral"
)

// CacheControl is the canonical top-level prompt-caching toggle.
//
// Effects per backend:
//   - anthropic / anthropic-compat (incl. MiniMax): emitted as
//     {"cache_control":{"type":"ephemeral"}} at the request top level.
//   - openai_responses, gemini_native, chat_completions: documented
//     no-op; those backends auto-cache server-side.
//
// Per-block manual cache_control breakpoints set on individual
// content blocks are unaffected and continue to work in both modes.
type CacheControl struct {
	Type CacheControlType `json:"type"`
}

// SetCacheControl installs (or, when cc.Type == "", clears) a
// top-level cache_control directive on req. It writes via the SDK's
// param.metadata.SetExtraFields, so the canonical struct stays
// dependency-free of anthropify-private state and any other extra
// fields the caller has installed are preserved.
//
// Calling SetCacheControl is idempotent and safe to call repeatedly.
//
// v0.2.0 only ships {Type: CacheControlEphemeral} (the default
// 5-minute TTL). The 1-hour TTL form is gated behind a beta header
// upstream and will be added in a later release.
func SetCacheControl(req *MessageNewParams, cc CacheControl) {
	if req == nil {
		return
	}
	extras := req.ExtraFields()
	out := make(map[string]any, len(extras)+1)
	for k, v := range extras {
		out[k] = v
	}
	if cc.Type == "" {
		delete(out, "cache_control")
	} else {
		out["cache_control"] = map[string]any{"type": string(cc.Type)}
	}
	req.SetExtraFields(out)
}

// GetCacheControl returns the currently-installed top-level
// cache_control directive on req, if any. Useful for tests and for
// adapters that want to peek (only the anthropic adapter does today).
//
// The second return is false when no top-level cache_control has
// been installed via SetCacheControl, or when the installed value
// has an unrecognised shape.
func GetCacheControl(req MessageNewParams) (CacheControl, bool) {
	extras := req.ExtraFields()
	if extras == nil {
		return CacheControl{}, false
	}
	raw, ok := extras["cache_control"]
	if !ok {
		return CacheControl{}, false
	}
	m, ok := raw.(map[string]any)
	if !ok {
		return CacheControl{}, false
	}
	t, ok := m["type"].(string)
	if !ok {
		return CacheControl{}, false
	}
	return CacheControl{Type: CacheControlType(t)}, true
}
