//go:build e2e

// Package e2e drives real-network smoke tests against every configured
// provider. The package is excluded from default builds via the `e2e`
// build tag; nothing in here is compiled by `go test ./...`.
//
// Configuration is read from the process environment. The companion
// `make test-e2e` target sources `.env.e2e` into the environment before
// invoking `go test`. We deliberately avoid godotenv (or any new
// dependency) here.
package e2e

import (
	"os"
	"strings"
	"testing"
)

// envProvider names every provider the matrix knows about. The string
// value is used in skip messages and as the sub-test name.
type envProvider string

const (
	ProviderAnthropic envProvider = "anthropic"
	ProviderOpenAI    envProvider = "openai_responses"
	ProviderGemini    envProvider = "gemini_native"
	ProviderKimi      envProvider = "chat_completions_kimi"
	ProviderGLM       envProvider = "chat_completions_glm"
)

// ProviderEnv carries the resolved environment for one provider. A
// zero-valued struct means "no key configured; tests must skip".
type ProviderEnv struct {
	// Configured is true when the provider has enough env vars to call
	// the upstream. Tests should `t.Skip` when this is false.
	Configured bool
	// Model is the upstream model id to send (e.g. "gemini-2.5-flash").
	Model string
	// APIKey is the bearer token (empty for Vertex auth).
	APIKey string
	// BaseURL, when non-empty, overrides the default upstream.
	BaseURL string
	// VertexProject / VertexLocation populate Gemini Vertex auth when
	// GEMINI_API_KEY is unset.
	VertexProject  string
	VertexLocation string
}

// AnthropicEnv reads ANTHROPIC_API_KEY + ANTHROPIC_MODEL.
func AnthropicEnv() ProviderEnv {
	key := os.Getenv("ANTHROPIC_API_KEY")
	return ProviderEnv{
		Configured: key != "",
		APIKey:     key,
		Model:      envOr("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
	}
}

// OpenAIEnv reads OPENAI_API_KEY + OPENAI_MODEL.
func OpenAIEnv() ProviderEnv {
	key := os.Getenv("OPENAI_API_KEY")
	return ProviderEnv{
		Configured: key != "",
		APIKey:     key,
		Model:      envOr("OPENAI_MODEL", "gpt-5-mini"),
	}
}

// GeminiEnv accepts either GEMINI_API_KEY (AI Studio) or the
// VERTEX_PROJECT/VERTEX_LOCATION pair (Vertex AI). Vertex auth still
// requires application-default credentials at runtime; the harness
// itself only forwards the values to the adapter.
func GeminiEnv() ProviderEnv {
	key := os.Getenv("GEMINI_API_KEY")
	proj := os.Getenv("VERTEX_PROJECT")
	loc := os.Getenv("VERTEX_LOCATION")
	configured := key != "" || (proj != "" && loc != "")
	return ProviderEnv{
		Configured:     configured,
		APIKey:         key,
		Model:          envOr("GEMINI_MODEL", "gemini-2.5-flash"),
		VertexProject:  proj,
		VertexLocation: loc,
	}
}

// KimiEnv reads KIMI_API_KEY / KIMI_BASE_URL / KIMI_MODEL.
func KimiEnv() ProviderEnv {
	key := os.Getenv("KIMI_API_KEY")
	return ProviderEnv{
		Configured: key != "",
		APIKey:     key,
		BaseURL:    envOr("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
		Model:      envOr("KIMI_MODEL", "kimi-k2"),
	}
}

// GLMEnv reads GLM_API_KEY / GLM_BASE_URL / GLM_MODEL.
func GLMEnv() ProviderEnv {
	key := os.Getenv("GLM_API_KEY")
	return ProviderEnv{
		Configured: key != "",
		APIKey:     key,
		BaseURL:    envOr("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
		Model:      envOr("GLM_MODEL", "glm-4.6"),
	}
}

// RecordFixtures returns true when E2E_RECORD_FIXTURES is set to a
// truthy value (1/true/yes). The fixture-recording hook is currently a
// reserved no-op; see `helpers/assertions.go` for the TODO.
func RecordFixtures() bool {
	v := strings.ToLower(os.Getenv("E2E_RECORD_FIXTURES"))
	return v == "1" || v == "true" || v == "yes"
}

// skipIfUnset is the canonical guard. Each Test* function calls this at
// the top so that missing keys produce a skipped (not failed) run.
func skipIfUnset(t *testing.T, p ProviderEnv, varName string) {
	t.Helper()
	if !p.Configured {
		t.Skipf("%s not set; skipping E2E", varName)
	}
}

func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}
