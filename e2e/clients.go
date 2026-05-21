//go:build e2e

package e2e

import (
	"testing"

	"github.com/shahao/hybridstream"
)

// newAnthropicClient builds a *hybridstream.Client wired exclusively to
// the Anthropic adapter. The build-tag isolation means we can import
// the parent package without inflating the default build graph.
func newAnthropicClient(t *testing.T) (*hybridstream.Client, string) {
	t.Helper()
	env := AnthropicEnv()
	skipIfUnset(t, env, "ANTHROPIC_API_KEY")
	cli, err := hybridstream.New(
		hybridstream.WithAnthropic(hybridstream.AnthropicConfig{APIKey: env.APIKey}),
	)
	if err != nil {
		t.Fatalf("hybridstream.New: %v", err)
	}
	return cli, env.Model
}

func newOpenAIClient(t *testing.T) (*hybridstream.Client, string) {
	t.Helper()
	env := OpenAIEnv()
	skipIfUnset(t, env, "OPENAI_API_KEY")
	cli, err := hybridstream.New(
		hybridstream.WithOpenAI(hybridstream.OpenAIConfig{APIKey: env.APIKey}),
	)
	if err != nil {
		t.Fatalf("hybridstream.New: %v", err)
	}
	return cli, env.Model
}

func newGeminiClient(t *testing.T) (*hybridstream.Client, string) {
	t.Helper()
	env := GeminiEnv()
	if !env.Configured {
		t.Skip("GEMINI_API_KEY (AI Studio) or VERTEX_PROJECT/VERTEX_LOCATION not set; skipping E2E")
	}
	cli, err := hybridstream.New(
		hybridstream.WithGemini(hybridstream.GeminiConfig{
			APIKey:   env.APIKey,
			Project:  env.VertexProject,
			Location: env.VertexLocation,
		}),
	)
	if err != nil {
		t.Fatalf("hybridstream.New: %v", err)
	}
	return cli, env.Model
}

// newChatCompletionsClient is the common path for OpenAI-compatible
// backends (Kimi, GLM). The `backend` string must match the model-route
// rule installed below.
func newChatCompletionsClient(t *testing.T, env ProviderEnv, backend, envVarName string) (*hybridstream.Client, string) {
	t.Helper()
	skipIfUnset(t, env, envVarName)
	cli, err := hybridstream.New(
		hybridstream.WithChatCompletion(backend, hybridstream.OpenAICompatConfig{
			BaseURL: env.BaseURL,
			APIKey:  env.APIKey,
		}),
		// The default router only knows about claude-/gpt-/gemini-.
		// Map the provider's model prefix to its chat_completions
		// backend so dispatch finds the adapter.
		hybridstream.WithModelOverride(func(model string) (hybridstream.Route, bool) {
			return hybridstream.Route{
				Provider: hybridstream.ProviderChatCompletions,
				Backend:  backend,
			}, true
		}),
	)
	if err != nil {
		t.Fatalf("hybridstream.New: %v", err)
	}
	return cli, env.Model
}

func newKimiClient(t *testing.T) (*hybridstream.Client, string) {
	return newChatCompletionsClient(t, KimiEnv(), "kimi", "KIMI_API_KEY")
}

func newGLMClient(t *testing.T) (*hybridstream.Client, string) {
	return newChatCompletionsClient(t, GLMEnv(), "glm", "GLM_API_KEY")
}
