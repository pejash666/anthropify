//go:build e2e

package e2e

import (
	"fmt"
	"strings"
	"testing"

	"github.com/pejash666/anthropify"
)

// newAnthropicClient builds a *anthropify.Client wired exclusively to
// the Anthropic adapter. The build-tag isolation means we can import
// the parent package without inflating the default build graph.
func newAnthropicClient(t *testing.T) (*anthropify.Client, string) {
	t.Helper()
	env := AnthropicEnv()
	skipIfUnset(t, env, "ANTHROPIC_API_KEY")
	cli, err := anthropify.New(
		anthropify.WithAnthropic(anthropify.AnthropicConfig{APIKey: env.APIKey}),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newOpenAIClient(t *testing.T) (*anthropify.Client, string) {
	t.Helper()
	env := OpenAIEnv()
	skipIfUnset(t, env, "OPENAI_API_KEY")
	cli, err := anthropify.New(
		anthropify.WithOpenAI(anthropify.OpenAIConfig{APIKey: env.APIKey}),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newGeminiClient(t *testing.T) (*anthropify.Client, string) {
	t.Helper()
	env := GeminiEnv()
	if !env.Configured {
		t.Skip("GEMINI_API_KEY (Studio/Express) or VERTEX_PROJECT/VERTEX_LOCATION not set; skipping E2E")
	}
	mode, err := parseGeminiMode(env.GeminiMode)
	if err != nil {
		t.Fatalf("parse GEMINI_MODE: %v", err)
	}
	cli, err := anthropify.New(
		anthropify.WithGemini(anthropify.GeminiConfig{
			Mode:     mode,
			APIKey:   env.APIKey,
			Project:  env.VertexProject,
			Location: env.VertexLocation,
		}),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

// parseGeminiMode maps a GEMINI_MODE env value to its anthropify
// constant. An empty string yields GeminiModeAuto so existing setups
// behave identically to the pre-Express harness.
func parseGeminiMode(s string) (anthropify.GeminiMode, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "":
		return anthropify.GeminiModeAuto, nil
	case "auto":
		return anthropify.GeminiModeAuto, nil
	case "studio":
		return anthropify.GeminiModeStudio, nil
	case "express":
		return anthropify.GeminiModeExpress, nil
	case "vertex":
		return anthropify.GeminiModeVertex, nil
	default:
		return 0, fmt.Errorf("unknown GEMINI_MODE %q (want auto/studio/express/vertex)", s)
	}
}

// newChatCompletionsClient is the common path for OpenAI-compatible
// backends (Kimi, GLM). The `backend` string must match the model-route
// rule installed below.
func newChatCompletionsClient(t *testing.T, env ProviderEnv, backend, envVarName string) (*anthropify.Client, string) {
	t.Helper()
	skipIfUnset(t, env, envVarName)
	cli, err := anthropify.New(
		anthropify.WithChatCompletion(backend, anthropify.OpenAICompatConfig{
			BaseURL: env.BaseURL,
			APIKey:  env.APIKey,
		}),
		// The default router only knows about claude-/gpt-/gemini-.
		// Map the provider's model prefix to its chat_completions
		// backend so dispatch finds the adapter.
		anthropify.WithModelOverride(func(model string) (anthropify.Route, bool) {
			return anthropify.Route{
				Provider: anthropify.ProviderChatCompletions,
				Backend:  backend,
			}, true
		}),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newKimiClient(t *testing.T) (*anthropify.Client, string) {
	return newChatCompletionsClient(t, KimiEnv(), "kimi", "KIMI_API_KEY")
}

func newGLMClient(t *testing.T) (*anthropify.Client, string) {
	return newChatCompletionsClient(t, GLMEnv(), "glm", "GLM_API_KEY")
}

// newMiniMaxClient builds a *anthropify.Client wired to the MiniMax
// Anthropic-compat backend. MiniMax shares the anthropic adapter with
// real Claude — the only difference is BaseURL — so we register it via
// WithAnthropicCompat under the "minimax" name and pin the MiniMax-
// model prefix to that backend with WithModelRoute.
func newMiniMaxClient(t *testing.T) (*anthropify.Client, string) {
	t.Helper()
	env := MiniMaxEnv()
	skipIfUnset(t, env, "MINIMAX_API_KEY")
	cli, err := anthropify.New(
		anthropify.WithAnthropicCompat("minimax", anthropify.AnthropicConfig{
			APIKey:  env.APIKey,
			BaseURL: env.BaseURL,
		}),
		anthropify.WithModelRoute("MiniMax-", anthropify.Route{
			Provider: anthropify.ProviderAnthropic,
			Backend:  "minimax",
		}),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}
