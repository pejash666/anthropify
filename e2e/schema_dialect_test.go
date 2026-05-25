//go:build e2e

package e2e

import (
	"context"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/anthropify"
	"github.com/shahao/anthropify/e2e/helpers"
)

// TestE2E_SchemaDialect_Portable verifies that one Anthropic-shaped
// tool whose input_schema deliberately exercises the v0.2.0 dialect
// pipeline (nested object + enum + additionalProperties:false) is
// translated correctly to every configured provider when the client
// is built with SchemaPolicyLossy.
//
// We expect each provider to either emit a tool_use block or, if the
// model decides to answer in text, at least produce a non-empty
// response. The point is to confirm the upstream accepts the
// post-Normalize body — not to verify reasoning quality.
//
// Sub-tests are skipped at build time when their credential is
// missing, matching the pattern in matrix_test.go.
func TestE2E_SchemaDialect_Portable(t *testing.T) {
	type providerCase struct {
		name  string
		build func(*testing.T) (*anthropify.Client, string)
	}
	cases := []providerCase{
		{name: string(ProviderAnthropic), build: newAnthropicClientLossy},
		{name: string(ProviderOpenAI), build: newOpenAIClientLossy},
		{name: string(ProviderGemini), build: newGeminiClientLossy},
		{name: string(ProviderKimi), build: newKimiClientLossy},
		{name: string(ProviderGLM), build: newGLMClientLossy},
	}
	ran := 0
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			cli, model := tc.build(t)
			req := nestedWeatherToolPrompt(model)
			stream, err := cli.CreateMessageStream(context.Background(), req)
			if err != nil {
				t.Fatalf("CreateMessageStream: %v", err)
			}
			defer stream.Close()
			sum := helpers.DrainStream(t, stream)
			helpers.AssertEventEnvelope(t, sum)
			helpers.AssertInputTokensPositive(t, sum)
			// We accept either tool_use (model invoked the tool) or
			// non-empty text (model answered in prose).
			gotToolUse := false
			for _, bt := range sum.BlockTypes {
				if strings.EqualFold(bt, "tool_use") {
					gotToolUse = true
					break
				}
			}
			if !gotToolUse {
				helpers.AssertNonEmptyText(t, sum)
			}
			ran++
		})
	}
	if ran == 0 {
		t.Skip("no provider credentials configured; full SchemaDialect matrix skipped")
	}
}

// nestedWeatherToolPrompt is the tool-use prompt for the v0.2.0 schema
// dialect e2e. The schema contains a nested location object with its
// own additionalProperties:false plus a top-level additionalProperties:
// false; the enum on `units` exercises the type-inference path inside
// the Gemini converter.
func nestedWeatherToolPrompt(model string) anthropic.MessageNewParams {
	src := `{
		"model": "` + model + `",
		"max_tokens": 512,
		"tools": [
			{
				"name": "get_weather",
				"description": "Look up the current weather for a given location.",
				"input_schema": {
					"type": "object",
					"properties": {
						"location": {
							"type": "object",
							"description": "Where to look up the weather.",
							"properties": {
								"city": {"type": "string", "description": "City name."},
								"country": {"type": "string", "description": "ISO-3166-1 alpha-2 country code."}
							},
							"required": ["city"],
							"additionalProperties": false
						},
						"units": {
							"type": "string",
							"description": "Temperature unit.",
							"enum": ["celsius", "fahrenheit"]
						}
					},
					"required": ["location"],
					"additionalProperties": false
				}
			}
		],
		"messages": [
			{"role":"user","content":[{"type":"text","text":"What is the weather in Beijing right now? Use the get_weather tool with country=CN and units=celsius."}]}
		]
	}`
	return helpers.MustParams(src)
}

// The lossy-policy variants mirror clients.go's newXxxClient but inject
// WithSchemaPolicy(SchemaPolicyLossy) so additionalProperties on Gemini
// downgrades cleanly instead of erroring under the default Strict.
//
// Code is duplicated rather than parameterised because each helper has
// a slightly different option set (modelOverride, gemini mode, etc.)
// and intermixing the policy parameter with the existing helpers would
// muddle the simple credential-skipping logic.

func newAnthropicClientLossy(t *testing.T) (*anthropify.Client, string) {
	t.Helper()
	env := AnthropicEnv()
	skipIfUnset(t, env, "ANTHROPIC_API_KEY")
	cli, err := anthropify.New(
		anthropify.WithAnthropic(anthropify.AnthropicConfig{APIKey: env.APIKey}),
		anthropify.WithSchemaPolicy(anthropify.SchemaPolicyLossy),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newOpenAIClientLossy(t *testing.T) (*anthropify.Client, string) {
	t.Helper()
	env := OpenAIEnv()
	skipIfUnset(t, env, "OPENAI_API_KEY")
	cli, err := anthropify.New(
		anthropify.WithOpenAI(anthropify.OpenAIConfig{APIKey: env.APIKey}),
		anthropify.WithSchemaPolicy(anthropify.SchemaPolicyLossy),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newGeminiClientLossy(t *testing.T) (*anthropify.Client, string) {
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
		anthropify.WithSchemaPolicy(anthropify.SchemaPolicyLossy),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newChatCompletionsClientLossy(t *testing.T, env ProviderEnv, backend, envVarName string) (*anthropify.Client, string) {
	t.Helper()
	skipIfUnset(t, env, envVarName)
	cli, err := anthropify.New(
		anthropify.WithChatCompletion(backend, anthropify.OpenAICompatConfig{
			BaseURL: env.BaseURL,
			APIKey:  env.APIKey,
		}),
		anthropify.WithModelOverride(func(model string) (anthropify.Route, bool) {
			return anthropify.Route{
				Provider: anthropify.ProviderChatCompletions,
				Backend:  backend,
			}, true
		}),
		anthropify.WithSchemaPolicy(anthropify.SchemaPolicyLossy),
	)
	if err != nil {
		t.Fatalf("anthropify.New: %v", err)
	}
	return cli, env.Model
}

func newKimiClientLossy(t *testing.T) (*anthropify.Client, string) {
	return newChatCompletionsClientLossy(t, KimiEnv(), "kimi", "KIMI_API_KEY")
}

func newGLMClientLossy(t *testing.T) (*anthropify.Client, string) {
	return newChatCompletionsClientLossy(t, GLMEnv(), "glm", "GLM_API_KEY")
}
