//go:build e2e

package e2e

import (
	"context"
	"testing"

	"github.com/shahao/hybridstream"
	"github.com/shahao/hybridstream/e2e/helpers"
)

// TestE2E_Matrix_Consistency runs the same BasicTextPrompt against
// every configured provider and asserts they all emit a non-empty text
// response, an end_turn stop reason, and a positive input-token count.
// Providers with missing credentials are silently skipped at the
// sub-test level so a partial credential set still produces useful
// signal.
func TestE2E_Matrix_Consistency(t *testing.T) {
	type providerCase struct {
		name  string
		build func(*testing.T) (*hybridstream.Client, string)
	}
	cases := []providerCase{
		{name: string(ProviderAnthropic), build: newAnthropicClient},
		{name: string(ProviderOpenAI), build: newOpenAIClient},
		{name: string(ProviderGemini), build: newGeminiClient},
		{name: string(ProviderKimi), build: newKimiClient},
		{name: string(ProviderGLM), build: newGLMClient},
	}
	ran := 0
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			cli, model := tc.build(t) // build calls t.Skip when unconfigured
			// 1024 tokens of headroom: reasoning-capable models
			// (e.g. gpt-5-mini via OpenAI Responses) easily burn
			// the entire budget on hidden reasoning when given a
			// short prompt, leaving zero tokens for visible text.
			// A direct curl probe against gpt-5-mini with
			// max_output_tokens=128 and this prompt confirmed the
			// upstream emits only `response.output_item.added`
			// (reasoning) + `response.incomplete` with
			// reason=max_output_tokens and NO output_text deltas,
			// so the converter produces no content blocks. 1024
			// matches the per-provider BasicStream budget.
			stream, err := cli.CreateMessageStream(context.Background(),
				helpers.BasicTextPrompt(model, 1024))
			if err != nil {
				t.Fatalf("CreateMessageStream: %v", err)
			}
			defer stream.Close()
			sum := helpers.DrainStream(t, stream)
			helpers.AssertEventEnvelope(t, sum)
			helpers.AssertNonEmptyText(t, sum)
			helpers.AssertStopReason(t, sum, "end_turn", "stop_sequence")
			helpers.AssertInputTokensPositive(t, sum)
			ran++
		})
	}
	if ran == 0 {
		t.Skip("matrix: no providers configured")
	}
}
