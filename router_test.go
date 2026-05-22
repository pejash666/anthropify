package anthropify

import (
	"errors"
	"testing"
)

func TestResolve_PrefixMatch(t *testing.T) {
	c := defaultConfig()
	r, err := c.resolve("claude-3-5-sonnet-20241022")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.Provider != ProviderAnthropic {
		t.Fatalf("provider = %v, want anthropic", r.Provider)
	}
}

func TestResolve_UnknownModel(t *testing.T) {
	c := defaultConfig()
	_, err := c.resolve("mistral-tiny")
	if !errors.Is(err, ErrUnknownModel) {
		t.Fatalf("err = %v, want ErrUnknownModel", err)
	}
}

func TestResolve_OverrideWins(t *testing.T) {
	c := defaultConfig()
	c.overrideFn = func(model string) (Route, bool) {
		if model == "claude-3" {
			return Route{Provider: ProviderChatCompletions, Backend: "kimi"}, true
		}
		return Route{}, false
	}
	r, err := c.resolve("claude-3")
	if err != nil {
		t.Fatal(err)
	}
	if r.Provider != ProviderChatCompletions || r.Backend != "kimi" {
		t.Fatalf("route = %+v", r)
	}
}

func TestResolve_LongestPrefixWins(t *testing.T) {
	c := defaultConfig()
	c.modelRoutes["gpt-5-"] = Route{Provider: ProviderOpenAIResponses, UpstreamModel: "gpt-5-override"}
	r, err := c.resolve("gpt-5-mini")
	if err != nil {
		t.Fatal(err)
	}
	if r.UpstreamModel != "gpt-5-override" {
		t.Fatalf("upstream model = %q", r.UpstreamModel)
	}
}
