package openai_responses

import (
	"strings"
	"testing"
)

// TestBuildEndpoint_VanillaOpenAI confirms the vanilla URL composition
// is byte-identical to the pre-Azure shape (regression guard for the
// refactor that moved URL building out of Stream).
func TestBuildEndpoint_VanillaOpenAI(t *testing.T) {
	a, err := New("openai", Config{APIKey: "sk-x"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got, want := a.buildEndpoint(), "https://api.openai.com/v1/responses"; got != want {
		t.Fatalf("buildEndpoint = %q, want %q", got, want)
	}
}

// TestBuildEndpoint_VanillaCustomBaseURL verifies a non-default BaseURL
// still produces the legacy /v1/responses path.
func TestBuildEndpoint_VanillaCustomBaseURL(t *testing.T) {
	a, err := New("openai", Config{APIKey: "sk-x", BaseURL: "https://api.example.com"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got, want := a.buildEndpoint(), "https://api.example.com/v1/responses"; got != want {
		t.Fatalf("buildEndpoint = %q, want %q", got, want)
	}
}

// TestBuildEndpoint_AzureDeploymentless covers the URL form documented
// in docs/design/v0.2.0-azure-responses.md §1: deployment-less URL
// when Config.Deployment is empty. This matches the llm-proxy
// reference implementation.
func TestBuildEndpoint_AzureDeploymentless(t *testing.T) {
	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    "https://my-resource.openai.azure.com",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := a.buildEndpoint()
	want := "https://my-resource.openai.azure.com/openai/responses?api-version=2025-03-01-preview"
	if got != want {
		t.Fatalf("buildEndpoint = %q, want %q", got, want)
	}
}

// TestBuildEndpoint_AzureDeploymentBound covers the second URL form:
// when Config.Deployment is set, the deployment is in the path.
func TestBuildEndpoint_AzureDeploymentBound(t *testing.T) {
	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    "https://my-resource.openai.azure.com",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
		Deployment: "gpt-5-PTU",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := a.buildEndpoint()
	want := "https://my-resource.openai.azure.com/openai/deployments/gpt-5-PTU/responses?api-version=2025-03-01-preview"
	if got != want {
		t.Fatalf("buildEndpoint = %q, want %q", got, want)
	}
}

// TestBuildEndpoint_AzureTrailingSlashTrim guards against double
// slashes when callers paste a BaseURL that already ends in "/".
func TestBuildEndpoint_AzureTrailingSlashTrim(t *testing.T) {
	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    "https://my-resource.openai.azure.com/",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := a.buildEndpoint()
	if strings.Contains(got, ".com//openai") {
		t.Fatalf("buildEndpoint contains double slash: %q", got)
	}
	want := "https://my-resource.openai.azure.com/openai/responses?api-version=2025-03-01-preview"
	if got != want {
		t.Fatalf("buildEndpoint = %q, want %q", got, want)
	}
}

// TestBuildEndpoint_AzureAPIVersionEncoded confirms the api-version
// query is percent-encoded via url.Values rather than concatenated raw.
// Hostile caller configs (commas, spaces) must not corrupt the query.
func TestBuildEndpoint_AzureAPIVersionEncoded(t *testing.T) {
	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    "https://x.openai.azure.com",
		Azure:      true,
		APIVersion: "2025-03-01-preview&injected=1",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := a.buildEndpoint()
	// Raw '&' would split the query and inject a second key. The
	// expected encoding turns '&' into %26.
	if strings.Contains(got, "&injected=1") {
		t.Fatalf("api-version not encoded; got %q", got)
	}
	if !strings.Contains(got, "%26injected%3D1") {
		t.Fatalf("expected percent-encoded api-version in %q", got)
	}
}

// TestNew_AzureRequiresAPIVersion enforces the public contract that
// AzureOpenAIConfig.APIVersion is mandatory.
func TestNew_AzureRequiresAPIVersion(t *testing.T) {
	_, err := New("azure", Config{
		APIKey:  "k",
		BaseURL: "https://x.openai.azure.com",
		Azure:   true,
	})
	if err == nil {
		t.Fatal("expected error when APIVersion missing in Azure mode")
	}
	if !strings.Contains(err.Error(), "APIVersion") {
		t.Fatalf("error %q should mention APIVersion", err.Error())
	}
}

// TestNew_AzureRequiresBaseURL guards against the easy mistake of
// forgetting the resource endpoint. Vanilla OpenAI defaults BaseURL;
// Azure must not.
func TestNew_AzureRequiresBaseURL(t *testing.T) {
	_, err := New("azure", Config{
		APIKey:     "k",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err == nil {
		t.Fatal("expected error when BaseURL missing in Azure mode")
	}
	if !strings.Contains(err.Error(), "BaseURL") {
		t.Fatalf("error %q should mention BaseURL", err.Error())
	}
}

// TestNew_AzureRequiresAPIKey applies in both modes; here we lock it
// in for the Azure branch to keep the contract symmetric.
func TestNew_AzureRequiresAPIKey(t *testing.T) {
	_, err := New("azure", Config{
		BaseURL:    "https://x.openai.azure.com",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err == nil {
		t.Fatal("expected error when APIKey missing")
	}
	if !strings.Contains(err.Error(), "APIKey") {
		t.Fatalf("error %q should mention APIKey", err.Error())
	}
}

// TestNew_VanillaUnaffected ensures the Azure branch did not break the
// vanilla constructor's defaulting behaviour (BaseURL fallback,
// HTTPClient fallback).
func TestNew_VanillaUnaffected(t *testing.T) {
	a, err := New("openai", Config{APIKey: "sk-x"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if a.cfg.BaseURL != "https://api.openai.com" {
		t.Fatalf("vanilla BaseURL default = %q", a.cfg.BaseURL)
	}
	if a.cfg.HTTPClient == nil {
		t.Fatal("vanilla HTTPClient default missing")
	}
	if a.cfg.Azure {
		t.Fatal("vanilla cfg.Azure should be false")
	}
	if got := a.effectiveModel(); got != "" {
		t.Fatalf("vanilla effectiveModel = %q, want empty", got)
	}
}

// TestEffectiveModel_AzureWithDeployment covers the layered model
// fallback rule: cfg.Deployment is the highest-priority override.
func TestEffectiveModel_AzureWithDeployment(t *testing.T) {
	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    "https://x.openai.azure.com",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
		Deployment: "my-gpt5",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got, want := a.effectiveModel(), "my-gpt5"; got != want {
		t.Fatalf("effectiveModel = %q, want %q", got, want)
	}
}

// TestEffectiveModel_AzureNoDeployment confirms the deployment-less
// branch returns "" so BuildRequestForAzure keeps req.Model verbatim.
// This is how the layered fallback degrades to Route.UpstreamModel /
// req.Model.
func TestEffectiveModel_AzureNoDeployment(t *testing.T) {
	a, err := New("azure", Config{
		APIKey:     "k",
		BaseURL:    "https://x.openai.azure.com",
		Azure:      true,
		APIVersion: "2025-03-01-preview",
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := a.effectiveModel(); got != "" {
		t.Fatalf("effectiveModel = %q, want empty", got)
	}
}
