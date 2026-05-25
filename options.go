package anthropify

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

	geminiadapter "github.com/shahao/anthropify/adapter/gemini_native"
	"github.com/shahao/anthropify/internal/schema"
)

// SchemaPolicy is the v0.2.0 client-global tool-schema normalisation
// policy. The default (PolicyStrict) makes every lossy transform a
// hard error so callers learn about silent field drops; PolicyLossy
// performs the documented rewrites and logs them at slog.Warn;
// PolicyBestEffort additionally tolerates the documented best-effort
// downgrades (e.g. OpenAI Strict additionalProperties:true -> false).
//
// See docs/design/v0.2.0-protocol-driven-adapters.md §4.3 for the
// authoritative behaviour table.
type SchemaPolicy = schema.Policy

// Re-exports of the underlying enum constants. See SchemaPolicy.
const (
	SchemaPolicyStrict     = schema.PolicyStrict
	SchemaPolicyLossy      = schema.PolicyLossy
	SchemaPolicyBestEffort = schema.PolicyBestEffort
)

// ErrSchemaIncompatible is returned (wrapped) by adapters when a tool's
// input_schema cannot be losslessly converted to the upstream dialect
// under the active SchemaPolicy. Detect with errors.Is.
var ErrSchemaIncompatible = schema.ErrSchemaIncompatible

// _ keeps errors imported unconditionally so future error helpers can
// compile without adding the import.
var _ = errors.Is

// GeminiMode re-exports the adapter-level mode enum so callers do not
// need to import the adapter package directly.
type GeminiMode = geminiadapter.GeminiMode

// Re-exported Gemini mode constants. See adapter/gemini_native for the
// authoritative documentation of each variant.
const (
	GeminiModeAuto    = geminiadapter.GeminiModeAuto
	GeminiModeStudio  = geminiadapter.GeminiModeStudio
	GeminiModeExpress = geminiadapter.GeminiModeExpress
	GeminiModeVertex  = geminiadapter.GeminiModeVertex
)

// OpenAIResponsesConfig configures the OpenAI Responses adapter
// (GPT-5 family). Multiple backends may be registered under distinct
// names via WithOpenAIResponsesCompat; the legacy WithOpenAI is sugar
// for the default-named "openai" slot.
type OpenAIResponsesConfig struct {
	// APIKey is the Bearer token used when calling the upstream.
	APIKey string
	// BaseURL overrides the default https://api.openai.com endpoint.
	// Useful for Azure OpenAI or OpenRouter.
	BaseURL string
	// Organization, if set, is forwarded as the OpenAI-Organization header.
	Organization string
	// Project, if set, is forwarded as the OpenAI-Project header.
	Project string
	// ExtraHeaders is merged into every request.
	ExtraHeaders http.Header
}

// OpenAIConfig is the legacy alias for OpenAIResponsesConfig retained
// for source compatibility with v0.1.x callers.
type OpenAIConfig = OpenAIResponsesConfig

// AzureOpenAIConfig configures an Azure OpenAI Responses backend.
// Registered via WithAzureOpenAI; multiple backends may be registered
// under distinct names. Internally this rides on the same
// adapter/openai_responses code path as vanilla OpenAI — only URL
// composition and authentication headers differ.
//
// Two URL forms are supported, auto-picked by whether Deployment is set:
//   - Deployment != "" => deployment-bound URL
//     <BaseURL>/openai/deployments/<Deployment>/responses?api-version=<APIVersion>
//   - Deployment == "" => deployment-less URL
//     <BaseURL>/openai/responses?api-version=<APIVersion>
//     The body's model field then carries the deployment name verbatim
//     (the caller's req.Model after Route.UpstreamModel rewrite).
//
// Both api-key and Authorization: Bearer headers are sent on every
// request — the former is Azure's canonical key header, the latter is
// required for Entra/AAD access tokens and works with resource keys
// too. This belt-and-suspenders approach is robust against APIM
// gateways that strip one or the other.
//
// Backend names share the same namespace as
// WithOpenAIResponsesCompat; registering both helpers under the same
// name is rejected at New() time. See
// docs/design/v0.2.0-azure-responses.md for the authoritative design
// and rationale.
type AzureOpenAIConfig struct {
	// BaseURL is the Azure resource endpoint, e.g.
	// "https://my-resource.openai.azure.com" (no path). Required.
	BaseURL string
	// APIKey is either an Azure resource key or an Entra (AAD)
	// access token. Required.
	APIKey string
	// APIVersion is the ?api-version=... query value, e.g.
	// "2025-03-01-preview". Required — anthropify deliberately does
	// not pick a default so api-version upgrades stay visible in
	// source control.
	APIVersion string
	// Deployment, when non-empty, pins this backend to a single Azure
	// deployment and selects the deployment-bound URL form. Empty
	// falls back to the deployment-less URL form, in which case the
	// deployment name is read from the request body's model field
	// (the caller's req.Model, optionally rewritten by
	// Route.UpstreamModel).
	Deployment string
	// ExtraHeaders is merged into every request.
	ExtraHeaders http.Header
}

// AnthropicConfig configures the Anthropic passthrough adapter.
type AnthropicConfig struct {
	APIKey       string
	BaseURL      string
	Version      string // defaults to "2023-06-01"
	ExtraHeaders http.Header
}

// GeminiConfig configures the Gemini Vertex streamGenerateContent adapter.
type GeminiConfig struct {
	// Mode selects between Auto (heuristic), Studio, Express, and
	// Vertex. The zero value (Auto) preserves the legacy behaviour:
	// project+location => Vertex, otherwise APIKey => Studio.
	Mode GeminiMode
	// Project is the GCP project ID. Required for Vertex mode.
	Project string
	// Location is the GCP region (e.g. "us-central1"). Required for Vertex mode.
	Location string
	// Publisher defaults to "google".
	Publisher string
	// APIKey carries either an AI Studio key (Studio mode), a Vertex
	// Express key such as AQ.* (Express mode), or a Bearer access
	// token (Vertex mode).
	APIKey string
	// BaseURL overrides the default upstream host for the resolved mode.
	BaseURL string
	// ExtraHeaders is merged into every request.
	ExtraHeaders http.Header
}

// OpenAICompatConfig configures a single OpenAI-compatible ChatCompletion
// backend. Multiple backends may be registered under distinct names.
type OpenAICompatConfig struct {
	// BaseURL must not end with "/chat/completions"; the adapter appends it.
	BaseURL string
	// APIKey is forwarded as Authorization: Bearer <APIKey>.
	APIKey string
	// ExtraHeaders is merged into every request.
	ExtraHeaders http.Header
	// DefaultModel, when non-empty, is substituted into the outgoing
	// request if the user did not provide an explicit model.
	DefaultModel string
}

// Option mutates an internal config struct during Client construction.
type Option func(*config)

type config struct {
	httpClient *http.Client
	logger     *slog.Logger

	// Multi-backend protocol adapters. Each map keys a backend name
	// (e.g. "anthropic", "minimax") to its per-backend config.
	anthropicCompat       map[string]AnthropicConfig
	openaiResponsesCompat map[string]OpenAIResponsesConfig
	azureOpenAI           map[string]AzureOpenAIConfig

	// Single-instance (Track D — stays single in v0.2.0).
	gemini *GeminiConfig

	chatCompat  map[string]OpenAICompatConfig
	modelRoutes map[string]Route
	overrideFn  func(model string) (Route, bool)

	// schemaPolicy controls how schema.Normalize reacts to lossy tool
	// input_schema transforms. Default = SchemaPolicyStrict.
	schemaPolicy SchemaPolicy
}

// Route resolves a model name to a concrete provider plus an optional
// remapped model string forwarded upstream.
type Route struct {
	Provider ProviderKind
	// Backend selects which named backend handles the request.
	// Meaningful for ProviderAnthropic, ProviderOpenAIResponses, and
	// ProviderChatCompletions. An empty value defaults to "anthropic"
	// for ProviderAnthropic and "openai" for ProviderOpenAIResponses;
	// ignored for ProviderGeminiNative.
	Backend string
	// UpstreamModel, if non-empty, replaces req.Model before dispatch.
	UpstreamModel string
}

// WithHTTPClient injects a shared http.Client. Passing nil restores the
// default (http.DefaultClient).
func WithHTTPClient(h *http.Client) Option {
	return func(c *config) { c.httpClient = h }
}

// WithLogger installs an slog.Logger. When nil (the default) the library
// emits no log output.
func WithLogger(l *slog.Logger) Option {
	return func(c *config) { c.logger = l }
}

// WithAnthropicCompat registers an Anthropic-protocol backend under the
// given name. Multiple calls accumulate; the name is used as the
// Route.Backend value. To address this backend, add a prefix rule via
// WithModelRoute, e.g.
//
//	ap.WithModelRoute("MiniMax-",
//	    ap.Route{Provider: ap.ProviderAnthropic, Backend: "minimax"})
//
// The reserved name "anthropic" is the default-named slot used by the
// built-in claude-* prefix routes; WithAnthropic is sugar for
// WithAnthropicCompat("anthropic", cfg).
func WithAnthropicCompat(name string, cfg AnthropicConfig) Option {
	return func(c *config) {
		if c.anthropicCompat == nil {
			c.anthropicCompat = make(map[string]AnthropicConfig)
		}
		c.anthropicCompat[name] = cfg
	}
}

// WithOpenAIResponsesCompat registers an OpenAI Responses-protocol
// backend under the given name. Multiple calls accumulate; the name is
// used as the Route.Backend value. The reserved name "openai" is the
// default-named slot used by the built-in gpt-/o1-/o3- prefix routes;
// WithOpenAI is sugar for WithOpenAIResponsesCompat("openai", cfg).
func WithOpenAIResponsesCompat(name string, cfg OpenAIResponsesConfig) Option {
	return func(c *config) {
		if c.openaiResponsesCompat == nil {
			c.openaiResponsesCompat = make(map[string]OpenAIResponsesConfig)
		}
		c.openaiResponsesCompat[name] = cfg
	}
}

// WithOpenAI enables the OpenAI Responses adapter under the default
// "openai" backend slot. Equivalent to
// WithOpenAIResponsesCompat("openai", cfg).
func WithOpenAI(cfg OpenAIResponsesConfig) Option {
	return WithOpenAIResponsesCompat("openai", cfg)
}

// WithAnthropic enables the Anthropic passthrough adapter under the
// default "anthropic" backend slot. Equivalent to
// WithAnthropicCompat("anthropic", cfg).
func WithAnthropic(cfg AnthropicConfig) Option {
	return WithAnthropicCompat("anthropic", cfg)
}

// WithAzureOpenAI registers an Azure OpenAI Responses backend under
// the given name. Multiple calls accumulate; the name is used as the
// Route.Backend value. Backend names share the same namespace as
// WithOpenAIResponsesCompat — registering both helpers under the same
// name is rejected at New() time.
//
// Typical usage pairs this option with WithModelRoute, since Azure
// deployment names are usually distinct from canonical OpenAI model
// names:
//
//	ap.WithAzureOpenAI("azure", ap.AzureOpenAIConfig{
//	    BaseURL:    "https://my-resource.openai.azure.com",
//	    APIKey:     os.Getenv("AZURE_OPENAI_API_KEY"),
//	    APIVersion: "2025-03-01-preview",
//	    Deployment: "gpt-5-deployment",
//	}),
//	ap.WithModelRoute("gpt-5", ap.Route{
//	    Provider: ap.ProviderOpenAIResponses,
//	    Backend:  "azure",
//	}),
//
// See docs/design/v0.2.0-azure-responses.md for the authoritative
// design covering URL forms, dual-auth headers, and the layered model
// fallback (cfg.Deployment > Route.UpstreamModel > req.Model).
func WithAzureOpenAI(name string, cfg AzureOpenAIConfig) Option {
	return func(c *config) {
		if c.azureOpenAI == nil {
			c.azureOpenAI = make(map[string]AzureOpenAIConfig)
		}
		c.azureOpenAI[name] = cfg
	}
}

// WithGemini enables the Gemini native adapter.
func WithGemini(cfg GeminiConfig) Option {
	return func(c *config) {
		cfgCopy := cfg
		c.gemini = &cfgCopy
	}
}

// WithChatCompletion registers an OpenAI-compatible ChatCompletion backend
// under the given name. Multiple calls accumulate; the name is used as
// the Route.Backend value.
func WithChatCompletion(name string, cfg OpenAICompatConfig) Option {
	return func(c *config) {
		if c.chatCompat == nil {
			c.chatCompat = make(map[string]OpenAICompatConfig)
		}
		c.chatCompat[name] = cfg
	}
}

// WithModelRoute adds an explicit prefix-based routing rule. Longer
// prefixes match first. Rules added via this option take precedence over
// the built-in defaults.
func WithModelRoute(prefix string, route Route) Option {
	return func(c *config) {
		if c.modelRoutes == nil {
			c.modelRoutes = make(map[string]Route)
		}
		c.modelRoutes[strings.ToLower(prefix)] = route
	}
}

// WithModelOverride installs a function that, when it returns ok, wins
// over all prefix rules. Intended for dynamic routing (A/B tests, config
// reloads).
func WithModelOverride(fn func(model string) (Route, bool)) Option {
	return func(c *config) { c.overrideFn = fn }
}

// WithSchemaPolicy selects the client-global tool-schema normalisation
// policy. The default (when this option is not supplied) is
// SchemaPolicyStrict: any non-cosmetic lossy transform returns
// ErrSchemaIncompatible from Invoke/Stream. SchemaPolicyLossy performs
// the documented rewrites/drops and logs each via slog.Warn;
// SchemaPolicyBestEffort additionally tolerates the documented
// best-effort downgrades (see schema package docs).
func WithSchemaPolicy(p SchemaPolicy) Option {
	return func(c *config) { c.schemaPolicy = p }
}

// defaultConfig produces a fresh config populated with library defaults.
func defaultConfig() *config {
	return &config{
		httpClient:            http.DefaultClient,
		logger:                newNopLogger(),
		anthropicCompat:       make(map[string]AnthropicConfig),
		openaiResponsesCompat: make(map[string]OpenAIResponsesConfig),
		azureOpenAI:           make(map[string]AzureOpenAIConfig),
		chatCompat:            make(map[string]OpenAICompatConfig),
		modelRoutes: map[string]Route{
			"claude-": {Provider: ProviderAnthropic},
			"gpt-":    {Provider: ProviderOpenAIResponses},
			"o1-":     {Provider: ProviderOpenAIResponses},
			"o3-":     {Provider: ProviderOpenAIResponses},
			"gemini-": {Provider: ProviderGeminiNative},
		},
	}
}

// newNopLogger returns a slog.Logger backed by io.Discard that will never
// emit output (its level is set to the maximum).
func newNopLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.Level(128)}))
}

// contextKey avoids accidental collisions with caller-owned keys.
type contextKey struct{ name string }

func (k contextKey) String() string { return "anthropify:" + k.name }

var ctxKeyLogger = contextKey{name: "logger"}

// withLogger stores a logger on ctx for adapter-local access.
func withLogger(ctx context.Context, l *slog.Logger) context.Context {
	return context.WithValue(ctx, ctxKeyLogger, l)
}

// loggerFrom recovers the adapter-local logger; returns a no-op when absent.
func loggerFrom(ctx context.Context) *slog.Logger {
	if v, ok := ctx.Value(ctxKeyLogger).(*slog.Logger); ok && v != nil {
		return v
	}
	return newNopLogger()
}
