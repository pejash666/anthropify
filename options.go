package hybridstream

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// OpenAIConfig configures the OpenAI Responses adapter (GPT-5 family).
type OpenAIConfig struct {
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

// AnthropicConfig configures the Anthropic passthrough adapter.
type AnthropicConfig struct {
	APIKey       string
	BaseURL      string
	Version      string // defaults to "2023-06-01"
	ExtraHeaders http.Header
}

// GeminiConfig configures the Gemini Vertex streamGenerateContent adapter.
type GeminiConfig struct {
	// Project is the GCP project ID. Required.
	Project string
	// Location is the GCP region (e.g. "us-central1"). Required.
	Location string
	// Publisher defaults to "google".
	Publisher string
	// APIKey, when set, uses the AI Studio endpoint instead of Vertex.
	APIKey string
	// BaseURL overrides the default Vertex endpoint.
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

	openai      *OpenAIConfig
	anthropic   *AnthropicConfig
	gemini      *GeminiConfig
	chatCompat  map[string]OpenAICompatConfig
	modelRoutes map[string]Route
	overrideFn  func(model string) (Route, bool)
}

// Route resolves a model name to a concrete provider plus an optional
// remapped model string forwarded upstream.
type Route struct {
	Provider ProviderKind
	// Backend is the key used for chat_completions (e.g. "kimi");
	// ignored for other providers.
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

// WithOpenAI enables the OpenAI Responses adapter.
func WithOpenAI(cfg OpenAIConfig) Option {
	return func(c *config) {
		cfgCopy := cfg
		c.openai = &cfgCopy
	}
}

// WithAnthropic enables the Anthropic passthrough adapter.
func WithAnthropic(cfg AnthropicConfig) Option {
	return func(c *config) {
		cfgCopy := cfg
		c.anthropic = &cfgCopy
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

// defaultConfig produces a fresh config populated with library defaults.
func defaultConfig() *config {
	return &config{
		httpClient: http.DefaultClient,
		logger:     newNopLogger(),
		chatCompat: make(map[string]OpenAICompatConfig),
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

func (k contextKey) String() string { return "hybridstream:" + k.name }

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
