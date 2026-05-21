# hybridstream

A Go library for converting between LLM streaming protocols, with Anthropic
as the canonical format. Supports OpenAI Response API, Gemini Native, and
OpenAI-compatible Chat Completions.

Callers only ever see Anthropic Messages-API types. Under the hood,
`hybridstream` translates to and from each provider's native protocol.

## Quick start

```go
client, _ := hybridstream.New(
    hybridstream.WithOpenAI(hybridstream.OpenAIConfig{APIKey: "sk-..."}),
)
stream, _ := client.CreateMessageStream(ctx, req) // req is anthropic.MessageNewParams
for stream.Next() { fmt.Printf("%+v\n", stream.Current()) }
```

## Providers

```go
client, err := hybridstream.New(
    hybridstream.WithAnthropic(hybridstream.AnthropicConfig{APIKey: "sk-ant-..."}),
    hybridstream.WithOpenAI(hybridstream.OpenAIConfig{APIKey: "sk-..."}),
    hybridstream.WithGemini(hybridstream.GeminiConfig{Project: "my-proj", Location: "us-central1"}),
    hybridstream.WithChatCompletion("kimi", hybridstream.OpenAICompatConfig{
        BaseURL: "https://api.moonshot.cn/v1",
        APIKey:  "sk-kimi-...",
    }),
)
```

Routing is prefix-based on `req.Model`:

| Prefix     | Provider            |
|------------|---------------------|
| `claude-`  | `anthropic`         |
| `gpt-`, `o1-`, `o3-` | `openai_responses` |
| `gemini-`  | `gemini_native`     |

Custom rules:

```go
hybridstream.WithModelRoute("kimi-", hybridstream.Route{
    Provider: hybridstream.ProviderChatCompletions,
    Backend:  "kimi",
})
```

Dynamic override (wins over prefixes):

```go
hybridstream.WithModelOverride(func(model string) (hybridstream.Route, bool) {
    if strings.HasSuffix(model, "-beta") {
        return hybridstream.Route{Provider: hybridstream.ProviderAnthropic}, true
    }
    return hybridstream.Route{}, false
})
```

## Non-streaming

```go
msg, err := client.CreateMessage(ctx, req)
// msg is *anthropic.Message
```

Adapters that lack a native non-streaming endpoint (OpenAI Responses,
Gemini) drain the stream and assemble the Message in-process.

## API surface

All request/response types are re-exported from
`github.com/anthropics/anthropic-sdk-go`:

- `hybridstream.MessageNewParams` = `anthropic.MessageNewParams`
- `hybridstream.Message`          = `anthropic.Message`
- `hybridstream.MessageStreamEventUnion` = `anthropic.MessageStreamEventUnion`

## What this is NOT

- Not an HTTP server, middleware, or SSE endpoint. Bring your own server.
- No YAML configuration, no global state, no logging framework.
- No billing, auth, rate limiting, prompt filtering, or fraud detection.
- No Secrets Manager / LangSmith / Prometheus integration.
- No BYOK workflow beyond "pass the API key to the constructor".
- No `cmd/`, no demo server, no opinionated CLI.

## Status

| Adapter            | State                     |
|--------------------|---------------------------|
| `anthropic`        | Complete (passthrough).   |
| `openai_responses` | Streaming complete; non-streaming path drains the stream. |
| `gemini_native`    | Streaming complete; non-streaming path drains the stream. |
| `chat_completions` | Streaming complete; serves Kimi / DeepSeek / Qwen / GLM and any OpenAI Chat Completions compatible upstream. |

See `adapter/<provider>/adapter.go` for TODOs.

## Testing

```
go test ./...
```

Tests are fixture-driven (see `adapter/openai_responses/testdata/`) and
do not hit any real network endpoint.

### E2E testing

Real-network smoke tests live under `e2e/` and are gated behind a build
tag, so `go test ./...` never touches them.

```sh
cp .env.e2e.example .env.e2e   # fill in only the providers you have keys for
make test-e2e                  # loads .env.e2e and runs `go test -tags=e2e ./e2e/...`
```

Providers whose API key is unset are skipped automatically; the suite
never fails because of a missing credential. See `e2e/README.md` for the
full matrix and per-provider test breakdown.

> `.env.e2e` is git-ignored. Only `.env.e2e.example` is tracked.

## License

MIT
