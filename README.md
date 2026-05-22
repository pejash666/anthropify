> English | [简体中文](./README.zh-CN.md)

<p align="center"><img src="./assets/banner.png" alt="Anthropify" width="640"></p>

<h1 align="center">Anthropify</h1>

<p align="center"><i>The Anthropic SDK, for every LLM. In Go.</i></p>

<p align="center">
<a href="https://pkg.go.dev/github.com/shahao/anthropify"><img src="https://pkg.go.dev/badge/github.com/shahao/anthropify.svg" alt="Go Reference"></a>
<a href="https://go.dev/"><img src="https://img.shields.io/badge/go-1.24%2B-00ADD8" alt="Go version"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

## What is this?

Anthropify is a Go library that lets you talk to OpenAI, Gemini, Kimi, GLM,
DeepSeek and other LLM providers through the Anthropic Messages API shape.
It does protocol translation, SSE streaming, tool use, thinking blocks and
`cache_control` in-process — no proxy server in front. Callers only ever see
`anthropic.MessageNewParams` and `anthropic.MessageStreamEventUnion`.

## Why Anthropic-canonical?

Streaming protocols differ widely across vendors, and the shape you pick as
the canonical event format affects every downstream agent.

- Anthropic's stream is a small, ordered set of events with explicit content
  block boundaries: `message_start`, `content_block_start`,
  `content_block_delta`, `content_block_stop`, `message_delta`,
  `message_stop`. Text, tool use and thinking each live in their own block.
- OpenAI Chat Completions packs everything into `choices[*].delta`. Tool
  calls, reasoning, and refusals are tacked on as side fields with no
  structural separation between content kinds.
- OpenAI Responses splits the same stream into a much larger event set
  (`response.output_item.added`, `response.content_part.added`,
  `response.output_text.delta`, and so on). The extra granularity adds
  bookkeeping without making the wire shape easier to consume.

For an agent framework that needs a stable internal event vocabulary, the
Anthropic shape sits at a useful point on the granularity curve. Anthropify
adopts it as the canonical form and normalises every provider into it.

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/anthropics/anthropic-sdk-go"
    ap "github.com/shahao/anthropify"
)

func main() {
    client, err := ap.New(
        ap.WithChatCompletion("kimi", ap.OpenAICompatConfig{
            BaseURL: "https://api.moonshot.cn/v1",
            APIKey:  os.Getenv("KIMI_API_KEY"),
        }),
        ap.WithModelRoute("kimi-", ap.Route{
            Provider: ap.ProviderChatCompletions,
            Backend:  "kimi",
        }),
    )
    if err != nil {
        panic(err)
    }

    stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
        Model:     "kimi-k2-thinking",
        MaxTokens: 1024,
        Messages: []anthropic.MessageParam{
            anthropic.NewUserMessage(anthropic.NewTextBlock("Hello, who are you?")),
        },
    })
    if err != nil {
        panic(err)
    }
    defer stream.Close()

    for stream.Next() {
        fmt.Printf("%+v\n", stream.Current()) // anthropic.MessageStreamEventUnion
    }
    if err := stream.Err(); err != nil {
        panic(err)
    }
}
```

Non-streaming uses the same request type and returns `*anthropic.Message`:

```go
msg, err := client.CreateMessage(ctx, req)
```

## Supported providers

| Provider          | Streaming | Non-streaming         | Tool use | Thinking          | cache_control |
|-------------------|-----------|------------------------|----------|-------------------|---------------|
| Anthropic         | yes       | native                 | yes      | yes               | yes           |
| OpenAI Responses  | yes       | drain-and-assemble     | yes      | yes (reasoning)   | —             |
| Gemini Native     | yes       | drain-and-assemble     | yes      | yes               | partial       |
| Kimi (Moonshot)   | yes       | drain-and-assemble     | yes      | yes               | —             |
| GLM (Zhipu)       | yes       | drain-and-assemble     | yes      | model-dependent   | —             |
| DeepSeek          | yes       | drain-and-assemble     | yes      | yes (V3.2)        | —             |

Only the Anthropic adapter exposes a real non-streaming endpoint. For every
other provider, `CreateMessage` opens the stream and assembles the final
`*anthropic.Message` from content blocks.

Routing is prefix-based on `req.Model`:

| Prefix              | Provider            |
|---------------------|---------------------|
| `claude-`           | `anthropic`         |
| `gpt-`, `o1-`, `o3-`| `openai_responses`  |
| `gemini-`           | `gemini_native`     |

Kimi / GLM / DeepSeek and other OpenAI-compatible upstreams are registered
with `WithChatCompletion(name, ...)` and routed via `WithModelRoute(prefix, Route{Provider: ProviderChatCompletions, Backend: name})`.

## Gemini provider configuration

The Gemini adapter speaks three auth flavours against two upstream hosts.
The mode is selected explicitly via `GeminiMode`; `GeminiModeAuto` keeps
the original heuristic (project+location implies Vertex, otherwise Studio).

| Mode    | When to use                       | Endpoint                                                     | Auth                          |
|---------|-----------------------------------|--------------------------------------------------------------|-------------------------------|
| Studio  | Free tier, prototyping            | `generativelanguage.googleapis.com/v1beta`                   | API key (`AIza...`) via `?key=` |
| Express | Vertex with simplified auth       | `aiplatform.googleapis.com/v1/publishers/google`             | API key (`AQ.*`) via `?key=`    |
| Vertex  | Production, GCP-native            | `aiplatform.googleapis.com/v1/projects/<p>/locations/<l>`    | OAuth2 Bearer token             |

```go
// Studio
ap.WithGemini(ap.GeminiConfig{
    Mode:   ap.GeminiModeStudio,
    APIKey: os.Getenv("GEMINI_API_KEY"),
})
```

```go
// Express (Vertex with API key)
ap.WithGemini(ap.GeminiConfig{
    Mode:   ap.GeminiModeExpress,
    APIKey: os.Getenv("VERTEX_EXPRESS_KEY"), // AQ.*
})
```

```go
// Vertex (OAuth2)
ap.WithGemini(ap.GeminiConfig{
    Mode:     ap.GeminiModeVertex,
    Project:  os.Getenv("VERTEX_PROJECT"),
    Location: os.Getenv("VERTEX_LOCATION"),
    APIKey:   bearerToken, // gcloud auth print-access-token
})
```

## Installation

```bash
go get github.com/shahao/anthropify
```

Requires Go 1.24 or later.

## How Anthropify compares

Anthropify is intentionally narrow: a Go library that converts between
Anthropic-canonical events and provider-native protocols. It is not a proxy
server, not a routing layer, not a multi-tenant gateway, and ships no
`cmd/`, no YAML, and no daemon.

If you need a hosted proxy with auth, rate-limiting, billing, and
multi-tenancy out of the box, LiteLLM is a more complete solution and is
Python-first. If you need a Go-native library to embed inside an agent —
with first-party streaming, tool use, and the Anthropic canonical event
shape — Anthropify is built for that.

## Status

> **alpha**. APIs may change. All four adapter paths are real-API tested
> (25 E2E tests, 0 failures across Anthropic / OpenAI / Gemini / Kimi /
> GLM / DeepSeek). Production users should pin to a specific commit.

## Testing

Unit tests are fixture-driven and never touch the network:

```bash
go test ./...
```

Real-network smoke tests live under `e2e/` and are gated behind a build tag:

```bash
cp .env.e2e.example .env.e2e   # fill in only the providers you have keys for
make test-e2e                  # loads .env.e2e and runs `go test -tags=e2e ./e2e/...`
```

Providers whose API key is unset are skipped automatically. See
`e2e/README.md` for the per-provider matrix. `.env.e2e` is git-ignored;
only `.env.e2e.example` is tracked.

## Contributing

Issues and pull requests are welcome. Before sending a PR:

- `go test ./...` must pass.
- If you change adapter behaviour, add a corresponding E2E test under
  `e2e/` and verify with `make test-e2e` against a real key.
- Keep the public API aliased to `github.com/anthropics/anthropic-sdk-go`
  types where possible; do not introduce parallel request/response shapes.

## License

MIT. See [LICENSE](./LICENSE).
