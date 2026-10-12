# Use the Anthropic Messages API with OpenAI and Gemini in Go

Anthropify lets a Go application call OpenAI and Gemini with the request
and response types from the official `anthropic-sdk-go`. It translates
requests inside your process and returns Anthropic-shaped streaming
events. You do not need an Anthropic API key or a proxy server to call
these providers.

This is an application-facing compatibility layer, not an HTTP server
that exposes `/v1/messages`. Existing Go code can keep its Anthropic
message types, but calls go through `anthropify.Client`.

## Choose the Correct Upstream Protocol

For OpenAI's Responses API, configure `WithOpenAI`. For Google's native
Gemini API, configure `WithGemini`. Both return the same
`anthropic.MessageStreamEventUnion` events to application code.

An endpoint that supports OpenAI Chat Completions is not necessarily a
Responses endpoint. If your provider only exposes `/chat/completions`,
use `WithChatCompletion` instead. OpenAI documents these as separate
APIs in its [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).

Anthropify normalizes the subset supported by its adapters; it does not
expose every feature of the upstream SDKs, such as OpenAI's built-in
hosted tools.

## Install Anthropify

Use Go 1.24.3 or later. In a Go module:

```bash
go get github.com/pejash666/anthropify@v0.2.0
```

The following program uses APIs available in `v0.2.0`. API keys and model
identifiers come from your local environment. Pick a model supported by
the selected endpoint and available to your account.

## A Complete Streaming Example

Put this program in your application's `main.go`. Select
`LLM_PROVIDER=openai` or `LLM_PROVIDER=gemini`; only the selected
provider's key and model are required.

```go
package main

import (
    "context"
    "fmt"
    "os"
    "time"

    "github.com/anthropics/anthropic-sdk-go"
    ap "github.com/pejash666/anthropify"
)

func main() {
    if err := run(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}

func run() error {
    provider := os.Getenv("LLM_PROVIDER")
    if provider == "" {
        provider = "openai"
    }

    var opts []ap.Option
    var model string
    switch provider {
    case "openai":
        key := os.Getenv("OPENAI_API_KEY")
        upstream := os.Getenv("OPENAI_MODEL")
        if key == "" || upstream == "" {
            return fmt.Errorf("set OPENAI_API_KEY and OPENAI_MODEL")
        }
        model = "openai-demo"
        opts = []ap.Option{
            ap.WithOpenAI(ap.OpenAIResponsesConfig{APIKey: key}),
            ap.WithModelRoute(model, ap.Route{
                Provider:      ap.ProviderOpenAIResponses,
                UpstreamModel: upstream,
            }),
        }
    case "gemini":
        key := os.Getenv("GEMINI_API_KEY")
        upstream := os.Getenv("GEMINI_MODEL")
        if key == "" || upstream == "" {
            return fmt.Errorf("set GEMINI_API_KEY and GEMINI_MODEL")
        }
        model = "gemini-demo"
        opts = []ap.Option{
            ap.WithGemini(ap.GeminiConfig{
                Mode:   ap.GeminiModeStudio,
                APIKey: key,
            }),
            ap.WithModelRoute(model, ap.Route{
                Provider:      ap.ProviderGeminiNative,
                UpstreamModel: upstream,
            }),
        }
    default:
        return fmt.Errorf("LLM_PROVIDER must be openai or gemini")
    }

    client, err := ap.New(opts...)
    if err != nil {
        return err
    }

    ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
    defer cancel()

    stream, err := client.CreateMessageStream(ctx, anthropic.MessageNewParams{
        Model:     anthropic.Model(model),
        MaxTokens: 1024,
        Messages: []anthropic.MessageParam{
            anthropic.NewUserMessage(anthropic.NewTextBlock(
                "Explain protocol adapters in one short sentence.",
            )),
        },
    })
    if err != nil {
        return err
    }
    defer stream.Close()

    for stream.Next() {
        event := stream.Current()
        if event.Type == "content_block_delta" &&
            event.Delta.Type == "text_delta" {
            fmt.Print(event.Delta.Text)
        }
    }
    if err := stream.Err(); err != nil {
        return err
    }
    fmt.Println()
    return nil
}
```

Set your key and model in the local environment, then run one provider:

```bash
LLM_PROVIDER=openai go run .
```

Or, after setting `GEMINI_API_KEY` and `GEMINI_MODEL`:

```bash
LLM_PROVIDER=gemini go run .
```

These are real API calls and may incur charges. Keep keys out of source
control, screenshots and shared logs. The 60-second deadline is an
application choice, not a provider limit.

## How the Model Routes Work

`openai-demo` and `gemini-demo` are application-side aliases.
`Route.UpstreamModel` replaces the alias with the model identifier from
your environment before sending the request upstream.

This avoids assuming that every OpenAI model name matches a built-in
prefix. Without explicit routes, Anthropify routes `gpt-`, `o1-` and
`o3-` to OpenAI Responses and `gemini-` to Gemini Native. Provider
registration alone does not add routes for arbitrary model names.

Routes are case-insensitive prefix matches, not exact matches. More
specific prefixes win. The application's aliases should therefore be
chosen so they do not unintentionally match other model names.

## Configure Custom Endpoints

The program leaves `BaseURL` empty to use the default upstream hosts.
If you override it, use the base expected by the selected adapter:

- **OpenAI Responses:** `WithOpenAI` appends `/v1/responses`. Set
  `BaseURL` to the host or gateway base, not a URL already ending in
  `/v1` or `/v1/responses`.
- **OpenAI Chat Completions:** `WithChatCompletion` appends
  `/chat/completions`. A typical base is `https://api.openai.com/v1`.
  Register a named backend and route models to
  `ProviderChatCompletions` with that `Backend`.
- **Azure OpenAI:** use `WithAzureOpenAI`, not a vanilla OpenAI
  `BaseURL` override. Azure needs its resource URL, API version and
  deployment configuration. See the
  [Azure integration](../../README.md#azure-openai-integration).
- **Gemini:** the adapter builds the mode-specific model path. Use a
  host or gateway base that supports that path, not a complete
  `streamGenerateContent` URL.

Changing `BaseURL` changes where your prompts and credentials are sent.
Only configure endpoints you trust.

## Gemini AI Studio, Express and Vertex AI

The example explicitly selects `GeminiModeStudio`, which sends an
AI Studio key to Google's native API.

For other Google authentication modes, change the Gemini configuration:

- `GeminiModeExpress` uses a Vertex Express API key in `APIKey`.
- `GeminiModeVertex` requires `Project`, `Location` and a Bearer access
  token in `APIKey`. The current adapter accepts a token string;
  applications are responsible for refreshing expiring tokens and
  updating their client configuration.
- `GeminiModeAuto` selects Vertex when project and location are set;
  otherwise it selects Studio. It does not automatically select Express.

See [Gemini provider configuration](../../README.md#gemini-provider-configuration)
for the configuration examples.

## Completed Responses and Conversation History

If you do not need incremental output, use
`client.CreateMessage(ctx, req)` with the same
`anthropic.MessageNewParams`. OpenAI Responses and Gemini Native are
stream-backed in this library: `CreateMessage` drains the stream and
assembles an `*anthropic.Message`.

Your application owns the conversation history. To continue a text
conversation, append the assistant's text as an
`anthropic.MessageParam`, then append the next user message. Anthropify
does not expose an upstream conversation ID as a shared cross-provider
session. See [switching providers mid-conversation](./switch-llm-providers-in-go.md).

## Compatibility and Troubleshooting

- **Unknown model:** add a matching route or use an alias with
  `UpstreamModel`. `ErrUnknownModel` and
  `ErrProviderNotConfigured` indicate different configuration problems.
- **HTTP 401 or 403:** check the selected endpoint, authentication mode
  and account permissions. Do not print the API key while debugging.
- **HTTP 404:** check both the endpoint path and model identifier.
  Responses and Chat Completions require different adapters.
- **Thinking:** this minimal example only prints text and does not
  enable thinking explicitly. When thinking is enabled, the Gemini
  adapter emits `thinkingLevel`; the repository documents this path for
  Gemini 3.x, rather than 2.5 models that reject that field.
- **Tools:** `SchemaPolicyStrict` is the default. An incompatible
  tool schema can return `ErrSchemaIncompatible`; do not switch to a
  lossy policy without understanding the transformation.
- **Caching:** `SetCacheControl` is a documented no-op on the OpenAI
  Responses and Gemini adapters. Upstream caching eligibility and
  behavior remain provider-specific.
- **History portability:** a shared request shape does not guarantee
  that reasoning signatures, images or arbitrary tool transcripts can
  be replayed across every model.

The program was compile-checked, not run against live provider accounts
as part of writing this guide. Model access and upstream behavior must
be verified in your deployment.

## Related Guides and Source

- [Embedded LLM library or proxy server?](./llm-library-vs-proxy.md)
- [Streaming events example](../../examples/06-streaming-events)
- [Portable tool use example](../../examples/04-tool-use-portable)
- [Public provider options](../../options.go)
- [OpenAI Responses adapter](../../adapter/openai_responses/adapter.go)
- [Gemini Native adapter](../../adapter/gemini_native/adapter.go)
- [API reference](https://pkg.go.dev/github.com/pejash666/anthropify)
