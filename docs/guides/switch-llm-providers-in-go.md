# Switch LLM Providers Mid-Conversation in Go

Turnmeld is an open-source Go library that lets an application use the
Anthropic Messages API with Anthropic, OpenAI, Gemini, Kimi, GLM, DeepSeek
and other configured providers. Protocol translation runs inside the
application; there is no proxy server to deploy.

This guide walks through the repository's
[model hot-switching example](../../examples/03-model-hot-switching).
It uses four text turns across three providers: Claude plans a trip, Kimi
turns the plan into code, GLM translates it, and Claude summarizes the
conversation.

![Four conversation turns across Claude, Kimi and GLM](../../examples/03-model-hot-switching/flow.png)

## Why Keep One Message Format?

An application that calls multiple LLM APIs normally has to handle different
request bodies and streaming protocols. Anthropic has content-block events,
OpenAI Chat Completions uses deltas, and Gemini has its own response shape.

Turnmeld uses types from the official
[`anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go) as the
application-facing format:

- `anthropic.MessageNewParams` for requests.
- `anthropic.MessageParam` for conversation history.
- `anthropic.MessageStreamEventUnion` for streaming events.
- `anthropic.Message` for completed responses.

Provider adapters handle the protocol conversion. Application code can keep
its own canonical history instead of maintaining a separate history per
provider.

## Register Providers and Model Routes

Install Turnmeld with Go 1.24.3 or later:

```bash
go get github.com/pejash666/turnmeld@v0.3.0
```

The example registers Anthropic directly and Kimi and GLM through their
OpenAI-compatible Chat Completions endpoints:

```go
client, err := tm.New(
    tm.WithAnthropic(tm.AnthropicConfig{
        APIKey: os.Getenv("ANTHROPIC_API_KEY"),
    }),
    tm.WithChatCompletion("kimi", tm.OpenAICompatConfig{
        BaseURL: "https://api.moonshot.ai/v1",
        APIKey:  os.Getenv("KIMI_API_KEY"),
    }),
    tm.WithChatCompletion("glm", tm.OpenAICompatConfig{
        BaseURL: "https://open.bigmodel.cn/api/paas/v4",
        APIKey:  os.Getenv("GLM_API_KEY"),
    }),
    tm.WithModelRoute("kimi-", tm.Route{
        Provider: tm.ProviderChatCompletions,
        Backend:  "kimi",
    }),
    tm.WithModelRoute("glm-", tm.Route{
        Provider: tm.ProviderChatCompletions,
        Backend:  "glm",
    }),
)
if err != nil {
    panic(err)
}
```

Here `tm` imports `github.com/pejash666/turnmeld`. The default `claude-`
route targets Anthropic. The explicit `kimi-` and `glm-` routes select the
named backends. The runnable example also supports endpoint overrides
through `KIMI_BASE_URL` and `GLM_BASE_URL`.

Provider registration and model routing are separate: configure credentials
and endpoints once, then choose a configured model on each request.

## Reuse the Conversation History

The application owns a single `[]anthropic.MessageParam` slice. For each
turn, it appends the user's question, calls the selected model, then appends
the assistant's text for the next turn:

```go
var messages []anthropic.MessageParam

for _, t := range turns {
    messages = append(messages, anthropic.NewUserMessage(
        anthropic.NewTextBlock(t.question),
    ))

    reply, err := runTurn(client, t.model, t.maxTokens, messages)
    if err != nil {
        return
    }
    if reply == "" {
        reply = "(no text emitted; thinking budget exhausted)"
    }

    messages = append(messages, anthropic.NewAssistantMessage(
        anthropic.NewTextBlock(reply),
    ))
}
```

This excerpt uses `turns` and `runTurn` from the
[complete source](../../examples/03-model-hot-switching/main.go). The
example changes the model and token budget per turn. It does not reconstruct
the history when switching providers.

The empty-text fallback keeps this demonstration's assistant turns nonempty.
An application may instead retry, report an error or stop when no text is
returned.

## Consume the Same Streaming Events

Every adapter exposes Anthropic-shaped events to the application. The
example collects only text deltas:

```go
func runTurn(
    client *tm.Client,
    model string,
    maxTokens int64,
    messages []anthropic.MessageParam,
) (string, error) {
    stream, err := client.CreateMessageStream(
        context.Background(),
        anthropic.MessageNewParams{
            Model:     anthropic.Model(model),
            MaxTokens: maxTokens,
            Messages:  messages,
        },
    )
    if err != nil {
        return "", err
    }
    defer stream.Close()

    var out strings.Builder
    for stream.Next() {
        event := stream.Current()
        if event.Type == "content_block_delta" &&
            event.Delta.Type == "text_delta" {
            out.WriteString(event.Delta.Text)
        }
    }
    return strings.TrimSpace(out.String()), stream.Err()
}
```

The stream consumer does not need a different loop for Claude, Kimi or GLM.
For applications that need completed responses rather than streaming,
`client.CreateMessage(ctx, req)` returns `*anthropic.Message`.

## Run the Complete Example

Clone the repository and run the example from its root:

```bash
git clone https://github.com/pejash666/turnmeld.git
cd turnmeld
go run ./examples/03-model-hot-switching
```

Set `ANTHROPIC_API_KEY`, `KIMI_API_KEY` and `GLM_API_KEY` in your local
environment first. Calls use your provider accounts and may incur charges.
Do not commit credentials.

The example's default model names are `claude-sonnet-4-5`, `kimi-k2.6` and
`glm-5.1`. Availability depends on your provider account; override them with
`ANTHROPIC_MODEL`, `KIMI_MODEL` and `GLM_MODEL` as needed. Model names for
Kimi and GLM must still match their configured prefixes, or you must update
the routes.

## What This Example Does Not Guarantee

A common protocol shape does not make all model capabilities identical.

- This example reuses text history. It does not demonstrate arbitrary
  cross-provider replay of thinking blocks, signatures, images or tool
  transcripts.
- Context limits, token budgets, supported tools and model access remain
  provider-specific.
- The example consumes thinking events without saving them into the next
  turn's history. Applications that preserve reasoning state must respect
  the upstream provider's requirements.
- Prompt caching is provider-specific. For example, top-level
  `SetCacheControl` is not supported by Bedrock; use accepted per-block
  caching instead. See the
  [prompt-caching documentation](../../README.md#prompt-caching).
- Turnmeld is an embedded protocol library, not a hosted gateway with
  billing, multi-tenancy or rate limiting.

## More Examples

- [Use OpenAI and Gemini with Anthropic types](./anthropic-api-with-openai-gemini.md).
- [Choose an embedded library or proxy server](./llm-library-vs-proxy.md).
- [Configure multiple providers](../../examples/02-multi-provider).
- [Portable tool use](../../examples/04-tool-use-portable).
- [Consume thinking blocks](../../examples/05-thinking-blocks).
- [Inspect streaming events](../../examples/06-streaming-events).
- [Register Anthropic-compatible backends](../../examples/07-anthropic-compat).
- [API reference](https://pkg.go.dev/github.com/pejash666/turnmeld).
