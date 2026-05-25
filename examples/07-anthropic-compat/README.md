# 07 - Anthropic Protocol, Two Vendors

The v0.2.0 thesis demo: **protocol is not vendor**. The Anthropic
Messages API is a wire spec; any provider that speaks it can plug into
anthropify as a named backend on the *same* in-process adapter.

This example registers Anthropic Inc. (real Claude) and MiniMax
side-by-side and dispatches by model prefix:

```go
client, _ := ap.New(
    ap.WithAnthropic(ap.AnthropicConfig{APIKey: anthropicKey}),
    ap.WithAnthropicCompat("minimax", ap.AnthropicConfig{
        BaseURL: "https://api.minimax.io/anthropic",
        APIKey:  os.Getenv("MINIMAX_API_KEY"),
    }),
    ap.WithModelRoute("MiniMax-",
        ap.Route{Provider: ap.ProviderAnthropic, Backend: "minimax"}),
)
```

The application code never branches on vendor. `claude-sonnet-4-5` and
`MiniMax-M2.7` flow through the *same* `CreateMessageStream` call;
the dispatcher routes on `req.Model` and the named backend handles
auth + URL.

## Why this matters

Before v0.2.0, the anthropic adapter was single-instance — only one
Anthropic-protocol upstream per `Client`. v0.2.0 promotes it to a
multi-backend registry symmetric with `chat_completions`, so any
Anthropic-compatible provider (MiniMax today, more tomorrow) can
coexist with real Claude in one client without subclassing or
duplicating router rules.

## Run

```bash
set -a; source .env.e2e; set +a
go run ./examples/07-anthropic-compat
```

Required env: `ANTHROPIC_API_KEY`, `MINIMAX_API_KEY`. Optional:
`ANTHROPIC_MODEL`, `MINIMAX_MODEL`, `MINIMAX_BASE_URL`.

## Sample Output

```
--- claude-sonnet-4-5 ---
I was trained by Anthropic.

--- MiniMax-M2.7 ---
I was trained by MiniMax (上海稀宇科技有限公司).
```

## Source

[`main.go`](./main.go)
