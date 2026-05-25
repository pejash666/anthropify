# 02 - Multi-Provider

![header](./header.png)

One client, five providers, two protocols, prefix routing. The
application code is identical across providers; only the model name
changes.

## What it shows

- A single `anthropify.New(...)` registers five backends across two
  wire protocols:
  - Anthropic (anthropic protocol, default backend)
  - **MiniMax** (anthropic protocol, second backend via
    `WithAnthropicCompat("minimax", …)`)
  - Gemini (gemini_native, AI Studio)
  - Kimi (chat-completions compatible)
  - GLM (chat-completions compatible)
- `WithModelRoute("kimi-", …)`, `WithModelRoute("glm-", …)`, and
  `WithModelRoute("MiniMax-", …)` tell the dispatcher which named
  backend serves a given prefix. Anthropic / OpenAI / Gemini are
  routed by their built-in prefixes.
- The same `CreateMessageStream` call is dispatched to a different
  upstream solely on the basis of `req.Model`. Two of those upstreams
  (Claude and MiniMax) share the *same* in-process adapter; the rest
  ride distinct adapters — the user never has to think about it.

## Run

```bash
set -a; source .env.e2e; set +a
go run ./examples/02-multi-provider
```

Required env: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `KIMI_API_KEY`,
`GLM_API_KEY`, `MINIMAX_API_KEY`. Optional: `*_MODEL`, `*_BASE_URL`.

## Sample Output

```
--- claude-sonnet-4-5 ---
I was trained on data up through April 2024.

--- gemini-3.5-flash ---
My training data was last updated in early 2024.

--- kimi-k2.6 ---
I was trained on data up to 2024.

--- glm-5.1 ---
I am continually learning and improving, so my model is regularly updated...

--- MiniMax-M2.7 ---
My training data goes up to mid-2024.
```

## Source

[`main.go`](./main.go)
