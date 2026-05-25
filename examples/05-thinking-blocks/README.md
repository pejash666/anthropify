# 05 - Thinking Blocks

![header](./header.png)

Anthropic, Kimi, and Gemini all stream a canonical event sequence; what
each model puts into a `thinking` block is provider-specific behaviour,
but the event vocabulary is the same across all three.

## What it shows

- A reasoning question is sent to three thinking-capable models:
  - `claude-sonnet-4-5` with extended thinking enabled (budget 12000).
  - `kimi-k2-thinking` (server-side default thinking, no flag needed).
  - `gemini-3.5-flash` with thinking enabled via the same
    canonical `Thinking` field; the adapter maps the budget to a
    `thinkingLevel` bucket on Gemini's `generationConfig`.
- The example walks `content_block_start` / `content_block_delta` /
  `content_block_stop` events and labels each delta as either
  `thinking_delta` or `text_delta`.
- It prints per-model totals so you can see how much budget each one
  spends on thinking vs final text.

## Run

```bash
set -a; source .env.e2e; set +a
go run ./examples/05-thinking-blocks
```

Required env: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `KIMI_API_KEY`.

> Anthropify recommends `gemini-3.5-flash` (GA, supports thinking on
> AI Studio billing-enabled projects). Any Gemini 3.x flash variant
> works. Pro-series models (`gemini-3-pro-preview`,
> `gemini-3.1-pro-preview`) require explicit Google preview enrolment.

## Sample Output

```
=== Anthropic (claude-sonnet-4-5) ===
[block 0 start: thinking]
[block 0 stop: thinking]
[block 1 start: text]
[block 1 stop: text]
totals: thinking=1696 chars, text=1391 chars

=== Kimi (kimi-k2-thinking) ===
[block 0 start: thinking]
[block 0 stop: thinking]
[block 1 start: text]
[block 1 stop: text]
totals: thinking=4640 chars, text=1729 chars

=== Gemini (gemini-3.5-flash) ===
[block 0 start: thinking]
[block 0 stop: thinking]
[block 1 start: text]
[block 1 stop: text]
totals: thinking=2104 chars, text=1683 chars
```

## Source

[`main.go`](./main.go)
