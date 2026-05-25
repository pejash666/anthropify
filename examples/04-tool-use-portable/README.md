# 04 - Tool Use Portable

![header](./header.png)

Define a tool once in Anthropic's schema. Call it from four different
providers without touching the schema.

## What it shows

- One `anthropic.ToolUnionParam{...}` value with a `get_weather(city)`
  function definition.
- The same client + tools are sent to Anthropic, OpenAI, Gemini, and
  Kimi. Each adapter translates the canonical tool definition into the
  upstream's native function-calling format.
- Round 1: every provider emits a `tool_use` content block with the
  same shape (`name`, `input` JSON, `id`).
- The example mocks a weather API result.
- Round 2: the `tool_result` is fed back as a user-message content
  block, and every provider produces a natural-language reply.

## Run

```bash
set -a; source .env.e2e; set +a
go run ./examples/04-tool-use-portable
```

Required env: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
`KIMI_API_KEY`.

## Sample Output

```
=== Anthropic (claude-sonnet-4-5) ===
round1 tool_use: name=get_weather input=map[city:Beijing]
round2 final: The current weather in Beijing is:
- Temperature: 21°C
- Condition: Sunny

=== OpenAI (gpt-5-mini) ===
round1 tool_use: name=get_weather input=map[city:Beijing]
round2 final: Right now in Beijing it's 21°C (≈70°F) and sunny....

=== Gemini (gemini-2.5-flash) ===
round1 tool_use: name=get_weather input=map[city:Beijing]
round2 final: The weather in Beijing is currently sunny with a temperature of 21 degrees Celsius.

=== Kimi (kimi-k2-thinking) ===
round1 tool_use: name=get_weather input=map[city:Beijing]
round2 final: The weather in Beijing right now is sunny with a temperature of 21°C.
```

## Source

[`main.go`](./main.go)
