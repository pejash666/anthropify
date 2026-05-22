# E2E tests

End-to-end smoke tests that exercise every adapter against its real
upstream provider. The suite is hidden behind the `e2e` build tag, so
`go test ./...` will neither build nor run it.

## Setup

```sh
cp .env.e2e.example .env.e2e
# Fill in only the keys for providers you want to exercise. Missing
# keys cause that provider's tests to skip; nothing fails.
```

Variables of interest:

| Env var               | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `ANTHROPIC_API_KEY`   | Native Anthropic adapter                                 |
| `OPENAI_API_KEY`      | OpenAI Responses (GPT-5)                                 |
| `GEMINI_API_KEY`      | Gemini AI Studio. Skip when using Vertex.                |
| `VERTEX_PROJECT` / `VERTEX_LOCATION` | Gemini Vertex (uses ADC at runtime)       |
| `KIMI_API_KEY`        | Chat Completions backend (Moonshot)                      |
| `GLM_API_KEY`         | Chat Completions backend (Zhipu BigModel)                |
| `E2E_RECORD_FIXTURES` | When `true`, mirrors SSE bytes to `testdata/fixtures/recorded/`. Reserved hook -- not yet wired. |

`.env.e2e` is **git-ignored**; only `.env.e2e.example` is tracked.

## Running

```sh
make test-e2e            # loads .env.e2e and runs `go test -tags=e2e ./e2e/... -count=1 -v`
make test-e2e-record     # same, but with E2E_RECORD_FIXTURES=true
```

You can also invoke `go test` directly once the env is loaded:

```sh
set -a; source .env.e2e; set +a
go test -tags=e2e ./e2e/... -run TestE2E_Matrix_Consistency -v
```

## Layout

```
e2e/
├── env.go                     // env-var loaders + skip helper
├── clients.go                 // per-provider anthropify.Client builders
├── matrix_test.go             // cross-provider consistency test
├── anthropic_test.go
├── openai_responses_test.go
├── gemini_native_test.go
├── chat_completions_kimi_test.go
├── chat_completions_glm_test.go
└── helpers/
    ├── prompts.go             // shared prompt fixtures
    └── assertions.go          // stream draining + assertions
```

## Per-provider tests

Every provider has five tests:

1. `BasicStream` -- short text prompt, asserts envelope + text + stop reason + usage.
2. `ToolUseSingleRound` -- weather tool definition, expects a `tool_use` block.
3. `ToolUseMultiRound` -- replays `tool_result`, expects a textual summary.
4. `NonStreaming` -- routes through `CreateMessage` (drain path for OpenAI/Gemini).
5. `StopReasonMapping` -- `max_tokens=10` against an open prompt -> `stop_reason=max_tokens`.

The matrix test (`TestE2E_Matrix_Consistency`) runs only the basic
prompt across every configured provider and asserts consistent shape.

## Notes

- No third-party deps. Env loading is `set -a; source .env.e2e; set +a`
  in `scripts/test-e2e.sh`.
- Fixture recording is a reserved hook (`helpers.RecordingTransport`);
  the writer is not yet implemented.
- Tests never compare provider outputs against a golden string -- LLM
  responses vary. They only assert structural invariants.
