# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0-alpha] - 2026-05-25

First public pre-release. Establishes the multi-backend Anthropic-protocol
gateway, formalises canonical-layer features (schema dialects, cache control),
and adds two new transport backends (AWS Bedrock, Azure OpenAI Responses).

### BREAKING

- **Module renamed** from `github.com/shahao/hybridstream` to
  `github.com/pejash666/anthropify`. Downstream callers must update import paths
  and option-package aliases (`hybridstream.With...` → `anthropify.With...`).
- **Multi-backend registries** replace the single-backend option model. Each
  protocol family (`Anthropic`, `OpenAIResponses`, `AnthropicCompat`,
  `ChatCompletions`, `GeminiNative`, `AnthropicBedrock`, `AzureOpenAI`) is
  registered as a named map. The first registered name in a family is the
  default for that family. Existing single-call options remain compatible
  through implicit `"default"` naming, but programs that previously relied on
  registering two providers of the same family without a name must now supply
  explicit names.

### Added

- **Track A — Multi-backend protocol registries.** `Client` now holds a map
  per protocol family. New options:
  - `WithAnthropic(name, AnthropicConfig)`
  - `WithOpenAIResponses(name, OpenAIResponsesConfig)`
  - `WithAnthropicCompat(name, AnthropicCompatConfig)` (DeepSeek, MiniMax, …)
  - `WithChatCompletions(name, ChatCompletionsConfig)` (Kimi, GLM, …)
  - `WithGeminiNative(name, GeminiNativeConfig)`
- **Track B — Schema dialect policy.** New `internal/schema` package and
  public `SchemaPolicy` enum (`SchemaPolicyAuto`, `SchemaPolicyPortable`,
  `SchemaPolicyNative`) on tool input schemas. The canonical layer now
  rewrites JSON-Schema draft-7 fragments to per-backend dialects (OpenAI
  Responses strict, Gemini OpenAPI 3.0 subset, Anthropic pass-through). Adds
  `TestE2E_SchemaDialect_Portable` matrix coverage.
- **Track C — Top-level cache_control.** Canonical request now carries an
  optional top-level `cache_control` array of `{type:"ephemeral"}` markers
  that adapters lower into Anthropic-style breakpoints. Includes
  `TestE2E_Anthropic_CacheControl_TopLevel` (round-trip cache_create →
  cache_read against live Anthropic).
- **MiniMax provider** wired through the `anthropic-compat` backend with
  full README/example coverage and matrix-consistency e2e
  (`anthropic_minimax`).
- **Track D — AWS Bedrock backend.** New `WithAnthropicBedrock(name,
  BedrockConfig)` option. Speaks the Anthropic protocol natively over Bedrock
  signed transport. Documented in README sections; no e2e (no Bedrock test
  account).
- **Track E — Azure OpenAI Responses adapter.** New `WithAzureOpenAI(name,
  AzureOpenAIConfig)` option. Reuses the OpenAI Responses adapter behind the
  Azure Responses endpoint; design doc + README sections shipped. No e2e (no
  Azure test account).
- **Examples**:
  - `examples/02-multi-provider` extended with MiniMax.
  - `examples/07-anthropic-compat` showcasing DeepSeek + MiniMax through the
    same backend.
- **Documentation**: design docs under `docs/design/` for v0.2.0, schema
  dialects, top-level cache_control, AWS Bedrock, and Azure OpenAI
  Responses.

### Changed

- **Default models** bumped:
  - Kimi → `kimi-k2.6` (was older Moonshot default).
  - GLM → `glm-5.1` (was older Zhipu default).
  - Gemini thinking-tested set noted in README (`gemini-3.5-pro`,
    `gemini-3.5-flash`).
- **Gemini native adapter**: preserves block order across multi-round
  thought_signature replays; matrix tests now exercise the multi-round path.
- **ChatCompletions adapter**: emits final usage on `message_delta` per
  Anthropic spec; DeepSeek V3.2 canonical name aligned.
- **Client**: `NonStreaming` now drains and assembles via the streaming
  pipeline across every adapter, with shared unit tests.

### Fixed

- Gemini multi-round tool-use tests now correctly replay
  `thought_signature` blocks (one-shot retry covers transient empty
  candidates).
- `chat_completions` final-event ordering matches Anthropic SSE
  expectations.
- `gemini_native` emits `usage` and `stop_reason` on `message_delta` per the
  Anthropic protocol.

### Removed

- The single-backend option layout from the pre-0.2 scaffold (superseded by
  named registries; see BREAKING above).

[Unreleased]: https://github.com/pejash666/anthropify/compare/v0.2.0-alpha...HEAD
[0.2.0-alpha]: https://github.com/pejash666/anthropify/releases/tag/v0.2.0-alpha
