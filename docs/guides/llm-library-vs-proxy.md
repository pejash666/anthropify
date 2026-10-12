# Embedded LLM Library or Proxy Server? A Guide for Go Applications

Use an embedded library when a Go application needs protocol translation
inside its own process. Use a proxy or gateway when multiple callers need
a shared HTTP service and centrally enforced access or usage policies.

Anthropify addresses the first problem: it lets Go applications call
multiple LLM providers with Anthropic Messages API types. It is not a
replacement for every feature of an LLM gateway.

## What Anthropify Actually Provides

Anthropify runs as a dependency in your Go application. The caller uses
`anthropic.MessageNewParams`; the selected adapter builds the upstream
request and normalizes responses into Anthropic-shaped events.

The normal request path is:

```text
Go application
  -> anthropify.Client
  -> model route and provider adapter
  -> upstream LLM API
```

The library supplies:

- Anthropic-shaped streaming and completed message types.
- Adapters for Anthropic, OpenAI Responses, Gemini Native and
  OpenAI-compatible Chat Completions, including documented compatible
  backend configurations such as Bedrock and Azure OpenAI.
- Named backends and prefix-based model routes, with optional upstream
  model-name rewriting.
- Tool-schema normalization with explicit strict or lossy policies.
- Provider-specific handling of thinking and prompt caching.

Anthropify does not ship an HTTP server, daemon, YAML configuration
service, shared user database or centralized billing system.

## When an Embedded Library Fits

Choose Anthropify when these conditions match your application:

- The caller is written in Go.
- You want Anthropic's message and content-block event shapes as the
  application's canonical format.
- Your application can own credentials, history, cancellation and
  provider configuration.
- You want to consume multiple provider protocols without deploying an
  additional translation service.
- You can implement any required tenant policies and operational
  controls in the surrounding application.

For example, a Go agent service may keep one canonical text history and
select a configured model per turn. Anthropify handles protocol
conversion; the service still decides which user may call that model,
how much they may spend and how to handle failures.

Start with [using OpenAI and Gemini through Anthropic types](./anthropic-api-with-openai-gemini.md)
or [switching providers mid-conversation](./switch-llm-providers-in-go.md).

## When a Proxy or Gateway Fits

A proxy adds a network service between applications and upstream APIs:

```text
Go, Python, JavaScript or other callers
  -> shared LLM proxy / gateway
  -> upstream LLM APIs
```

This architecture is useful when you need:

- One HTTP interface for callers in several languages.
- Central storage of upstream credentials instead of distributing them
  to each caller.
- Shared access controls, tenant quotas or budgets.
- Centrally managed routing, failover or provider policy.
- A shared place to inspect usage and operational events.

These are requirements to evaluate, not capabilities guaranteed by the
word "proxy". Check the candidate's actual edition, configuration,
authentication model and deployment before relying on any of them.

[LiteLLM](https://docs.litellm.ai/docs/) is one project to evaluate for
a shared gateway. Compare the supported HTTP protocols and the features
you need using its own documentation. This guide makes no performance,
cost or feature-parity claim between LiteLLM and Anthropify.

## Responsibilities Stay Somewhere

An embedded library and a gateway place responsibilities in different
components; neither removes them.

### Credentials and Access

With Anthropify, your application supplies upstream credentials when
constructing the client. It must store them securely and prevent users
from selecting unauthorized backends or endpoints.

With a gateway, callers may use gateway-issued credentials while
upstream keys stay in the service. The gateway and its administrator
must then enforce the intended separation.

Neither approach eliminates the need to understand upstream data
retention and processing. A gateway is an additional component that
handles prompts, responses and potentially sensitive logs.

### Conversation State

Anthropify does not own a durable conversation store. Your application
keeps messages and decides which parts can be replayed. A common type
does not make every provider's thinking signatures or tool history
interchangeable.

A proxy may be stateless or offer a conversation store. Do not assume
that an HTTP gateway automatically provides durable history.

### Retries, Quotas and Failover

Anthropify's prefix routes choose configured backends; they are not an
automatic failover or load-balancing policy. `WithModelOverride` lets
the application provide a routing decision, but does not supply a
complete policy engine.

Applications still need to decide whether a failed or interrupted
request is retryable. Retrying after partial streamed output can
duplicate output or charges, and replaying a tool call can repeat
side effects. Gateway retry settings also need to be checked against
these application-level constraints.

### Observability

Anthropify accepts an application logger through `WithLogger`; it does
not provide a centralized request dashboard or usage ledger. Integrate
the observability your application requires without leaking API keys
or prompt content.

A gateway can be a shared observation point, but you must validate
what it records, redacts and retains.

## A Library and Gateway Can Coexist

The choice is not always exclusive. A Go application can use Anthropify
for its canonical message shape and send translated requests to a
trusted gateway that exposes the expected upstream protocol.

For example, the OpenAI Responses adapter can target a gateway that
supports `/v1/responses` by configuring its base URL. A gateway that
only exposes Chat Completions needs the Chat Completions adapter.
Compatible authentication and streaming behavior must also be tested.

Anthropify does not make an arbitrary gateway Anthropic-compatible,
and its Go client does not expose `/v1/messages` for external callers.
See the [endpoint configuration guide](./anthropic-api-with-openai-gemini.md#configure-custom-endpoints).

## Questions to Answer Before Choosing

1. Is the consumer one Go application or several applications in
   different languages?
2. Which canonical API shape does the application actually need?
3. Where should upstream keys, tenant permissions and usage limits live?
4. Who owns durable history, tool execution and retry decisions?
5. Does the chosen adapter or gateway support the exact endpoint,
   model, streaming behavior and authentication mode you will deploy?
6. Are thinking, tools, images and prompt caching compatible with that
   specific configuration?
7. Which operational controls must exist before production traffic?

For a Go application that owns those controls and wants Anthropic-shaped
messages, Anthropify is a focused protocol dependency. For a shared
network interface with centralized policies, evaluate a gateway.

## Implementation References

- [Client dispatch and completed-response handling](../../client.go)
- [Provider options and model routes](../../options.go)
- [Route resolution](../../router.go)
- [Tool-schema policy](../design/v0.2.0-protocol-driven-adapters.md)
- [Prompt-caching behavior](../../README.md#prompt-caching)
- [API reference](https://pkg.go.dev/github.com/pejash666/anthropify)
