# Migrate from Anthropify to Turnmeld

Turnmeld is the new name of Anthropify. Version v0.3.0 is the first release
with the Go module path `github.com/pejash666/turnmeld` and root package
name `turnmeld`.

This is a module and package rename, not a new protocol implementation.
Provider options, exported type names and canonical Anthropic SDK
message/event types are unchanged.

## Update Your Dependency

Use Go 1.24.3 or later:

```sh
go get github.com/pejash666/turnmeld@v0.3.0
```

Replace every `github.com/pejash666/anthropify` import with
`github.com/pejash666/turnmeld`, including imports of `adapter` and its
subpackages.

For example, before:

```go
import ap "github.com/pejash666/anthropify"

client, err := ap.New(ap.WithAnthropic(ap.AnthropicConfig{APIKey: key}))
```

After, keeping the explicit alias minimizes code changes:

```go
import ap "github.com/pejash666/turnmeld"

client, err := ap.New(ap.WithAnthropic(ap.AnthropicConfig{APIKey: key}))
```

The current examples use `tm` as the alias. If your code uses unaliased
imports, change package qualifiers from `anthropify.New(...)` to
`turnmeld.New(...)`.

After updating imports, run:

```sh
go mod tidy
go test ./...
```

Avoid importing both module paths in the same application. Go treats them
as separate packages: their `Client`, `Option`, adapter interfaces and
sentinel errors have distinct identities. Update dependent packages that
exchange these types together.

## Versions and Repository Redirects

- The repository is now
  [pejash666/turnmeld](https://github.com/pejash666/turnmeld).
- GitHub redirects the former repository URL, but that does not make old
  Go module versions compatible with the new module path.
- `v0.2.0` and `v0.2.0-alpha` still declare
  `github.com/pejash666/anthropify` in their `go.mod`. Do not request those
  versions under the Turnmeld path.
- To stay on the old API, keep the original import path and pin
  `github.com/pejash666/anthropify@v0.2.0`. The already published versions
  are not rewritten. Do not use `@latest` on the old module path to obtain
  Turnmeld releases.
- Historical design and release documents retain their original names
  and code snippets; use current guides for Turnmeld imports.

Update an existing checkout's remote:

```sh
git remote set-url origin https://github.com/pejash666/turnmeld.git
```

## Diagnostics and Behavior

Error text and context-key diagnostic strings now use the `turnmeld`
prefix rather than `anthropify`. Update log filters or string comparisons
that depend on the old prefix. Prefer `errors.Is` with the exported
Turnmeld sentinel errors for programmatic checks.

The library still translates protocols in-process, uses the official
Anthropic SDK's request/event types, and does not provide a proxy server.
The rename does not change streaming, tool use, thinking, model routing
or provider-specific prompt-cache behavior.

## Related Documentation

- [English README](../../README.md)
- [Chinese README](../../README.zh-CN.md)
- [Use OpenAI and Gemini through the Anthropic Messages API](anthropic-api-with-openai-gemini.md)
- [Switch providers mid-conversation](switch-llm-providers-in-go.md)
- [Changelog](../../CHANGELOG.md)
