> 中文版 | [English](./README.md)

<p align="center"><img src="./assets/banner.png" alt="Anthropify" width="640"></p>

<h1 align="center">Anthropify</h1>

<p align="center"><i>用 Anthropic 的 API 形态调用所有 LLM，Go 原生。</i></p>

<p align="center">
<a href="https://pkg.go.dev/github.com/shahao/anthropify"><img src="https://pkg.go.dev/badge/github.com/shahao/anthropify.svg" alt="Go Reference"></a>
<a href="https://go.dev/"><img src="https://img.shields.io/badge/go-1.24%2B-00ADD8" alt="Go version"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
</p>

## 这是什么？

Anthropify 是一个 Go 库，让你用 Anthropic Messages API 的形态去调用 OpenAI、
Gemini、Kimi、GLM、DeepSeek 等多家 LLM 提供商。它在进程内完成协议转换、SSE
流式传输、tool use、thinking 块和 `cache_control` —— 前面不需要任何代理服务。
调用方始终只面对 `anthropic.MessageNewParams` 和
`anthropic.MessageStreamEventUnion`。

## 快速开始

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/anthropics/anthropic-sdk-go"
    ap "github.com/shahao/anthropify"
)

func main() {
    client, err := ap.New(
        ap.WithChatCompletion("kimi", ap.OpenAICompatConfig{
            BaseURL: "https://api.moonshot.cn/v1",
            APIKey:  os.Getenv("KIMI_API_KEY"),
        }),
        ap.WithModelRoute("kimi-", ap.Route{
            Provider: ap.ProviderChatCompletions,
            Backend:  "kimi",
        }),
    )
    if err != nil {
        panic(err)
    }

    stream, err := client.CreateMessageStream(context.Background(), anthropic.MessageNewParams{
        Model:     "kimi-k2.6",
        MaxTokens: 1024,
        Messages: []anthropic.MessageParam{
            anthropic.NewUserMessage(anthropic.NewTextBlock("Hello, who are you?")),
        },
    })
    if err != nil {
        panic(err)
    }
    defer stream.Close()

    for stream.Next() {
        fmt.Printf("%+v\n", stream.Current()) // anthropic.MessageStreamEventUnion
    }
    if err := stream.Err(); err != nil {
        panic(err)
    }
}
```

非流式调用使用同一种请求类型，返回 `*anthropic.Message`：

```go
msg, err := client.CreateMessage(ctx, req)
```

## 实战演示

Anthropify 的核心主张是：一份 Anthropic 规范形态的对话切片，可以在中途任意切
换 provider。为了证明这一点，[example 03](./examples/03-model-hot-switching)
让一段对话先后流过四个上游——Claude 规划、Kimi 写代码、GLM 翻译、Claude 收
尾——全程不重建消息历史。每一轮唯一变化的字段是 `req.Model`。

<p align="center"><img src="./examples/03-model-hot-switching/flow.png" alt="model hot-switching" width="720"></p>

```go
var messages []anthropic.MessageParam
for _, t := range turns {
    messages = append(messages, anthropic.NewUserMessage(anthropic.NewTextBlock(t.question)))
    reply, _ := runTurn(client, t.model, t.maxTokens, messages) // 只 Model 变
    messages = append(messages, anthropic.NewAssistantMessage(anthropic.NewTextBlock(reply)))
}
```

> 完整可运行版本见 [examples/03-model-hot-switching](./examples/03-model-hot-switching)。

## 为什么选 Anthropic 作为标准形态？

各家流式协议差异很大，选什么形态作为 canonical event 会影响下游所有 agent 代码。

- Anthropic 的流是一组语义清晰、有序、有结构边界的事件：`message_start`、
  `content_block_start`、`content_block_delta`、`content_block_stop`、
  `message_delta`、`message_stop`。文本、tool use、thinking 各占独立的 content
  block。
- OpenAI Chat Completions 把所有内容塞进 `choices[*].delta`。工具调用、
  reasoning、refusal 都是边角字段，不同类型的内容之间没有结构性分隔。
- OpenAI Responses 把同一条流拆得很细（`response.output_item.added`、
  `response.content_part.added`、`response.output_text.delta` 等），额外的粒度
  带来记账负担，但并没有让线上形态更易消费。

对一个需要稳定内部事件词汇的 agent 框架来说，Anthropic 的形态在粒度曲线上落在
一个合用的点。Anthropify 把它作为规范形态，并把所有 provider 都归一化到它。

## 支持的 Provider

| Provider          | 流式  | 非流式                  | Tool use | Thinking         | cache_control |
|-------------------|-------|--------------------------|----------|------------------|---------------|
| Anthropic         | 是    | 原生                     | 是       | 是               | 是            |
| OpenAI Responses  | 是    | drain-and-assemble       | 是       | 是 (reasoning)   | —             |
| Gemini Native     | 是    | drain-and-assemble       | 是       | 是               | 部分          |
| Kimi (Moonshot)   | 是    | drain-and-assemble       | 是       | 是               | —             |
| GLM (智谱)        | 是    | drain-and-assemble       | 是       | 视模型而定       | —             |
| DeepSeek          | 是    | drain-and-assemble       | 是       | 是 (V3.2)        | —             |

> Anthropify 推荐使用 `gemini-3.5-flash`（或任意 Gemini 3.x flash 系列）。
> Adapter 在 `generationConfig.thinkingConfig` 中发送 `thinkingLevel`，
> 2.5 系列会以 HTTP 400 拒绝，因此请使用 3.x 系列。Pro 系列模型
> （`gemini-3-pro-preview`、`gemini-3.1-pro-preview`）需要 Google
> 显式 preview 准入，开箱即用拿不到。

已在 `gemini-3.5-flash`、`gemini-3.1-pro-preview`、`gemini-3-pro-preview`、
`gemini-pro-latest` 上实测 thinking 功能——四个 model 都返回 canonical
thinking blocks，并支持 `thoughtSignature` 跨轮回放（前提：AI Studio
project 已开启 billing）。

只有 Anthropic adapter 直接暴露原生非流式接口。其他 provider 在调用
`CreateMessage` 时，会打开流并由库内部把 content block 组装成最终的
`*anthropic.Message`。

路由基于 `req.Model` 的前缀：

| 前缀                | Provider            |
|---------------------|---------------------|
| `claude-`           | `anthropic`         |
| `gpt-`、`o1-`、`o3-`| `openai_responses`  |
| `gemini-`           | `gemini_native`     |

Kimi / GLM / DeepSeek 等 OpenAI 兼容上游通过 `WithChatCompletion(name, ...)`
注册，并用
`WithModelRoute(prefix, Route{Provider: ProviderChatCompletions, Backend: name})`
路由。

## Gemini 三种鉴权模式

Gemini adapter 支持三种鉴权方式、两个上游 host。模式通过 `GeminiMode` 显式指
定；`GeminiModeAuto` 保留原有启发式（同时配 project+location 即 Vertex，否则
Studio）。

| Mode    | 适用场景                  | Endpoint                                                     | 鉴权                              |
|---------|---------------------------|--------------------------------------------------------------|-----------------------------------|
| Studio  | 免费层、原型               | `generativelanguage.googleapis.com/v1beta`                   | API key (`AIza...`)，走 `?key=`   |
| Express | 用 API key 调 Vertex      | `aiplatform.googleapis.com/v1/publishers/google`             | API key (`AQ.*`)，走 `?key=`      |
| Vertex  | 生产、GCP 原生             | `aiplatform.googleapis.com/v1/projects/<p>/locations/<l>`    | OAuth2 Bearer token               |

```go
// Studio
ap.WithGemini(ap.GeminiConfig{
    Mode:   ap.GeminiModeStudio,
    APIKey: os.Getenv("GEMINI_API_KEY"),
})
```

```go
// Express（用 API key 调 Vertex）
ap.WithGemini(ap.GeminiConfig{
    Mode:   ap.GeminiModeExpress,
    APIKey: os.Getenv("VERTEX_EXPRESS_KEY"), // AQ.*
})
```

```go
// Vertex（OAuth2）
ap.WithGemini(ap.GeminiConfig{
    Mode:     ap.GeminiModeVertex,
    Project:  os.Getenv("VERTEX_PROJECT"),
    Location: os.Getenv("VERTEX_LOCATION"),
    APIKey:   bearerToken, // gcloud auth print-access-token
})
```

## 安装

```bash
go get github.com/shahao/anthropify
```

需要 Go 1.24 或更高版本。

## 与其他方案的对比

Anthropify 的边界是有意收窄的：一个把 Anthropic 规范事件和 provider 原生协
议互相转换的 Go 库。它不是代理服务、不是路由层、不是多租户网关，仓库里没有
`cmd/`、没有 YAML、没有 daemon。

如果你需要一个开箱即用的托管代理，带鉴权、限流、计费和多租户，LiteLLM 是更完
整的方案，并以 Python 为主。如果你需要一个 Go 原生、嵌进 agent 里用的库，要
求一等公民级别的流式、tool use 支持，并且事件形态对齐 Anthropic 规范，那
Anthropify 就是为此而生。

## 状态

> **alpha**。API 可能调整。四条 adapter 链路都跑过真实 API 测试（25 个 E2E 测
> 试，0 失败，覆盖 Anthropic / OpenAI / Gemini / Kimi / GLM / DeepSeek）。生
> 产使用请固定到具体 commit。

## 测试

单元测试基于 fixture，不会触网络：

```bash
go test ./...
```

真实网络的 smoke 测试位于 `e2e/`，由 build tag 隔离：

```bash
cp .env.e2e.example .env.e2e   # 只填你手上有 key 的 provider
make test-e2e                  # 加载 .env.e2e 并运行 `go test -tags=e2e ./e2e/...`
```

未配置 API key 的 provider 会自动跳过。具体矩阵见 `e2e/README.md`。`.env.e2e`
已被 git 忽略，仓库里只追踪 `.env.e2e.example`。

## 贡献

欢迎 issue 和 pull request。发 PR 前请确认：

- `go test ./...` 通过。
- 改动了 adapter 行为时，请在 `e2e/` 增加对应 E2E 测试，并用真实 key 跑过
  `make test-e2e`。
- 公共 API 尽量复用 `github.com/anthropics/anthropic-sdk-go` 的类型别名，不要
  另起一套请求/响应结构。

## 许可证

MIT，详见 [LICENSE](./LICENSE)。
