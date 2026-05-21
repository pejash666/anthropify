//go:build e2e

// Package helpers holds reusable prompt fixtures and assertion utilities
// for the E2E suite. Keeping prompts in one place ensures the matrix
// test compares apples to apples across providers.
package helpers

import (
	"encoding/json"

	"github.com/anthropics/anthropic-sdk-go"
)

// BasicTextPrompt is the smoke prompt sent in TestE2E_<Provider>_BasicStream.
// Single short user turn, no tools, no system message.
func BasicTextPrompt(model string, maxTokens int64) anthropic.MessageNewParams {
	src := `{
		"model": "` + model + `",
		"max_tokens": ` + itoa(maxTokens) + `,
		"messages": [
			{"role":"user","content":[{"type":"text","text":"用1句话介绍 Go 语言。"}]}
		]
	}`
	return mustParams(src)
}

// MaxTokensClipPrompt forces stop_reason=max_tokens by setting
// max_tokens very low against an open-ended prompt.
func MaxTokensClipPrompt(model string) anthropic.MessageNewParams {
	src := `{
		"model": "` + model + `",
		"max_tokens": 10,
		"messages": [
			{"role":"user","content":[{"type":"text","text":"Write a 500-word essay about the history of the Roman Empire."}]}
		]
	}`
	return mustParams(src)
}

// WeatherToolPrompt is the single-round tool_use fixture. The model is
// expected to emit a tool_use block targeting get_weather.
func WeatherToolPrompt(model string) anthropic.MessageNewParams {
	src := `{
		"model": "` + model + `",
		"max_tokens": 256,
		"tools": [
			{
				"name": "get_weather",
				"description": "Look up the current weather for a city.",
				"input_schema": {
					"type": "object",
					"properties": {
						"city": {"type": "string", "description": "City name, e.g. Beijing"}
					},
					"required": ["city"]
				}
			}
		],
		"messages": [
			{"role":"user","content":[{"type":"text","text":"What is the weather in Beijing right now? Use the tool."}]}
		]
	}`
	return mustParams(src)
}

// WeatherToolFollowUpPrompt is the second turn for the multi-round
// tool-use test. The caller fills in a fake tool_use_id from the first
// response. The model is then expected to summarise the tool result.
func WeatherToolFollowUpPrompt(model, toolUseID string) anthropic.MessageNewParams {
	src := `{
		"model": "` + model + `",
		"max_tokens": 256,
		"tools": [
			{
				"name": "get_weather",
				"description": "Look up the current weather for a city.",
				"input_schema": {
					"type": "object",
					"properties": {"city": {"type":"string"}},
					"required": ["city"]
				}
			}
		],
		"messages": [
			{"role":"user","content":[{"type":"text","text":"What is the weather in Beijing right now? Use the tool."}]},
			{"role":"assistant","content":[{"type":"tool_use","id":"` + toolUseID + `","name":"get_weather","input":{"city":"Beijing"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + toolUseID + `","content":"21C, sunny"}]}
		]
	}`
	return mustParams(src)
}

func itoa(n int64) string {
	// Tiny helper to keep prompt JSON readable. strconv.FormatInt would
	// pull in another import for one call site.
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func mustParams(src string) anthropic.MessageNewParams {
	var p anthropic.MessageNewParams
	if err := json.Unmarshal([]byte(src), &p); err != nil {
		// Prompts are constants; a parse failure is a programming bug.
		panic(err)
	}
	return p
}
