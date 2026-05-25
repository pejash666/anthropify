package anthropify

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/anthropify/adapter"
	anthropicadapter "github.com/shahao/anthropify/adapter/anthropic"
	chatadapter "github.com/shahao/anthropify/adapter/chat_completions"
	geminiadapter "github.com/shahao/anthropify/adapter/gemini_native"
	openairesponses "github.com/shahao/anthropify/adapter/openai_responses"
)

// Client is the top-level entry point. Construct one with New.
type Client struct {
	cfg *config

	anthropic adapter.Adapter
	openai    adapter.Adapter
	gemini    adapter.Adapter
	chats     map[string]adapter.Adapter
}

// New constructs a Client and pre-creates each enabled adapter so that
// subsequent requests share connections/SDK state.
func New(opts ...Option) (*Client, error) {
	cfg := defaultConfig()
	for _, o := range opts {
		o(cfg)
	}

	cli := &Client{cfg: cfg, chats: make(map[string]adapter.Adapter)}

	if cfg.anthropic != nil {
		a, err := anthropicadapter.New(anthropicadapter.Config{
			APIKey:       cfg.anthropic.APIKey,
			BaseURL:      cfg.anthropic.BaseURL,
			Version:      cfg.anthropic.Version,
			ExtraHeaders: cfg.anthropic.ExtraHeaders,
			HTTPClient:   cfg.httpClient,
		})
		if err != nil {
			return nil, fmt.Errorf("anthropify: anthropic adapter: %w", err)
		}
		cli.anthropic = a
	}

	if cfg.openai != nil {
		a, err := openairesponses.New(openairesponses.Config{
			APIKey:       cfg.openai.APIKey,
			BaseURL:      cfg.openai.BaseURL,
			Organization: cfg.openai.Organization,
			Project:      cfg.openai.Project,
			ExtraHeaders: cfg.openai.ExtraHeaders,
			HTTPClient:   cfg.httpClient,
		})
		if err != nil {
			return nil, fmt.Errorf("anthropify: openai_responses adapter: %w", err)
		}
		cli.openai = a
	}

	if cfg.gemini != nil {
		a, err := geminiadapter.New(geminiadapter.Config{
			Mode:         cfg.gemini.Mode,
			Project:      cfg.gemini.Project,
			Location:     cfg.gemini.Location,
			Publisher:    cfg.gemini.Publisher,
			APIKey:       cfg.gemini.APIKey,
			BaseURL:      cfg.gemini.BaseURL,
			ExtraHeaders: cfg.gemini.ExtraHeaders,
			HTTPClient:   cfg.httpClient,
		})
		if err != nil {
			return nil, fmt.Errorf("anthropify: gemini_native adapter: %w", err)
		}
		cli.gemini = a
	}

	for name, cc := range cfg.chatCompat {
		a, err := chatadapter.New(name, chatadapter.Config{
			BaseURL:      cc.BaseURL,
			APIKey:       cc.APIKey,
			ExtraHeaders: cc.ExtraHeaders,
			DefaultModel: cc.DefaultModel,
			HTTPClient:   cfg.httpClient,
		})
		if err != nil {
			return nil, fmt.Errorf("anthropify: chat_completions[%s] adapter: %w", name, err)
		}
		cli.chats[name] = a
	}

	return cli, nil
}

// CreateMessageStream dispatches req to the routed provider and returns
// a StreamReader emitting Anthropic-shaped events.
func (c *Client) CreateMessageStream(ctx context.Context, req anthropic.MessageNewParams) (*StreamReader, error) {
	ad, req2, err := c.dispatch(req)
	if err != nil {
		return nil, err
	}
	ctx = withLogger(ctx, c.cfg.logger)
	ch, err := ad.Stream(ctx, req2)
	if err != nil {
		return nil, err
	}
	return newStreamReader(ch), nil
}

// CreateMessage performs a non-streaming request. When the adapter
// exposes a native non-streaming endpoint it is used; otherwise the
// stream is drained and the final message assembled from content blocks.
func (c *Client) CreateMessage(ctx context.Context, req anthropic.MessageNewParams) (*anthropic.Message, error) {
	ad, req2, err := c.dispatch(req)
	if err != nil {
		return nil, err
	}
	ctx = withLogger(ctx, c.cfg.logger)
	if msg, err := ad.Invoke(ctx, req2); err == nil {
		return msg, nil
	} else if !errors.Is(err, ErrUnsupported) {
		return nil, err
	}
	// Fallback: drain the stream and assemble.
	ch, err := ad.Stream(ctx, req2)
	if err != nil {
		return nil, err
	}
	return assembleFromStream(ch)
}

// dispatch resolves the route and returns the adapter plus the
// (possibly rewritten) request.
func (c *Client) dispatch(req anthropic.MessageNewParams) (adapter.Adapter, anthropic.MessageNewParams, error) {
	route, err := c.cfg.resolve(string(req.Model))
	if err != nil {
		return nil, req, err
	}
	if route.UpstreamModel != "" {
		req.Model = anthropic.Model(route.UpstreamModel)
	}
	switch route.Provider {
	case ProviderAnthropic:
		if c.anthropic == nil {
			return nil, req, fmt.Errorf("%w: anthropic", ErrProviderNotConfigured)
		}
		return c.anthropic, req, nil
	case ProviderOpenAIResponses:
		if c.openai == nil {
			return nil, req, fmt.Errorf("%w: openai_responses", ErrProviderNotConfigured)
		}
		return c.openai, req, nil
	case ProviderGeminiNative:
		if c.gemini == nil {
			return nil, req, fmt.Errorf("%w: gemini_native", ErrProviderNotConfigured)
		}
		return c.gemini, req, nil
	case ProviderChatCompletions:
		a, ok := c.chats[route.Backend]
		if !ok {
			return nil, req, fmt.Errorf("%w: chat_completions[%s]", ErrProviderNotConfigured, route.Backend)
		}
		return a, req, nil
	}
	return nil, req, fmt.Errorf("%w: provider kind %q", ErrUnknownModel, route.Provider)
}

// assembleFromStream coalesces a stream of Anthropic events into a single
// Message. It intentionally ignores fields it does not understand.
func assembleFromStream(ch <-chan adapter.RawEvent) (*anthropic.Message, error) {
	var msg anthropic.Message
	// Per-block scratch state. blockStarts holds the original
	// content_block_start payload as a generic map so we can mutate it
	// (e.g. accumulate text/thinking/input_json) before materialising.
	var blockStarts []map[string]any
	var blockTexts []string         // accumulated text/thinking
	var blockToolInputJSON []string // accumulated tool_use partial_json

	ensureIndex := func(i int) {
		for len(blockStarts) <= i {
			blockStarts = append(blockStarts, map[string]any{"type": "text"})
			blockTexts = append(blockTexts, "")
			blockToolInputJSON = append(blockToolInputJSON, "")
		}
	}

	for evt := range ch {
		if evt.Err != nil {
			return nil, evt.Err
		}
		var peek struct {
			Type    string          `json:"type"`
			Message json.RawMessage `json:"message"`
			Index   int             `json:"index"`
			Delta   json.RawMessage `json:"delta"`
			Content json.RawMessage `json:"content_block"`
			Usage   json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(evt.Data, &peek); err != nil {
			continue
		}
		switch peek.Type {
		case "message_start":
			if len(peek.Message) > 0 {
				_ = json.Unmarshal(peek.Message, &msg)
			}
		case "content_block_start":
			var cb map[string]any
			if err := json.Unmarshal(peek.Content, &cb); err != nil || cb == nil {
				cb = map[string]any{"type": "text"}
			}
			ensureIndex(peek.Index)
			blockStarts[peek.Index] = cb
			if t, _ := cb["type"].(string); t == "text" {
				if s, ok := cb["text"].(string); ok {
					blockTexts[peek.Index] = s
				}
			} else if t == "thinking" {
				if s, ok := cb["thinking"].(string); ok {
					blockTexts[peek.Index] = s
				}
			}
		case "content_block_delta":
			var d struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Thinking    string `json:"thinking"`
				PartialJSON string `json:"partial_json"`
			}
			_ = json.Unmarshal(peek.Delta, &d)
			ensureIndex(peek.Index)
			switch d.Type {
			case "text_delta":
				blockTexts[peek.Index] += d.Text
			case "thinking_delta":
				blockTexts[peek.Index] += d.Thinking
			case "input_json_delta":
				blockToolInputJSON[peek.Index] += d.PartialJSON
			}
		case "message_delta":
			var d struct {
				StopReason   string `json:"stop_reason"`
				StopSequence string `json:"stop_sequence"`
			}
			_ = json.Unmarshal(peek.Delta, &d)
			if d.StopReason != "" {
				msg.StopReason = anthropic.StopReason(d.StopReason)
			}
			if d.StopSequence != "" {
				msg.StopSequence = d.StopSequence
			}
			// Usage from message_delta carries the final output_tokens
			// (and may restate input_tokens). Merge non-zero fields.
			if len(peek.Usage) > 0 {
				var u struct {
					InputTokens              int64 `json:"input_tokens"`
					OutputTokens             int64 `json:"output_tokens"`
					CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
					CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
				}
				if err := json.Unmarshal(peek.Usage, &u); err == nil {
					if u.InputTokens > 0 {
						msg.Usage.InputTokens = u.InputTokens
					}
					if u.OutputTokens > 0 {
						msg.Usage.OutputTokens = u.OutputTokens
					}
					if u.CacheCreationInputTokens > 0 {
						msg.Usage.CacheCreationInputTokens = u.CacheCreationInputTokens
					}
					if u.CacheReadInputTokens > 0 {
						msg.Usage.CacheReadInputTokens = u.CacheReadInputTokens
					}
				}
			}
		}
	}
	// Materialise content blocks. We round-trip through JSON because the
	// SDK's ContentBlockUnion is not externally constructable.
	if len(blockStarts) > 0 {
		pieces := make([]map[string]any, 0, len(blockStarts))
		for i, cb := range blockStarts {
			t, _ := cb["type"].(string)
			switch t {
			case "", "text":
				pieces = append(pieces, map[string]any{"type": "text", "text": blockTexts[i]})
			case "thinking":
				piece := map[string]any{"type": "thinking", "thinking": blockTexts[i]}
				if sig, ok := cb["signature"].(string); ok && sig != "" {
					piece["signature"] = sig
				}
				pieces = append(pieces, piece)
			case "tool_use":
				piece := map[string]any{"type": "tool_use"}
				if id, ok := cb["id"].(string); ok {
					piece["id"] = id
				}
				if name, ok := cb["name"].(string); ok {
					piece["name"] = name
				}
				// Prefer the accumulated partial_json if any frames arrived;
				// otherwise honour any input that came on content_block_start.
				if blockToolInputJSON[i] != "" {
					var inp any
					if err := json.Unmarshal([]byte(blockToolInputJSON[i]), &inp); err == nil {
						piece["input"] = inp
					} else {
						piece["input"] = map[string]any{}
					}
				} else if inp, ok := cb["input"]; ok {
					piece["input"] = inp
				} else {
					piece["input"] = map[string]any{}
				}
				// Preserve provider-specific thought_signature (e.g. Gemini
				// 3.x) so callers can replay it on the next turn via
				// SetExtraFields. The SDK stores unknown fields in
				// ContentBlockUnion.JSON.raw; callers retrieve them with
				// the .RawJSON() helper.
				if sig, ok := cb["thought_signature"].(string); ok && sig != "" {
					piece["thought_signature"] = sig
				}
				pieces = append(pieces, piece)
			default:
				// Pass through unknown block types verbatim.
				pieces = append(pieces, cb)
			}
		}
		data, _ := json.Marshal(pieces)
		_ = json.Unmarshal(data, &msg.Content)
	}
	if msg.Role == "" {
		msg.Role = "assistant"
	}
	return &msg, nil
}
