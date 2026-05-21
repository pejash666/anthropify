package hybridstream

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/shahao/hybridstream/adapter"
	anthropicadapter "github.com/shahao/hybridstream/adapter/anthropic"
	chatadapter "github.com/shahao/hybridstream/adapter/chat_completions"
	geminiadapter "github.com/shahao/hybridstream/adapter/gemini_native"
	openairesponses "github.com/shahao/hybridstream/adapter/openai_responses"
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
			return nil, fmt.Errorf("hybridstream: anthropic adapter: %w", err)
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
			return nil, fmt.Errorf("hybridstream: openai_responses adapter: %w", err)
		}
		cli.openai = a
	}

	if cfg.gemini != nil {
		a, err := geminiadapter.New(geminiadapter.Config{
			Project:      cfg.gemini.Project,
			Location:     cfg.gemini.Location,
			Publisher:    cfg.gemini.Publisher,
			APIKey:       cfg.gemini.APIKey,
			BaseURL:      cfg.gemini.BaseURL,
			ExtraHeaders: cfg.gemini.ExtraHeaders,
			HTTPClient:   cfg.httpClient,
		})
		if err != nil {
			return nil, fmt.Errorf("hybridstream: gemini_native adapter: %w", err)
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
			return nil, fmt.Errorf("hybridstream: chat_completions[%s] adapter: %w", name, err)
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
	} else if err != ErrUnsupported {
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
	var blockTexts []string
	var blockTypes []string
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
			var cb struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			_ = json.Unmarshal(peek.Content, &cb)
			for len(blockTexts) <= peek.Index {
				blockTexts = append(blockTexts, "")
				blockTypes = append(blockTypes, "")
			}
			blockTypes[peek.Index] = cb.Type
			blockTexts[peek.Index] = cb.Text
		case "content_block_delta":
			var d struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				Thinking string `json:"thinking"`
			}
			_ = json.Unmarshal(peek.Delta, &d)
			if peek.Index >= len(blockTexts) {
				continue
			}
			switch d.Type {
			case "text_delta":
				blockTexts[peek.Index] += d.Text
			case "thinking_delta":
				blockTexts[peek.Index] += d.Thinking
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
		}
	}
	// Materialise content blocks. We round-trip through JSON because the
	// SDK's ContentBlockUnion is not externally constructable.
	if len(blockTypes) > 0 {
		pieces := make([]map[string]any, 0, len(blockTypes))
		for i, t := range blockTypes {
			switch t {
			case "text", "":
				pieces = append(pieces, map[string]any{"type": "text", "text": blockTexts[i]})
			case "thinking":
				pieces = append(pieces, map[string]any{"type": "thinking", "thinking": blockTexts[i]})
			default:
				pieces = append(pieces, map[string]any{"type": t})
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
