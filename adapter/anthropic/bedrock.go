// Bedrock branch of the Anthropic adapter. This file is the only place
// in anthropify that imports aws-sdk-go-v2 and the SDK's bedrock
// subpackage. It is compiled unconditionally; the runtime decision is
// made by Adapter.cfg.Mode == AnthropicModeBedrock.
//
// What this branch delegates to anthropic-sdk-go's bedrock subpackage
// (verified against v1.36.0/bedrock/bedrock.go):
//
//   - SigV4 request signing (bedrock.go:206, 285).
//   - URL rewrite /v1/messages -> /model/{id-or-arn}/invoke (or
//     -with-response-stream); strips "model" / "stream" from body
//     (bedrock.go:243-259).
//   - Body injection of "anthropic_version": "bedrock-2023-05-31" when
//     missing (bedrock.go:230-232).
//   - "anthropic-beta" header to body field translation
//     (bedrock.go:235-241). Anthropify's ExtraHeaders["anthropic-beta"]
//     therefore round-trips through this middleware automatically; we
//     do NOT port the llm-proxy betaMiddleware.
//   - Decoding of application/vnd.amazon.eventstream binary frames
//     into ssestream.Event (bedrock.go:47-160, 171-175). The bytes we
//     push onto the RawEvent channel are byte-identical to canonical
//     Anthropic SSE event JSON.
//
// Architectural reference: codeck-backend/llm-proxy
// clients/anthropic/aws/bedrock_wrapper.go (74 lines, same shape).
// Anthropify's branch is shorter because the SDK release we pin
// (v1.36.0) has the betaMiddleware functionality built in.
package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"

	"github.com/pejash666/anthropify/adapter"
)

// newBedrock validates Bedrock-mode credentials and constructs an
// SDK client wired through bedrock.WithConfig. Caller-supplied
// ExtraHeaders are forwarded via option.WithHeaderAdd so that the
// SDK's beta-header-to-body middleware sees them.
func newBedrock(name string, cfg Config) (*Adapter, error) {
	if cfg.AWSAccessKeyID == "" {
		return nil, errors.New("anthropify/anthropic: AWSAccessKeyID is required for Bedrock mode")
	}
	if cfg.AWSSecretAccessKey == "" {
		return nil, errors.New("anthropify/anthropic: AWSSecretAccessKey is required for Bedrock mode")
	}
	if cfg.AWSRegion == "" {
		return nil, errors.New("anthropify/anthropic: AWSRegion is required for Bedrock mode")
	}

	awsCfg := aws.Config{
		Region: cfg.AWSRegion,
		Credentials: credentials.NewStaticCredentialsProvider(
			cfg.AWSAccessKeyID,
			cfg.AWSSecretAccessKey,
			cfg.AWSSessionToken,
		),
	}

	// Order matters: bedrock.WithConfig sets the default base URL to
	// https://bedrock-runtime.<region>.amazonaws.com and registers the
	// SigV4 + URL-rewrite + event-stream middleware. Any explicit
	// BaseURL/HTTPClient passed afterwards overrides those defaults
	// (intended for httptest.NewServer in unit tests).
	opts := []option.RequestOption{bedrock.WithConfig(awsCfg)}
	if cfg.BaseURL != "" {
		opts = append(opts, option.WithBaseURL(cfg.BaseURL))
	}
	if cfg.HTTPClient != nil {
		opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
	}
	for k, vs := range cfg.ExtraHeaders {
		for _, v := range vs {
			opts = append(opts, option.WithHeaderAdd(k, v))
		}
	}

	client := anthropicsdk.NewClient(opts...)
	return &Adapter{name: name, cfg: cfg, bedrockClient: &client}, nil
}

// invokeBedrock dispatches a non-streaming request through the SDK
// client. The SDK signs, rewrites the URL, and decodes the response
// into *anthropicsdk.Message directly.
func (a *Adapter) invokeBedrock(ctx context.Context, req anthropicsdk.MessageNewParams) (*anthropicsdk.Message, error) {
	msg, err := a.bedrockClient.Messages.New(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("anthropify/anthropic[bedrock]: %w", err)
	}
	return msg, nil
}

// streamBedrock dispatches a streaming request through the SDK
// client. The SDK decodes AWS event-stream binary frames into
// MessageStreamEventUnion values; we re-emit each event's RawJSON()
// onto the RawEvent channel so downstream code (StreamReader,
// assembleFromStream) sees byte-identical canonical Anthropic SSE
// event JSON.
func (a *Adapter) streamBedrock(ctx context.Context, req anthropicsdk.MessageNewParams) (<-chan adapter.RawEvent, error) {
	stream := a.bedrockClient.Messages.NewStreaming(ctx, req)

	ch := make(chan adapter.RawEvent, 16)
	go func() {
		defer close(ch)
		for stream.Next() {
			if ctx.Err() != nil {
				ch <- adapter.RawEvent{Err: ctx.Err()}
				return
			}
			evt := stream.Current()
			raw := evt.RawJSON()
			if raw == "" {
				continue
			}
			ch <- adapter.RawEvent{Data: []byte(raw)}
		}
		if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
			ch <- adapter.RawEvent{Err: fmt.Errorf("anthropify/anthropic[bedrock]: %w", err)}
		}
	}()
	return ch, nil
}
