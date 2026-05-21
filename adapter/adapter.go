// Package adapter defines the interface every provider backend must
// implement. Adapters translate an Anthropic Messages request into the
// provider's native protocol and translate the provider's response back
// into Anthropic-shaped SSE events.
package adapter

import (
	"context"

	"github.com/anthropics/anthropic-sdk-go"
)

// RawEvent is a single Anthropic-shaped SSE payload. Producers emit bytes
// to avoid forcing every adapter to construct the SDK's union types.
type RawEvent struct {
	// Data is the JSON payload (without "event:" / "data:" SSE prefix).
	Data []byte
	// Err, if non-nil, terminates the stream. The reader MUST NOT be
	// expected to emit further events after an Err.
	Err error
}

// Adapter is implemented by every provider backend.
//
// Implementations are expected to be safe for concurrent use across
// distinct calls; per-call state must be kept on the returned channel or
// inside a short-lived helper object.
type Adapter interface {
	// Name returns a human-readable identifier used in log lines and
	// error messages.
	Name() string
	// Stream executes the request against the upstream provider and emits
	// Anthropic-shaped events on the returned channel. The channel is
	// closed when the stream ends. Emitters MUST NOT panic on ctx
	// cancellation; they should emit a final RawEvent{Err: ctx.Err()}
	// and close.
	Stream(ctx context.Context, req anthropic.MessageNewParams) (<-chan RawEvent, error)
	// Invoke performs a non-streaming request. The default implementation
	// in the top-level Client simply runs Stream to completion and
	// assembles the final Message; adapters may override this when their
	// upstream exposes a cheaper non-streaming endpoint.
	Invoke(ctx context.Context, req anthropic.MessageNewParams) (*anthropic.Message, error)
}
