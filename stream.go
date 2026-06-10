package anthropify

import (
	"encoding/json"

	"github.com/pejash666/anthropify/adapter"
)

// StreamReader is a pull-style iterator over Anthropic-shaped streaming
// events. Zero-value readers are unusable; construct one via
// Client.CreateMessageStream.
type StreamReader struct {
	ch      <-chan adapter.RawEvent
	current MessageStreamEventUnion
	raw     []byte
	err     error
	closed  bool
}

func newStreamReader(ch <-chan adapter.RawEvent) *StreamReader {
	return &StreamReader{ch: ch}
}

// Next pulls the next event from the underlying channel. It returns false
// when the stream has ended; the caller should consult Err to check for
// an error termination.
func (s *StreamReader) Next() bool {
	if s.closed {
		return false
	}
	evt, ok := <-s.ch
	if !ok {
		s.closed = true
		return false
	}
	if evt.Err != nil {
		s.err = evt.Err
		s.closed = true
		return false
	}
	var union MessageStreamEventUnion
	if err := json.Unmarshal(evt.Data, &union); err != nil {
		s.err = err
		s.closed = true
		return false
	}
	s.current = union
	s.raw = evt.Data
	return true
}

// Current returns the last event produced by Next.
func (s *StreamReader) Current() MessageStreamEventUnion { return s.current }

// CurrentRaw returns the raw JSON bytes for the last event. Useful when
// the caller wants to forward the payload verbatim (e.g. over their own
// HTTP SSE connection).
func (s *StreamReader) CurrentRaw() []byte { return s.raw }

// Err returns the first error encountered, if any.
func (s *StreamReader) Err() error { return s.err }

// Close drains the underlying channel. It is safe to call multiple times.
func (s *StreamReader) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	for range s.ch {
		// drain
	}
	return nil
}
