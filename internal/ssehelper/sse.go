// Package ssehelper provides a minimal Server-Sent Events line decoder
// suitable for parsing Anthropic, OpenAI, and Gemini streams. Only the
// "data:" and "event:" fields are honoured; comments and "id:" lines are
// ignored, per the SSE spec.
package ssehelper

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// Event is a single decoded SSE frame.
type Event struct {
	// Event is the optional event name (e.g. "message_start").
	Event string
	// Data is the concatenated data lines, joined by "\n", with no
	// trailing newline.
	Data string
}

// Decoder reads SSE frames from an io.Reader.
type Decoder struct {
	scanner *bufio.Scanner
	cur     Event
	err     error
	closed  bool
}

// NewDecoder wraps r in a Decoder. The caller is responsible for closing r.
func NewDecoder(r io.Reader) *Decoder {
	sc := bufio.NewScanner(r)
	// SSE frames can be large (e.g. JSON-encoded tool arguments); grow
	// the buffer to 1 MiB with a 16 MiB cap to match production use.
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	return &Decoder{scanner: sc}
}

// Next advances to the next non-empty event. It returns false on EOF or
// error; use Err to distinguish.
func (d *Decoder) Next() bool {
	if d.closed {
		return false
	}
	var buf bytes.Buffer
	var evt string
	for d.scanner.Scan() {
		line := d.scanner.Text()
		if line == "" {
			if buf.Len() == 0 && evt == "" {
				// Stray blank line between events; keep scanning.
				continue
			}
			// Terminate frame.
			d.cur = Event{Event: evt, Data: strings.TrimRight(buf.String(), "\n")}
			return true
		}
		if strings.HasPrefix(line, ":") {
			continue // comment
		}
		name, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		// SSE spec: a single space after the colon is stripped.
		value = strings.TrimPrefix(value, " ")
		switch name {
		case "event":
			evt = value
		case "data":
			if buf.Len() > 0 {
				buf.WriteByte('\n')
			}
			buf.WriteString(value)
		}
	}
	if err := d.scanner.Err(); err != nil {
		d.err = err
	}
	// Flush any trailing frame without terminating blank line.
	if buf.Len() > 0 || evt != "" {
		d.cur = Event{Event: evt, Data: strings.TrimRight(buf.String(), "\n")}
		d.closed = true
		return true
	}
	d.closed = true
	return false
}

// Event returns the most recent frame produced by Next.
func (d *Decoder) Event() Event { return d.cur }

// Err returns the first non-EOF error encountered while scanning.
func (d *Decoder) Err() error { return d.err }
