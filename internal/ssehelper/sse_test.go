package ssehelper

import (
	"strings"
	"testing"
)

func TestDecoder_Basic(t *testing.T) {
	raw := "event: message_start\ndata: {\"a\":1}\n\nevent: ping\ndata: {}\n\n"
	d := NewDecoder(strings.NewReader(raw))
	var got []Event
	for d.Next() {
		got = append(got, d.Event())
	}
	if len(got) != 2 {
		t.Fatalf("len=%d, want 2 (%v)", len(got), got)
	}
	if got[0].Event != "message_start" || got[0].Data != `{"a":1}` {
		t.Fatalf("frame 0 mismatch: %+v", got[0])
	}
	if got[1].Event != "ping" || got[1].Data != `{}` {
		t.Fatalf("frame 1 mismatch: %+v", got[1])
	}
}

func TestDecoder_MultiLineDataAndComments(t *testing.T) {
	raw := ":comment line\nevent: m\ndata: line1\ndata: line2\n\n"
	d := NewDecoder(strings.NewReader(raw))
	if !d.Next() {
		t.Fatalf("expected one frame")
	}
	evt := d.Event()
	if evt.Event != "m" {
		t.Fatalf("event name = %q", evt.Event)
	}
	if evt.Data != "line1\nline2" {
		t.Fatalf("data = %q", evt.Data)
	}
}

func TestDecoder_TrailingFrameNoBlankLine(t *testing.T) {
	raw := "event: done\ndata: bye"
	d := NewDecoder(strings.NewReader(raw))
	if !d.Next() {
		t.Fatalf("expected trailing frame")
	}
	if d.Event().Data != "bye" {
		t.Fatalf("data = %q", d.Event().Data)
	}
	if d.Next() {
		t.Fatalf("unexpected extra frame")
	}
}
