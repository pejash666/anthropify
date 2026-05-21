package openai_responses

import (
	"crypto/rand"
	"encoding/hex"
)

// genMessageID returns a short opaque message id used in the emitted
// message_start event. Errors from crypto/rand fall back to a constant;
// collisions are harmless here because the id is only echoed back to the
// caller.
func genMessageID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "msg_00000000"
	}
	return "msg_" + hex.EncodeToString(b[:])
}
