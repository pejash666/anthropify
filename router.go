package hybridstream

import (
	"fmt"
	"sort"
	"strings"
)

// resolve returns the Route selected for a given model name. The lookup
// order is:
//  1. Explicit WithModelOverride callback.
//  2. Longest matching prefix from WithModelRoute / built-in defaults.
//
// When no rule matches, (Route{}, ErrUnknownModel) is returned.
func (c *config) resolve(model string) (Route, error) {
	if c.overrideFn != nil {
		if r, ok := c.overrideFn(model); ok {
			return r, nil
		}
	}
	lower := strings.ToLower(model)

	// Sort prefixes longest-first so more specific rules win.
	prefixes := make([]string, 0, len(c.modelRoutes))
	for p := range c.modelRoutes {
		prefixes = append(prefixes, p)
	}
	sort.Slice(prefixes, func(i, j int) bool { return len(prefixes[i]) > len(prefixes[j]) })

	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return c.modelRoutes[p], nil
		}
	}
	return Route{}, fmt.Errorf("%w: %q", ErrUnknownModel, model)
}
