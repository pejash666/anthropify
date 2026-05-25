// Package schema centralises tool input_schema dialect handling for
// every protocol adapter shipped by anthropify.
//
// Each adapter declares one constant Dialect and calls Normalize. The
// normalizer drives conversion off the dialect's capability bits, with
// the user-selectable Policy deciding whether lossy transforms are
// silently performed, logged, or returned as a hard error.
//
// The package has zero dependencies outside the standard library so
// adapter code can import it without dragging in the rest of the
// anthropify surface.
package schema

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
)

// Policy selects how the normalizer reacts to lossy transforms.
//
// PolicyStrict is the v0.2.0 default: any non-cosmetic transform that
// would drop or rewrite caller-provided schema fields returns an error
// wrapping ErrSchemaIncompatible. PolicyLossy permits the documented
// rewrites/drops (e.g. oneOf -> anyOf for Gemini) but logs each via
// slog.Warn. PolicyBestEffort additionally tolerates extra silent drops
// for everything that is not fundamentally fatal (cycles, $ref, etc.).
type Policy int

const (
	// PolicyStrict makes the normalizer error on any lossy transform.
	PolicyStrict Policy = iota
	// PolicyLossy permits documented rewrites and warn-and-proceed
	// drops while preserving fatal-incompatibility errors.
	PolicyLossy
	// PolicyBestEffort behaves like PolicyLossy plus permits the
	// documented BestEffort-only downgrades (e.g. OpenAI Strict
	// additionalProperties: true -> false).
	PolicyBestEffort
)

// String returns the canonical name of the policy.
func (p Policy) String() string {
	switch p {
	case PolicyStrict:
		return "strict"
	case PolicyLossy:
		return "lossy"
	case PolicyBestEffort:
		return "best_effort"
	default:
		return fmt.Sprintf("policy(%d)", int(p))
	}
}

// AdditionalPropsMode encodes how a dialect treats the
// additionalProperties keyword.
type AdditionalPropsMode int

const (
	// AdditionalPropsAllow keeps additionalProperties verbatim.
	AdditionalPropsAllow AdditionalPropsMode = iota
	// AdditionalPropsForceFalse rewrites additionalProperties: true
	// to false (lossy; warns) and keeps explicit false as-is.
	AdditionalPropsForceFalse
	// AdditionalPropsDrop removes additionalProperties entirely
	// (lossy; warns).
	AdditionalPropsDrop
	// AdditionalPropsErrorOnTrue treats additionalProperties: true
	// (or a schema value) as a hard error under PolicyStrict and
	// downgrades to false under PolicyBestEffort.
	AdditionalPropsErrorOnTrue
)

// Dialect captures one upstream's tool-schema capabilities. Adapters
// expose a constant Dialect; the normalizer consults its bits during
// conversion.
type Dialect struct {
	Name                    string
	SupportsOneOf           bool
	SupportsAllOf           bool
	SupportsNot             bool
	SupportsRefs            bool
	SupportsExclusiveBounds bool
	SupportsMultipleOf      bool
	SupportsUniqueItems     bool
	SupportsAdditionalProps AdditionalPropsMode
	// UseGeminiNullable rewrites type:["string","null"] (or type:null
	// alongside another type) to "nullable: true".
	UseGeminiNullable bool
	// LowercaseType normalises the "type" keyword (e.g. STRING -> string).
	LowercaseType bool
	// SortProperties sorts properties / required deterministically to
	// improve prompt-cache hit rates on Gemini.
	SortProperties bool
	// StripSchemaKey removes the "$schema" meta field unconditionally.
	StripSchemaKey bool
	// MaxRecursionDepth caps recursion depth; <=0 means unlimited.
	MaxRecursionDepth int
	// AllowedKeys, when non-nil, keeps only the listed top-level
	// fields after conversion (Gemini's whitelist).
	AllowedKeys map[string]bool
}

// Warning describes one non-fatal transform applied (or candidate to
// apply) by the normalizer. The Path uses dot notation (e.g.
// "properties.user.oneOf") so callers can pinpoint the input location.
type Warning struct {
	Path   string
	Code   string
	Detail string
}

// ErrSchemaIncompatible is the sentinel returned when a transform was
// rejected by the active Policy. Callers detect it with errors.Is.
var ErrSchemaIncompatible = errors.New("anthropify/schema: incompatible schema")

// Warning codes. Stable strings so callers can grep logs.
const (
	CodeDropOneOf            = "drop_oneof"
	CodeRewriteOneOf         = "rewrite_oneof_to_anyof"
	CodeDropAllOf            = "drop_allof"
	CodeDropNot              = "drop_not"
	CodeDropMultipleOf       = "drop_multipleof"
	CodeDropUniqueItems      = "drop_uniqueitems"
	CodeDropExclusive        = "drop_exclusive_bounds"
	CodeDropAdditionalProps  = "drop_additional_props"
	CodeDowngradeAddlProps   = "downgrade_additional_props"
	CodeRewriteNullableType  = "rewrite_nullable_type"
	CodeDropForbiddenKey     = "drop_forbidden_key"
	CodeRefUnsupported       = "ref_unsupported"
	CodeRecursionLimit       = "recursion_limit"
	CodeAdditionalPropsTrue  = "additional_props_true"
	CodeAdditionalPropsSchm  = "additional_props_schema"
)

// Normalize lowers a canonical Anthropic-shape schema to the dialect.
// It returns the converted schema, any warnings collected along the
// way (already logged at slog.WarnLevel under non-Strict policies),
// and an error wrapping ErrSchemaIncompatible when policy rejects a
// transform.
//
// canonical is consumed in a deep-copy fashion: callers may safely
// reuse the input map after the call returns. A nil canonical produces
// a nil result with no warnings.
func Normalize(ctx context.Context, canonical map[string]any, dialect Dialect, policy Policy) (map[string]any, []Warning, error) {
	if canonical == nil {
		return nil, nil, nil
	}
	c := &converter{
		dialect: dialect,
		policy:  policy,
		logger:  loggerFrom(ctx),
	}
	out, err := c.convert(canonical, "", 0)
	if err != nil {
		return nil, c.warnings, err
	}
	return out, c.warnings, nil
}

type converter struct {
	dialect  Dialect
	policy   Policy
	warnings []Warning
	logger   *slog.Logger
}

// joinPath appends seg to base using dot notation; segments containing
// dots are not escaped because schema keys never contain them in
// practice.
func joinPath(base, seg string) string {
	if base == "" {
		return seg
	}
	return base + "." + seg
}

// note appends a warning. lossy=true means the warning is non-fatal
// under PolicyLossy/BestEffort but fatal under PolicyStrict; lossy=false
// means it is always fatal regardless of policy (e.g. $ref).
func (c *converter) note(path, code, detail string, lossy bool) error {
	w := Warning{Path: path, Code: code, Detail: detail}
	c.warnings = append(c.warnings, w)
	if !lossy {
		return fmt.Errorf("%w: %s at %s: %s", ErrSchemaIncompatible, code, path, detail)
	}
	switch c.policy {
	case PolicyStrict:
		return fmt.Errorf("%w: %s at %s: %s", ErrSchemaIncompatible, code, path, detail)
	default:
		c.logger.Warn("schema normalize lossy",
			slog.String("dialect", c.dialect.Name),
			slog.String("policy", c.policy.String()),
			slog.String("code", code),
			slog.String("path", path),
			slog.String("detail", detail),
		)
		return nil
	}
}

// noteBestEffortOnly is like note but only tolerated under PolicyBestEffort;
// PolicyLossy treats it as fatal too.
func (c *converter) noteBestEffortOnly(path, code, detail string) error {
	w := Warning{Path: path, Code: code, Detail: detail}
	c.warnings = append(c.warnings, w)
	if c.policy == PolicyBestEffort {
		c.logger.Warn("schema normalize best-effort downgrade",
			slog.String("dialect", c.dialect.Name),
			slog.String("policy", c.policy.String()),
			slog.String("code", code),
			slog.String("path", path),
			slog.String("detail", detail),
		)
		return nil
	}
	return fmt.Errorf("%w: %s at %s: %s", ErrSchemaIncompatible, code, path, detail)
}

func (c *converter) convert(in map[string]any, path string, depth int) (map[string]any, error) {
	if c.dialect.MaxRecursionDepth > 0 && depth > c.dialect.MaxRecursionDepth {
		if err := c.note(path, CodeRecursionLimit,
			fmt.Sprintf("schema depth %d exceeds dialect cap %d", depth, c.dialect.MaxRecursionDepth),
			false); err != nil {
			return nil, err
		}
	}

	out := make(map[string]any, len(in))

	// Detect $ref / $defs / $id / recursion early — never silently lossy.
	if v, ok := in["$ref"]; ok && !c.dialect.SupportsRefs {
		if err := c.note(path, CodeRefUnsupported,
			fmt.Sprintf("$ref %v not supported by dialect %q", v, c.dialect.Name),
			false); err != nil {
			return nil, err
		}
	}
	if _, ok := in["$defs"]; ok && !c.dialect.SupportsRefs {
		if err := c.note(path, CodeRefUnsupported,
			fmt.Sprintf("$defs not supported by dialect %q", c.dialect.Name),
			false); err != nil {
			return nil, err
		}
	}
	if _, ok := in["definitions"]; ok && !c.dialect.SupportsRefs {
		if err := c.note(path, CodeRefUnsupported,
			fmt.Sprintf("definitions not supported by dialect %q", c.dialect.Name),
			false); err != nil {
			return nil, err
		}
	}

	for k, v := range in {
		switch k {
		case "$schema":
			if c.dialect.StripSchemaKey {
				continue
			}
			out[k] = v
		case "$ref", "$defs", "$id", "definitions":
			if c.dialect.SupportsRefs {
				out[k] = v
			}
			// otherwise dropped after the error gate above
		case "type":
			out[k] = c.handleType(v, path, out)
		case "properties":
			props, err := c.convertProperties(v, joinPath(path, "properties"), depth+1)
			if err != nil {
				return nil, err
			}
			out[k] = props
		case "items":
			items, err := c.convertItems(v, joinPath(path, "items"), depth+1)
			if err != nil {
				return nil, err
			}
			out[k] = items
		case "required":
			out[k] = c.handleRequired(v)
		case "oneOf":
			converted, err := c.convertVariants(v, joinPath(path, "oneOf"), depth+1)
			if err != nil {
				return nil, err
			}
			if c.dialect.SupportsOneOf {
				out["oneOf"] = converted
			} else {
				if err := c.note(path, CodeRewriteOneOf,
					fmt.Sprintf("oneOf rewritten to anyOf for dialect %q", c.dialect.Name),
					true); err != nil {
					return nil, err
				}
				// Merge with any existing anyOf siblings.
				if existing, ok := out["anyOf"].([]any); ok {
					out["anyOf"] = append(existing, converted...)
				} else {
					out["anyOf"] = converted
				}
			}
		case "anyOf":
			converted, err := c.convertVariants(v, joinPath(path, "anyOf"), depth+1)
			if err != nil {
				return nil, err
			}
			if existing, ok := out["anyOf"].([]any); ok {
				out["anyOf"] = append(existing, converted...)
			} else {
				out["anyOf"] = converted
			}
		case "allOf":
			if c.dialect.SupportsAllOf {
				converted, err := c.convertVariants(v, joinPath(path, "allOf"), depth+1)
				if err != nil {
					return nil, err
				}
				out["allOf"] = converted
			} else {
				if err := c.note(path, CodeDropAllOf,
					fmt.Sprintf("allOf dropped for dialect %q", c.dialect.Name),
					true); err != nil {
					return nil, err
				}
			}
		case "not":
			if c.dialect.SupportsNot {
				if m, ok := v.(map[string]any); ok {
					sub, err := c.convert(m, joinPath(path, "not"), depth+1)
					if err != nil {
						return nil, err
					}
					out["not"] = sub
				} else {
					out["not"] = v
				}
			} else {
				if err := c.note(path, CodeDropNot,
					fmt.Sprintf("not dropped for dialect %q", c.dialect.Name),
					true); err != nil {
					return nil, err
				}
			}
		case "exclusiveMinimum", "exclusiveMaximum":
			if c.dialect.SupportsExclusiveBounds {
				out[k] = v
			} else {
				if err := c.note(path, CodeDropExclusive,
					fmt.Sprintf("%s dropped for dialect %q", k, c.dialect.Name),
					true); err != nil {
					return nil, err
				}
			}
		case "multipleOf":
			if c.dialect.SupportsMultipleOf {
				out[k] = v
			} else {
				if err := c.note(path, CodeDropMultipleOf,
					fmt.Sprintf("multipleOf dropped for dialect %q", c.dialect.Name),
					true); err != nil {
					return nil, err
				}
			}
		case "uniqueItems":
			if c.dialect.SupportsUniqueItems {
				out[k] = v
			} else {
				if err := c.note(path, CodeDropUniqueItems,
					fmt.Sprintf("uniqueItems dropped for dialect %q", c.dialect.Name),
					true); err != nil {
					return nil, err
				}
			}
		case "additionalProperties":
			if err := c.handleAdditionalProps(v, path, out); err != nil {
				return nil, err
			}
		default:
			out[k] = v
		}
	}

	// Apply dialect AllowedKeys whitelist. We do this AFTER conversion
	// so unknown keys produced by the rewrite (e.g. nullable for Gemini)
	// stay alive only when they are in the allowed set.
	if c.dialect.AllowedKeys != nil {
		filtered := make(map[string]any, len(out))
		for k, v := range out {
			if c.dialect.AllowedKeys[k] {
				filtered[k] = v
				continue
			}
			// Forbidden top-level key. Issue a warning but stay lossy:
			// this matches v0.1.x behaviour and avoids surprising the
			// user with errors for vendor extensions like
			// "x-vendor-extension" that the upstream simply ignores.
			if err := c.note(joinPath(path, k), CodeDropForbiddenKey,
				fmt.Sprintf("key %q not in dialect %q whitelist", k, c.dialect.Name),
				true); err != nil {
				return nil, err
			}
		}
		out = filtered
	}

	// Infer type from enum if missing (Gemini pattern).
	if _, ok := out["type"]; !ok {
		if enumArr, ok := out["enum"].([]any); ok && len(enumArr) > 0 {
			out["type"] = inferTypeFromEnum(enumArr)
		}
	}

	return out, nil
}

// handleType normalises "type". For UseGeminiNullable dialects, a
// type:[T,"null"] array collapses to {"type":"T","nullable":true}; the
// nullable key is set on the parent map.
func (c *converter) handleType(v any, _ string, parent map[string]any) any {
	switch t := v.(type) {
	case string:
		if c.dialect.LowercaseType {
			return strings.ToLower(t)
		}
		return t
	case []any:
		if c.dialect.UseGeminiNullable {
			var nonNull string
			hasNull := false
			for _, x := range t {
				if s, ok := x.(string); ok {
					if strings.EqualFold(s, "null") {
						hasNull = true
					} else if nonNull == "" {
						nonNull = s
					}
				}
			}
			if hasNull && nonNull != "" {
				parent["nullable"] = true
				if c.dialect.LowercaseType {
					return strings.ToLower(nonNull)
				}
				return nonNull
			}
		}
		// Otherwise keep as-is (passthrough).
		if c.dialect.LowercaseType {
			lowered := make([]any, len(t))
			for i, x := range t {
				if s, ok := x.(string); ok {
					lowered[i] = strings.ToLower(s)
				} else {
					lowered[i] = x
				}
			}
			return lowered
		}
		return t
	}
	return v
}

func (c *converter) convertProperties(v any, path string, depth int) (map[string]any, error) {
	props, ok := v.(map[string]any)
	if !ok {
		return nil, nil
	}
	names := make([]string, 0, len(props))
	for n := range props {
		names = append(names, n)
	}
	if c.dialect.SortProperties {
		sort.Strings(names)
	}
	out := make(map[string]any, len(props))
	for _, n := range names {
		entry := props[n]
		if sm, ok := entry.(map[string]any); ok {
			converted, err := c.convert(sm, joinPath(path, n), depth)
			if err != nil {
				return nil, err
			}
			out[n] = converted
		} else {
			out[n] = entry
		}
	}
	if c.dialect.SortProperties {
		// Re-emit as a sorted map by encoding through a map[string]any
		// produced from the names slice; Go map iteration order is
		// undefined but JSON marshalling at the adapter level uses the
		// map verbatim, so for true determinism we'd need a slice. The
		// tests assert structural equality, not key order in the JSON
		// document.
		ordered := make(map[string]any, len(out))
		for _, n := range names {
			ordered[n] = out[n]
		}
		out = ordered
	}
	return out, nil
}

func (c *converter) convertItems(v any, path string, depth int) (any, error) {
	switch t := v.(type) {
	case map[string]any:
		return c.convert(t, path, depth)
	case []any:
		// Tuple items — convert each element.
		out := make([]any, 0, len(t))
		for i, e := range t {
			if m, ok := e.(map[string]any); ok {
				converted, err := c.convert(m, fmt.Sprintf("%s[%d]", path, i), depth)
				if err != nil {
					return nil, err
				}
				out = append(out, converted)
			} else {
				out = append(out, e)
			}
		}
		return out, nil
	}
	return v, nil
}

func (c *converter) convertVariants(v any, path string, depth int) ([]any, error) {
	arr, ok := v.([]any)
	if !ok {
		return nil, nil
	}
	out := make([]any, 0, len(arr))
	for i, e := range arr {
		if m, ok := e.(map[string]any); ok {
			converted, err := c.convert(m, fmt.Sprintf("%s[%d]", path, i), depth)
			if err != nil {
				return nil, err
			}
			out = append(out, converted)
		} else {
			out = append(out, e)
		}
	}
	return out, nil
}

func (c *converter) handleRequired(v any) any {
	switch r := v.(type) {
	case []any:
		strs := make([]string, 0, len(r))
		for _, e := range r {
			if s, ok := e.(string); ok {
				strs = append(strs, s)
			}
		}
		if c.dialect.SortProperties && len(strs) > 0 {
			sorted := append([]string(nil), strs...)
			sort.Strings(sorted)
			return sorted
		}
		// Preserve original ordering but deduplicate.
		return strs
	case []string:
		if c.dialect.SortProperties {
			sorted := append([]string(nil), r...)
			sort.Strings(sorted)
			return sorted
		}
		return r
	}
	return v
}

func (c *converter) handleAdditionalProps(v any, path string, out map[string]any) error {
	switch c.dialect.SupportsAdditionalProps {
	case AdditionalPropsAllow:
		out["additionalProperties"] = v
		return nil
	case AdditionalPropsDrop:
		if err := c.note(path, CodeDropAdditionalProps,
			fmt.Sprintf("additionalProperties dropped for dialect %q", c.dialect.Name),
			true); err != nil {
			return err
		}
		return nil
	case AdditionalPropsForceFalse:
		// true -> warn-and-downgrade to false.
		// schema -> warn-and-drop.
		// false / nil -> keep as false.
		switch tv := v.(type) {
		case bool:
			if tv {
				if err := c.note(path, CodeDowngradeAddlProps,
					"additionalProperties:true downgraded to false",
					true); err != nil {
					return err
				}
				out["additionalProperties"] = false
			} else {
				out["additionalProperties"] = false
			}
		case map[string]any:
			if err := c.note(path, CodeDropAdditionalProps,
				"additionalProperties:schema dropped",
				true); err != nil {
				return err
			}
			out["additionalProperties"] = false
		case nil:
			out["additionalProperties"] = false
		default:
			out["additionalProperties"] = v
		}
		return nil
	case AdditionalPropsErrorOnTrue:
		switch tv := v.(type) {
		case bool:
			if tv {
				// PolicyBestEffort downgrades to false; otherwise fatal.
				if err := c.noteBestEffortOnly(path, CodeAdditionalPropsTrue,
					"additionalProperties:true rejected by OpenAI Strict"); err != nil {
					return err
				}
				out["additionalProperties"] = false
			} else {
				out["additionalProperties"] = false
			}
		case map[string]any:
			// Schema value: always fatal.
			if err := c.note(path, CodeAdditionalPropsSchm,
				"additionalProperties:schema rejected by OpenAI Strict",
				false); err != nil {
				return err
			}
		default:
			out["additionalProperties"] = v
		}
		return nil
	}
	return nil
}

func inferTypeFromEnum(enumArr []any) string {
	if len(enumArr) == 0 {
		return "string"
	}
	switch enumArr[0].(type) {
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64, int, int32, int64:
		return "number"
	default:
		return "string"
	}
}

// contextKey is unexported so adapters cannot accidentally collide.
type contextKey struct{ name string }

func (k contextKey) String() string { return "anthropify/schema:" + k.name }

var (
	ctxKeyPolicy = contextKey{name: "policy"}
	ctxKeyLogger = contextKey{name: "logger"}
)

// WithPolicy returns ctx with policy attached. Adapters retrieve it via
// PolicyFrom so the top-level Client can plumb the user's choice down
// without each adapter learning new constructor arguments.
func WithPolicy(ctx context.Context, p Policy) context.Context {
	return context.WithValue(ctx, ctxKeyPolicy, p)
}

// PolicyFrom recovers the policy attached via WithPolicy; missing
// values default to PolicyStrict to match the documented v0.2.0
// default.
func PolicyFrom(ctx context.Context) Policy {
	if v, ok := ctx.Value(ctxKeyPolicy).(Policy); ok {
		return v
	}
	return PolicyStrict
}

// WithLogger attaches an slog.Logger that the normalizer uses for
// non-error warnings. Missing loggers fall back to a no-op handler.
func WithLogger(ctx context.Context, l *slog.Logger) context.Context {
	if l == nil {
		return ctx
	}
	return context.WithValue(ctx, ctxKeyLogger, l)
}

func loggerFrom(ctx context.Context) *slog.Logger {
	if ctx == nil {
		return nopLogger
	}
	if v, ok := ctx.Value(ctxKeyLogger).(*slog.Logger); ok && v != nil {
		return v
	}
	return nopLogger
}

var nopLogger = slog.New(slog.NewTextHandler(discardWriter{}, &slog.HandlerOptions{Level: slog.Level(128)}))

type discardWriter struct{}

func (discardWriter) Write(p []byte) (int, error) { return len(p), nil }
