package schema

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// loadJSON reads testdata/<name>.json and decodes it into a generic map.
func loadJSON(t *testing.T, name string) map[string]any {
	t.Helper()
	path := filepath.Join("testdata", name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return out
}

// compareSchemas checks structural equality, normalising slice element
// order for "required" (which the converter sometimes sorts to improve
// cache hit rates). Properties maps compare order-insensitively because
// they are Go maps; only their values matter.
func compareSchemas(t *testing.T, got, want map[string]any) {
	t.Helper()
	if !deepEqualSchema(got, want) {
		gotJSON, _ := json.MarshalIndent(got, "", "  ")
		wantJSON, _ := json.MarshalIndent(want, "", "  ")
		t.Fatalf("schema mismatch\n--- got ---\n%s\n--- want ---\n%s", gotJSON, wantJSON)
	}
}

func deepEqualSchema(a, b any) bool {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok {
			return false
		}
		if len(av) != len(bv) {
			return false
		}
		for k, va := range av {
			vb, ok := bv[k]
			if !ok {
				return false
			}
			if !deepEqualSchema(va, vb) {
				return false
			}
		}
		return true
	case []any:
		bv, ok := b.([]any)
		if !ok {
			return false
		}
		if len(av) != len(bv) {
			return false
		}
		// If both slices look like []string (every element is a string),
		// sort-normalise so that "required" comparisons stay
		// order-insensitive across Gemini's deterministic sort.
		if allStrings(av) && allStrings(bv) {
			as := stringsOf(av)
			bs := stringsOf(bv)
			sort.Strings(as)
			sort.Strings(bs)
			return reflect.DeepEqual(as, bs)
		}
		for i := range av {
			if !deepEqualSchema(av[i], bv[i]) {
				return false
			}
		}
		return true
	case []string:
		bv, ok := b.([]string)
		if !ok {
			// Compare against []any of strings.
			ai := make([]any, len(av))
			for i, s := range av {
				ai[i] = s
			}
			return deepEqualSchema(ai, b)
		}
		if len(av) != len(bv) {
			return false
		}
		ax := append([]string(nil), av...)
		bx := append([]string(nil), bv...)
		sort.Strings(ax)
		sort.Strings(bx)
		return reflect.DeepEqual(ax, bx)
	default:
		return reflect.DeepEqual(a, b)
	}
}

func allStrings(arr []any) bool {
	for _, e := range arr {
		if _, ok := e.(string); !ok {
			return false
		}
	}
	return true
}

func stringsOf(arr []any) []string {
	out := make([]string, len(arr))
	for i, e := range arr {
		out[i] = e.(string)
	}
	return out
}

type fixtureCase struct {
	name     string
	policy   Policy
	expected map[string]expectedResult
}

type expectedResult struct {
	// errCode, when non-empty, asserts that Normalize returns an error
	// wrapping ErrSchemaIncompatible whose first warning code matches.
	errCode string
	// warnCodes lists the codes that must appear in warnings on
	// success (order-insensitive).
	warnCodes []string
	// goldenFile, when non-empty, overrides the default
	// "<name>.<dialect>.json" path.
	goldenFile string
}

func TestNormalize_FixtureMatrix(t *testing.T) {
	dialects := map[string]Dialect{
		"anthropic":     DialectAnthropic,
		"openai_strict": DialectOpenAIStrict,
		"gemini":        DialectGemini,
		"azure_strict":  DialectAzureStrict,
	}

	cases := []fixtureCase{
		{
			name:   "01_nested",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {},
				"azure_strict":  {},
			},
		},
		{
			name:   "02_array_of_object",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {},
				"azure_strict":  {},
			},
		},
		{
			name:   "03_oneof",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {errCode: CodeRewriteOneOf},
				"azure_strict":  {},
			},
		},
		{
			name:   "03_oneof_lossy",
			policy: PolicyLossy,
			expected: map[string]expectedResult{
				"gemini": {warnCodes: []string{CodeRewriteOneOf}, goldenFile: "03_oneof.gemini.json"},
			},
		},
		{
			name:   "04_ref",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {errCode: CodeRefUnsupported},
				"gemini":        {errCode: CodeRefUnsupported},
				"azure_strict":  {errCode: CodeRefUnsupported},
			},
		},
		{
			name:   "05_addprops_true",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {errCode: CodeAdditionalPropsTrue},
				"gemini":        {errCode: CodeDropAdditionalProps},
				"azure_strict":  {errCode: CodeDowngradeAddlProps},
			},
		},
		{
			name:   "05_addprops_true_besteffort",
			policy: PolicyBestEffort,
			expected: map[string]expectedResult{
				"openai_strict": {warnCodes: []string{CodeAdditionalPropsTrue}, goldenFile: "05_addprops_true.openai_strict.json"},
				"gemini":        {warnCodes: []string{CodeDropAdditionalProps}, goldenFile: "05_addprops_true.gemini.json"},
				"azure_strict":  {warnCodes: []string{CodeDowngradeAddlProps}, goldenFile: "05_addprops_true.azure_strict.json"},
			},
		},
		{
			name:   "06_format_datetime",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {},
				"azure_strict":  {},
			},
		},
		{
			name:   "07_pattern",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {},
				"azure_strict":  {},
			},
		},
		{
			name:   "08_recursive_ref",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {errCode: CodeRefUnsupported},
				"gemini":        {errCode: CodeRefUnsupported},
				"azure_strict":  {errCode: CodeRefUnsupported},
			},
		},
		{
			name:   "09_multipleof",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {errCode: CodeDropMultipleOf},
				"azure_strict":  {},
			},
		},
		{
			name:   "09_multipleof_lossy",
			policy: PolicyLossy,
			expected: map[string]expectedResult{
				"gemini": {warnCodes: []string{CodeDropMultipleOf}, goldenFile: "09_multipleof.gemini.json"},
			},
		},
		{
			name:   "10_nullable_type",
			policy: PolicyStrict,
			expected: map[string]expectedResult{
				"anthropic":     {},
				"openai_strict": {},
				"gemini":        {},
				"azure_strict":  {},
			},
		},
	}

	for _, tc := range cases {
		canonicalName := canonicalFile(tc.name)
		canonical := loadJSON(t, canonicalName)
		for dialectName, want := range tc.expected {
			tc := tc
			dialectName := dialectName
			want := want
			t.Run(fmt.Sprintf("%s/%s", tc.name, dialectName), func(t *testing.T) {
				dialect := dialects[dialectName]
				out, warnings, err := Normalize(context.Background(), deepCopyMap(canonical), dialect, tc.policy)

				if want.errCode != "" {
					if err == nil {
						t.Fatalf("expected error %q, got nil; out=%v warnings=%v", want.errCode, out, warnings)
					}
					if !errors.Is(err, ErrSchemaIncompatible) {
						t.Fatalf("error does not wrap ErrSchemaIncompatible: %v", err)
					}
					if !hasWarningCode(warnings, want.errCode) {
						t.Fatalf("expected warning code %q in %v; err=%v", want.errCode, warnings, err)
					}
					return
				}

				if err != nil {
					t.Fatalf("unexpected error: %v (warnings=%v)", err, warnings)
				}

				goldenName := want.goldenFile
				if goldenName == "" {
					goldenName = fmt.Sprintf("%s.%s.json", tc.name, dialectName)
				}
				wantSchema := loadJSON(t, goldenName)
				compareSchemas(t, out, wantSchema)

				for _, code := range want.warnCodes {
					if !hasWarningCode(warnings, code) {
						t.Errorf("expected warning code %q, got %v", code, warnings)
					}
				}
			})
		}
	}
}

// canonicalFile resolves a fixture name to its canonical-input filename,
// allowing several test cases to share one canonical (e.g. "03_oneof"
// and "03_oneof_lossy" both load 03_oneof.canonical.json).
func canonicalFile(name string) string {
	switch name {
	case "03_oneof_lossy":
		return "03_oneof.canonical.json"
	case "05_addprops_true_besteffort":
		return "05_addprops_true.canonical.json"
	case "09_multipleof_lossy":
		return "09_multipleof.canonical.json"
	default:
		return name + ".canonical.json"
	}
}

func hasWarningCode(ws []Warning, code string) bool {
	for _, w := range ws {
		if w.Code == code {
			return true
		}
	}
	return false
}

func deepCopyMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = deepCopyAny(v)
	}
	return out
}

func deepCopyAny(v any) any {
	switch t := v.(type) {
	case map[string]any:
		return deepCopyMap(t)
	case []any:
		cp := make([]any, len(t))
		for i, e := range t {
			cp[i] = deepCopyAny(e)
		}
		return cp
	default:
		return v
	}
}

func TestNormalize_ErrSchemaIncompatibleSentinel(t *testing.T) {
	in := map[string]any{
		"type":  "object",
		"$ref":  "#",
	}
	_, _, err := Normalize(context.Background(), in, DialectGemini, PolicyStrict)
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrSchemaIncompatible) {
		t.Fatalf("expected ErrSchemaIncompatible, got %T: %v", err, err)
	}
}

func TestNormalize_DefaultPolicyIsStrict(t *testing.T) {
	// Caller passes the zero-value Policy (== PolicyStrict). A Gemini
	// oneOf must error.
	in := map[string]any{
		"type":  "object",
		"oneOf": []any{map[string]any{"type": "string"}},
	}
	_, _, err := Normalize(context.Background(), in, DialectGemini, PolicyStrict)
	if err == nil {
		t.Fatal("expected ErrSchemaIncompatible under default Strict policy")
	}
	if !errors.Is(err, ErrSchemaIncompatible) {
		t.Fatalf("not a schema-incompat error: %v", err)
	}
}

func TestNormalize_LossyPolicyAllowsOneOfRewrite(t *testing.T) {
	in := map[string]any{
		"type":  "object",
		"oneOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "number"}},
	}
	out, warnings, err := Normalize(context.Background(), in, DialectGemini, PolicyLossy)
	if err != nil {
		t.Fatalf("expected nil err under PolicyLossy, got %v", err)
	}
	if _, has := out["oneOf"]; has {
		t.Fatalf("oneOf must have been rewritten away, out=%v", out)
	}
	if _, has := out["anyOf"]; !has {
		t.Fatalf("expected anyOf in out, got %v", out)
	}
	if !hasWarningCode(warnings, CodeRewriteOneOf) {
		t.Fatalf("expected rewrite_oneof_to_anyof warning, got %v", warnings)
	}
}

func TestNormalize_NilSchema(t *testing.T) {
	out, warns, err := Normalize(context.Background(), nil, DialectGemini, PolicyStrict)
	if err != nil || out != nil || len(warns) != 0 {
		t.Fatalf("nil canonical must produce empty result, got out=%v warns=%v err=%v", out, warns, err)
	}
}
