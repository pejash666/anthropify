package schema

// Dialect constants shipped with anthropify. Each adapter imports its
// matching dialect; downstream users do not normally need to.
//
// Cells in the capability matrix (see docs/design/v0.2.0-...md §4.1)
// drive every flag below.

// DialectAnthropic is the canonical Anthropic input_schema shape. The
// normalizer is essentially identity for this dialect and exists only
// so the conversion pipeline is uniform across adapters.
var DialectAnthropic = Dialect{
	Name:                    "anthropic",
	SupportsOneOf:           true,
	SupportsAllOf:           true,
	SupportsNot:             true,
	SupportsRefs:            true,
	SupportsExclusiveBounds: true,
	SupportsMultipleOf:      true,
	SupportsUniqueItems:     true,
	SupportsAdditionalProps: AdditionalPropsAllow,
	UseGeminiNullable:       false,
	LowercaseType:           false,
	SortProperties:          false,
	StripSchemaKey:          false,
	MaxRecursionDepth:       0,
	AllowedKeys:             nil,
}

// DialectOpenAIStrict targets the OpenAI Responses API in its
// "structured outputs / strict tools" mode. $ref / $defs / recursion
// are rejected; additionalProperties:true is rejected unless the
// caller opts in to PolicyBestEffort, which downgrades to false.
var DialectOpenAIStrict = Dialect{
	Name:                    "openai_strict",
	SupportsOneOf:           true,
	SupportsAllOf:           false, // see capability matrix
	SupportsNot:             false,
	SupportsRefs:            false,
	SupportsExclusiveBounds: true,
	SupportsMultipleOf:      true,
	SupportsUniqueItems:     true,
	SupportsAdditionalProps: AdditionalPropsErrorOnTrue,
	UseGeminiNullable:       false,
	LowercaseType:           false,
	SortProperties:          false,
	StripSchemaKey:          true,
	MaxRecursionDepth:       0,
	AllowedKeys:             nil,
}

// DialectGemini targets Vertex AI / AI Studio. The whitelist is taken
// verbatim from the production Gemini schema validator and is the
// reason oneOf, multipleOf, uniqueItems, etc. silently disappeared in
// v0.1.x; v0.2.0 surfaces the loss via warnings or errors depending on
// the active Policy.
var DialectGemini = Dialect{
	Name:                    "gemini",
	SupportsOneOf:           false,
	SupportsAllOf:           false,
	SupportsNot:             false,
	SupportsRefs:            false,
	SupportsExclusiveBounds: false,
	SupportsMultipleOf:      false,
	SupportsUniqueItems:     false,
	SupportsAdditionalProps: AdditionalPropsDrop,
	UseGeminiNullable:       true,
	LowercaseType:           true,
	SortProperties:          true,
	StripSchemaKey:          true,
	MaxRecursionDepth:       0,
	AllowedKeys: map[string]bool{
		"type":             true,
		"format":           true,
		"title":            true,
		"description":      true,
		"nullable":         true,
		"enum":             true,
		"maxItems":         true,
		"minItems":         true,
		"properties":       true,
		"required":         true,
		"minProperties":    true,
		"maxProperties":    true,
		"minLength":        true,
		"maxLength":        true,
		"pattern":          true,
		"example":          true,
		"anyOf":            true,
		"propertyOrdering": true,
		"default":          true,
		"items":            true,
		"minimum":          true,
		"maximum":          true,
	},
}

// DialectAzureStrict targets Azure-flavoured OpenAI ChatCompletion (and
// the bulk of OpenAI-compatible endpoints — Kimi, GLM, DeepSeek). It is
// permissive on most fields but defaults additionalProperties to false
// because Azure rejects unconstrained objects.
var DialectAzureStrict = Dialect{
	Name:                    "azure_strict",
	SupportsOneOf:           true,
	SupportsAllOf:           false,
	SupportsNot:             false,
	SupportsRefs:            false,
	SupportsExclusiveBounds: true,
	SupportsMultipleOf:      true,
	SupportsUniqueItems:     true,
	SupportsAdditionalProps: AdditionalPropsForceFalse,
	UseGeminiNullable:       false,
	LowercaseType:           true,
	SortProperties:          true,
	StripSchemaKey:          true,
	MaxRecursionDepth:       0,
	AllowedKeys:             nil,
}
