package gemini_native

import (
	"net/url"
	"strings"
	"testing"
)

func mustNew(t *testing.T, cfg Config) *Adapter {
	t.Helper()
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("New(%+v): %v", cfg, err)
	}
	return a
}

// parseURL splits a returned endpoint into (path, query). Helpful for
// asserting individual pieces without worrying about query ordering.
func parseURL(t *testing.T, raw string) (*url.URL, url.Values) {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("url.Parse(%q): %v", raw, err)
	}
	return u, u.Query()
}

func TestBuildEndpoint_StudioMode(t *testing.T) {
	a := mustNew(t, Config{
		Mode:   GeminiModeStudio,
		APIKey: "studio-key",
	})
	got, useBearer := a.buildEndpoint("gemini-2.5-flash", true)
	if useBearer {
		t.Fatalf("Studio must not use Bearer")
	}
	u, q := parseURL(t, got)
	if u.Host != "generativelanguage.googleapis.com" {
		t.Errorf("host=%q, want generativelanguage.googleapis.com", u.Host)
	}
	if u.Path != "/v1beta/models/gemini-2.5-flash:streamGenerateContent" {
		t.Errorf("path=%q", u.Path)
	}
	if q.Get("alt") != "sse" {
		t.Errorf("alt=%q, want sse", q.Get("alt"))
	}
	if q.Get("key") != "studio-key" {
		t.Errorf("key query missing or wrong: %q", q.Get("key"))
	}
}

func TestBuildEndpoint_ExpressMode(t *testing.T) {
	a := mustNew(t, Config{
		Mode:   GeminiModeExpress,
		APIKey: "AQ.express-key",
	})
	got, useBearer := a.buildEndpoint("gemini-2.5-flash", true)
	if useBearer {
		t.Fatalf("Express must not use Bearer")
	}
	u, q := parseURL(t, got)
	if u.Host != "aiplatform.googleapis.com" {
		t.Errorf("host=%q, want aiplatform.googleapis.com", u.Host)
	}
	if u.Path != "/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent" {
		t.Errorf("path=%q", u.Path)
	}
	if strings.Contains(u.Path, "projects/") {
		t.Errorf("Express path must not contain projects/: %q", u.Path)
	}
	if q.Get("alt") != "sse" {
		t.Errorf("alt=%q, want sse", q.Get("alt"))
	}
	if q.Get("key") != "AQ.express-key" {
		t.Errorf("key query wrong: %q", q.Get("key"))
	}
}

func TestBuildEndpoint_VertexMode(t *testing.T) {
	a := mustNew(t, Config{
		Mode:     GeminiModeVertex,
		Project:  "proj-123",
		Location: "us-central1",
		APIKey:   "ya29.bearer-token",
	})
	got, useBearer := a.buildEndpoint("gemini-2.5-pro", true)
	if !useBearer {
		t.Fatalf("Vertex must use Bearer")
	}
	u, q := parseURL(t, got)
	if u.Host != "aiplatform.googleapis.com" {
		t.Errorf("host=%q", u.Host)
	}
	if u.Path != "/v1/projects/proj-123/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent" {
		t.Errorf("path=%q", u.Path)
	}
	if q.Get("alt") != "sse" {
		t.Errorf("alt=%q", q.Get("alt"))
	}
	if q.Get("key") != "" {
		t.Errorf("Vertex must not include key query param: %q", q.Get("key"))
	}
}

func TestBuildEndpoint_AutoMode_FallbackToVertex(t *testing.T) {
	a := mustNew(t, Config{
		// Mode left as zero (Auto).
		Project:  "proj-auto",
		Location: "us-east5",
		APIKey:   "ya29.bearer",
	})
	if a.resolved != GeminiModeVertex {
		t.Fatalf("resolved=%d, want Vertex(%d)", a.resolved, GeminiModeVertex)
	}
	got, useBearer := a.buildEndpoint("gemini-2.5-flash", true)
	if !useBearer {
		t.Fatal("auto+vertex must use Bearer")
	}
	u, _ := parseURL(t, got)
	if !strings.Contains(u.Path, "/projects/proj-auto/locations/us-east5/") {
		t.Errorf("path missing project/location: %q", u.Path)
	}
}

func TestBuildEndpoint_AutoMode_FallbackToStudio(t *testing.T) {
	a := mustNew(t, Config{
		// Mode left as zero (Auto), no project/location.
		APIKey: "studio-only",
	})
	if a.resolved != GeminiModeStudio {
		t.Fatalf("resolved=%d, want Studio(%d)", a.resolved, GeminiModeStudio)
	}
	got, useBearer := a.buildEndpoint("gemini-2.5-flash", true)
	if useBearer {
		t.Fatal("auto+studio must not use Bearer")
	}
	u, _ := parseURL(t, got)
	if u.Host != "generativelanguage.googleapis.com" {
		t.Errorf("host=%q", u.Host)
	}
}

func TestBuildEndpoint_StreamVsNonStream(t *testing.T) {
	for _, mode := range []GeminiMode{GeminiModeStudio, GeminiModeExpress} {
		cfg := Config{Mode: mode, APIKey: "k"}
		if mode == GeminiModeVertex {
			cfg.Project = "p"
			cfg.Location = "l"
		}
		a := mustNew(t, cfg)

		streamURL, _ := a.buildEndpoint("gemini-2.5-flash", true)
		nonStreamURL, _ := a.buildEndpoint("gemini-2.5-flash", false)

		if !strings.Contains(streamURL, ":streamGenerateContent") {
			t.Errorf("mode=%d stream URL missing :streamGenerateContent: %q", mode, streamURL)
		}
		if !strings.Contains(nonStreamURL, ":generateContent") || strings.Contains(nonStreamURL, ":streamGenerateContent") {
			t.Errorf("mode=%d non-stream URL wrong: %q", mode, nonStreamURL)
		}

		_, q := parseURL(t, streamURL)
		if q.Get("alt") != "sse" {
			t.Errorf("mode=%d stream missing alt=sse: %q", mode, streamURL)
		}
		_, qns := parseURL(t, nonStreamURL)
		if qns.Get("alt") != "" {
			t.Errorf("mode=%d non-stream must not include alt: %q", mode, nonStreamURL)
		}
	}

	// Vertex separately because it also requires project/location.
	a := mustNew(t, Config{
		Mode:     GeminiModeVertex,
		Project:  "p",
		Location: "l",
		APIKey:   "bearer",
	})
	streamURL, _ := a.buildEndpoint("gemini-2.5-flash", true)
	nonStreamURL, _ := a.buildEndpoint("gemini-2.5-flash", false)
	if !strings.Contains(streamURL, ":streamGenerateContent?alt=sse") {
		t.Errorf("Vertex stream URL wrong: %q", streamURL)
	}
	if !strings.HasSuffix(nonStreamURL, ":generateContent") {
		t.Errorf("Vertex non-stream URL should not have query string: %q", nonStreamURL)
	}
}

func TestNew_RejectsInvalidConfigs(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
	}{
		{"auto without anything", Config{}},
		{"studio without key", Config{Mode: GeminiModeStudio}},
		{"express without key", Config{Mode: GeminiModeExpress}},
		{"vertex without project", Config{Mode: GeminiModeVertex, Location: "us-central1", APIKey: "b"}},
		{"vertex without location", Config{Mode: GeminiModeVertex, Project: "p", APIKey: "b"}},
		{"vertex without bearer", Config{Mode: GeminiModeVertex, Project: "p", Location: "l"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := New(tc.cfg); err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestBuildEndpoint_CustomBaseURL(t *testing.T) {
	// BaseURL override must be honoured across modes (e.g. for
	// regional aiplatform hosts or proxy testing).
	a := mustNew(t, Config{
		Mode:    GeminiModeExpress,
		APIKey:  "k",
		BaseURL: "https://us-central1-aiplatform.googleapis.com/",
	})
	got, _ := a.buildEndpoint("gemini-2.5-flash", true)
	u, _ := parseURL(t, got)
	if u.Host != "us-central1-aiplatform.googleapis.com" {
		t.Errorf("custom host not honoured: %q", u.Host)
	}
	if !strings.HasPrefix(u.Path, "/v1/publishers/google/models/") {
		t.Errorf("Express path wrong on custom base: %q", u.Path)
	}
}
