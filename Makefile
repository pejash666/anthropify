# Convenience targets. The library itself has no build step; everything
# routes through `go`.

.PHONY: test test-e2e test-e2e-record vet build

build:
	go build ./...

vet:
	go vet ./...

test:
	go test ./...

# Real-network smoke tests. Requires a populated .env.e2e (see
# .env.e2e.example). Use `make test-e2e ARGS="-run TestE2E_Anthropic"`
# to filter.
test-e2e:
	@./scripts/test-e2e.sh $(ARGS)

# Same as test-e2e but flips the fixture-recording flag. The recorder
# itself is reserved; see e2e/helpers/assertions.go for the TODO.
test-e2e-record:
	@E2E_RECORD_FIXTURES=true ./scripts/test-e2e.sh $(ARGS)
