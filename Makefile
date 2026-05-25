# Convenience targets. The library itself has no build step; everything
# routes through `go`.

.PHONY: test test-e2e test-e2e-record vet build examples \
	example-01 example-02 example-03 example-04 example-05 example-06 example-07

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

# Compile-check every example program. Building into /dev/null keeps
# binaries from littering the repo root.
examples:
	@for d in examples/*/; do \
		go build -o /dev/null ./$$d || exit 1; \
	done

# Run a single example. Each target loads .env.e2e if present so API
# keys are available without manual sourcing.
example-01:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/01-hello
example-02:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/02-multi-provider
example-03:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/03-model-hot-switching
example-04:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/04-tool-use-portable
example-05:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/05-thinking-blocks
example-06:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/06-streaming-events
example-07:
	@set -a; [ -f .env.e2e ] && . ./.env.e2e; set +a; go run ./examples/07-anthropic-compat
