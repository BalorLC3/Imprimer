# Architecture Overview

Imprimer is built as a two-service system with a strict contract between them.
The split is intentional: keep infrastructure and model logic separate, while sharing a single protobuf schema.

## Services

### Gateway (Go)

- Receives HTTP requests from clients
- Validates auth and request content
- Generates a trace ID for each request
- Sends typed gRPC requests to the engine
- Returns structured JSON responses

Responsibilities:

- HTTP ingress and routing
- Audit middleware and trace propagation
- Metrics collection and request shaping
- Safe gRPC client connection management

### Engine (Python)

- Receives gRPC calls from the gateway
- Scans prompts for injection and unsafe content
- Executes prompt variants against an LLM backend
- Computes reachability and composite scores
- Persists results to SQLite

Responsibilities:

- backend model integration (Ollama, OpenAI)
- prompt rendering and template execution
- logprob extraction and reachability scoring
- optimizer search and trial storage
- audit-friendly trace logging

## gRPC contract

The contract lives in `proto/imprimer.proto` and defines:

- `EvaluatePrompt` for A/B prompt evaluation
- `BestVariant` for retrieving the best prompt template by task
- `OptimizePrompt` for running optimizer trials

By using gRPC, the project gets:

- cross-language protobuf bindings
- versioned request/response schemas
- fixed interface between gateway and engine

## Data flow

1. Client sends POST `/prompt` or `/optimize`
2. Gateway validates and converts JSON into protobuf messages
3. Gateway calls Python engine over gRPC
4. Engine executes model backend and computes scores
5. Engine writes evaluation/optimization history to SQLite
6. Engine returns response to gateway
7. Gateway returns JSON and metrics to the client

## Why this design

- Keeps the LLM runtime away from the web-facing service
- Makes the API layer lightweight and easy to secure
- Makes the model engine easier to test, evolve, and replace
- Supports future distributed or multi-container deployments

## Important files

- `proto/imprimer.proto` — shared schema for both services
- `gateway/cmd/main.go` — gateway startup and HTTP wiring
- `gateway/internal/client/python_client.go` — gRPC client wrapper
- `gateway/internal/handler/` — HTTP handlers for `/prompt`, `/best`, `/optimize`
- `engine/main.py` — Python gRPC server entrypoint
- `engine/core/chains/prompt_chain.py` — prompt execution and backend selection
- `engine/core/evaluator/scorer.py` — reachability and composite scoring
- `engine/core/optimize/bayesian_search.py` — current optimizer implementation
- `engine/core/registry/prompt_store.py` — evaluation persistence

## Extension points

- Add new backends by extending `ModelBackend` and its runtime adapter
- Add new scoring dimensions in `scorer.py`
- Add new endpoints by extending protobuf and gateway handler logic
- Replace SQLite with PostgreSQL by updating `prompt_store.py`
