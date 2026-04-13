# LangGraph Integration Plan

This document describes the next step for Imprimer: turning the optimizer into a graph-based control loop.
The current optimizer is a monolithic Bayesian search implementation. The future design separates it into reusable nodes.

## Why LangGraph?

LangGraph is a natural fit because Imprimer is already a prompt generation and evaluation workflow.
Graph-based execution brings:

- explicit flow control
- reusable generator/evaluator components
- easier inspection of decisions
- clearer termination and loop behavior

## Proposed node design

### Generator node

The generator node produces prompt candidates from a base template.
It can be reused for:

- prompt rewrites
- few-shot injection
- step-by-step expansion

The generator node should expose:

- `input`: task, base prompt, example input
- `output`: list of candidate prompt templates

### Evaluator node

The evaluator node scores prompt candidates.
It should compute:

- reachability
- latency
- length score
- optional LLM judge feedback
- similarity to an expected answer

It can be used for both evaluation and optimization.

### Controller node

The controller node orchestrates the search.
It decides:

- which candidates to keep
- which mutation strategy to apply next
- when the search has converged
- whether to emit a refined prompt

This node maintains state across trials and can execute the graph in multiple iterations.

## Mapping to current implementation

The current `engine/core/optimize/bayesian_search.py` is the first version of the controller logic:

- it defines the mutation search space
- it evaluates candidate prompts
- it scores and stores trial results

The LangGraph refactor splits this into:

- `generator` node: the mutation space and prompt rewrites
- `evaluator` node: `score()` from `scorer.py` plus similarity checks
- `controller` node: TPE-based decision making and trial orchestration

## How it will be used

1. The gateway calls `OptimizePrompt` as it does today.
2. The engine starts a LangGraph workflow with a controller node.
3. The controller asks the generator for new candidate prompts.
4. The evaluator scores the candidates.
5. The controller decides whether to continue or return the best prompt.

This keeps the external API stable while moving internals toward a graph execution pattern.

## Benefits for Imprimer

- clearer separation between prompt generation and scoring
- easier to add new mutation strategies
- better observability of why a prompt was chosen
- simpler support for multi-agent workflows in the future

## Next step

Move the optimizer from one monolithic file into these nodes, then wire the nodes through:

- engine entrypoint
- `OptimizePrompt` RPC handler
- internal engine graph execution layer

This is the clean graph design for future prompt-control workflows.
