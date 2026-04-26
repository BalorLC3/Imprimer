"""
Shared state for the LangGraph optimization loop.

Fields are grouped by lifecycle:
  - immutable : set once at init, never change
  - mutable   : updated each cycle by generator / evaluator / controller
  - terminal  : set by controller, read by should_continue
"""

from typing import TypedDict, Optional


class PromptState(TypedDict):
    run_id: str

    # Immutable task definition 
    task: str
    input_example: str
    expected_output: str
    backend: str
    base_prompt: str           # anchor; never overwritten

    # Control parameters 
    target_score: float
    max_iterations: int
    n_variants: int            # GRPO group size

    # Cycle-local state 
    current_prompt: str        # winner produced by last generator cycle
    current_iteration: int
    last_feedback: str         # verbal explanation carried into next generation
    residual_content: str      # RiOT: proven constraints to preserve
    extra_samples: list
    
    # Best-so-far (promoted upward only, never regressed) 
    best_prompt: str
    best_reachability: float
    best_score: float
    logprobs_available: Optional[bool]   # detected on first evaluator call

    # Observability
    grpo_group_mean: float             # group mean from last GRPO step
    current_cycle_reachability: float  # this cycle's reachability (for UI timeline)

    # Baseline (set once, never changes) 
    baseline_score: float
    baseline_reachability: float

    # Terminal flags 
    target_reached: bool
    iterations_completed: int