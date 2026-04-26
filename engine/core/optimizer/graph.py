"""
LangGraph optimization graph — outer control loop.

Graph structure:
  generator → evaluator → controller → (generator | END)
"""

from typing import Generator

from langgraph.graph import StateGraph, END

from core.optimizer.state import PromptState
from core.optimizer.nodes import generator_node, evaluator_node, controller_node, should_continue
from core.chains.prompt_chain import ModelBackend, run_variant
from core.evaluator.scorer import rank_score
from utils.create_logger import get_logger

logger = get_logger(__name__)


def _build_graph() -> StateGraph:
    graph = StateGraph(PromptState)
    graph.add_node("generator",  generator_node)
    graph.add_node("evaluator",  evaluator_node)
    graph.add_node("controller", controller_node)
    graph.add_edge("generator",  "evaluator")
    graph.add_edge("evaluator",  "controller")
    graph.add_conditional_edges(
        "controller",
        should_continue,
        {"generator": "generator", "end": END},
    )
    graph.set_entry_point("generator")
    return graph.compile()


_graph = _build_graph()


def optimize(
    task: str,
    base_prompt: str,
    input_example: str = "",
    expected_output: str = "",
    n_variants: int = 3,
    backend: ModelBackend = ModelBackend.OLLAMA,
    extra_samples: list | None = None,
    target_score: float = 0.70,
    target_reachability: float = 0.80,
    max_iterations: int = 5,
) -> Generator[dict, None, None]:
    """
    Entry point for the GRPO+RiOT prompt optimization loop.

    Yields one progress dict after each controller cycle for live UI updates.
    The primary optimization signal is reachability (token-level logprobs, RiOT).
    """
    backend_str      = backend.value
    effective_target = max(target_score, target_reachability)

    try:
        baseline_result = run_variant(
            template=base_prompt, input_text=input_example,
            task=task, backend=backend,
        )
        baseline_obj = rank_score(
            result=baseline_result, expected_output=expected_output, task=task,
        )
    except Exception as exc:
        logger.error(f"baseline evaluation failed: {exc}")
        yield {
            "best_prompt": base_prompt, "best_score": 0.0, "best_reachability": 0.0,
            "baseline_score": 0.0, "baseline_reachability": 0.0, "improvement": 0.0,
            "iterations_completed": 0, "target_reached": False,
            "feedback": f"Baseline evaluation failed: {exc}", "grpo_group_mean": 0.0,
            "logprobs_available": False,
        }
        return

    baseline_score        = baseline_obj.combined
    baseline_reachability = baseline_obj.reachability

    logger.info(
        f"graph starting task={task} backend={backend_str} "
        f"baseline_reachability={baseline_reachability:.4f} "
        f"target={effective_target:.4f} max_iterations={max_iterations}"
    )

    initial_state: PromptState = {
        "run_id":                     "",
        "task":                       task,
        "input_example":              input_example,
        "expected_output":            expected_output,
        "backend":                    backend_str,
        "base_prompt":                base_prompt,
        "target_score":               effective_target,
        "max_iterations":             max_iterations,
        "n_variants":                 n_variants,
        "current_prompt":             base_prompt,
        "extra_samples":              extra_samples or [],
        "current_iteration":          0,
        "last_feedback":              "",
        "residual_content":           "",
        "best_prompt":                base_prompt,
        "best_reachability":          baseline_reachability,
        "best_score":                 baseline_score,
        "logprobs_available":         None,
        "grpo_group_mean":            0.0,
        "current_cycle_reachability": baseline_reachability,
        "baseline_score":             baseline_score,
        "baseline_reachability":      baseline_reachability,
        "target_reached":             False,
        "iterations_completed":       0,
    }

    current_state = initial_state.copy()

    try:
        for event in _graph.stream(initial_state):
            for node_name, state_update in event.items():
                if state_update:
                    current_state.update(state_update)

                if node_name == "controller":
                    best_reach  = current_state["best_reachability"]
                    cycle_reach = current_state["current_cycle_reachability"]
                    reach_delta = round(best_reach - baseline_reachability, 4)

                    logger.info(
                        f"cycle complete iter={current_state['current_iteration']} "
                        f"best_reach={best_reach:.4f} "
                        f"target_reached={current_state['target_reached']} "
                        f"improvement={reach_delta:+.4f}"
                    )

                    yield {
                        "best_prompt":           current_state["best_prompt"],
                        "best_score":            best_reach,
                        "best_reachability":     best_reach,
                        "cycle_reachability":    cycle_reach,
                        "baseline_score":        baseline_reachability,
                        "baseline_reachability": baseline_reachability,
                        "improvement":           reach_delta,
                        "current_iteration":     current_state["current_iteration"],
                        "iterations_completed":  current_state["iterations_completed"],
                        "target_reached":        current_state["target_reached"],
                        "feedback":              current_state.get("last_feedback", ""),
                        "grpo_group_mean":       current_state.get("grpo_group_mean", 0.0),
                        "logprobs_available":    current_state.get("logprobs_available", True),
                    }

    except Exception as exc:
        best_reach  = current_state.get("best_reachability", baseline_reachability)
        reach_delta = round(best_reach - baseline_reachability, 4)
        logger.error(
            f"graph stream failed at iteration "
            f"{current_state.get('current_iteration', '?')}: {exc}"
        )
        yield {
            "best_prompt":           current_state.get("best_prompt", base_prompt),
            "best_score":            best_reach,
            "best_reachability":     best_reach,
            "cycle_reachability":    best_reach,
            "baseline_score":        baseline_reachability,
            "baseline_reachability": baseline_reachability,
            "improvement":           reach_delta,
            "iterations_completed":  current_state.get("iterations_completed", 0),
            "target_reached":        False,
            "feedback":              f"Optimization interrupted: {exc}",
            "grpo_group_mean":       current_state.get("grpo_group_mean", 0.0),
            "logprobs_available":    current_state.get("logprobs_available", True),
        }