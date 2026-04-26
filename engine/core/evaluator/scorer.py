"""
Score = 0.60 * reachability + 0.28 * quality + 0.12 * latency

Reachability is the primary signal (RiOT paper: token-level logprob control).
Quality is task-specific similarity or diversity heuristic.
Latency is a practical budget constraint.
SSC was removed: it measures output entropy at temp>0, which is a diagnostic
tool (see stability.py), not an optimization signal. Its 10% weight was absorbed
proportionally into the remaining three terms.
"""

import math
import json
import hashlib
from dataclasses import dataclass
from typing import Optional

from core.chains.prompt_chain import VariantResult
from core.evaluator.embedder import similarity as _similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)

_SCORE_CACHE: dict = {}

OPEN_ENDED_TASKS = {
    "summarize",
    "creative_writing",
    "roleplay",
    "reasoning",
    "code_generation",
    "rewrite",
}

LATENCY_BUDGET_MS   = 1000.0
SIGMOID_STEEP       = 2.0
REACHABLE_THRESHOLD = math.log(0.40)   # ln(0.40) = -0.916

_W_REACH   = 0.60
_W_QUALITY = 0.28
_W_LATENCY = 0.12

assert abs(_W_REACH + _W_QUALITY + _W_LATENCY - 1.0) < 1e-9, "weights must sum to 1"


@dataclass
class Score:
    reachability: float
    quality:      float
    latency:      float
    combined:     float
    similarity:   Optional[float] = None   # filled for non-creative tasks


def compute_reachability(
    logprobs: list,
    baseline_logprobs: Optional[list] = None,
) -> float:
    """
    Token-level logprobs → reachability ∈ (0, 1).

    With baseline: sigmoid over delta avg logprob (relative mode).
    Without: sigmoid over absolute avg logprob vs REACHABLE_THRESHOLD.
    Returns 0.5 neutral when logprobs are empty.
    """
    if not logprobs:
        return 0.5

    def _avg(lps: list) -> float:
        valid = [t.get("logprob", -10.0) for t in lps if t.get("logprob") is not None]
        return sum(valid) / len(valid) if valid else -10.0

    conf = _avg(logprobs)

    if baseline_logprobs:
        val = 1.0 / (1.0 + math.exp(-SIGMOID_STEEP * (conf - _avg(baseline_logprobs))))
    else:
        val = 1.0 / (1.0 + math.exp(-SIGMOID_STEEP * (conf - REACHABLE_THRESHOLD)))

    return round(val, 4)


def _creative_quality_heuristic(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    diversity    = len(set(tokens)) / len(tokens)
    length_score = 1.0 / (1.0 + math.exp(-0.1 * (len(tokens) - 50)))
    return round(0.6 * diversity + 0.4 * length_score, 4)


def _quality_and_similarity(text: str, task: str, expected_output: str) -> tuple[float, float]:
    if task in OPEN_ENDED_TASKS:
        return _creative_quality_heuristic(text), 0.5

    if not expected_output:
        return 0.5, 0.5

    if task in {"classify", "extract"}:
        norm_out = text.strip().lower()
        norm_exp = expected_output.strip().lower()
        sim = 1.0 if norm_exp in norm_out else _similarity(text, expected_output)
    else:
        sim = _similarity(text, expected_output)

    return round(sim, 4), round(sim, 4)


def _combined(reach: float, quality: float, latency: float) -> float:
    return round(_W_REACH * reach + _W_QUALITY * quality + _W_LATENCY * latency, 4)


def rank_score(
    result: VariantResult,
    task: str = "",
    expected_output: str = "",
) -> Score:
    """
    Score a prompt variant. Used by GRPO for candidate ranking and by the
    evaluator for authoritative promotion decisions.

    The cache is keyed on (text, task, expected_output) so parallel GRPO calls
    and the subsequent evaluator call for the same winner pay zero cost on the
    second hit.
    """
    cache_key = hashlib.sha256(
        json.dumps(
            {"text": result.text, "task": task, "expected": expected_output},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    if cache_key in _SCORE_CACHE:
        return _SCORE_CACHE[cache_key]

    reach        = compute_reachability(result.logprobs)
    latency      = round(max(0.0, 1.0 - result.latency_ms / LATENCY_BUDGET_MS), 4)
    quality, sim = _quality_and_similarity(result.text, task, expected_output)

    s = Score(
        reachability = reach,
        quality      = quality,
        latency      = latency,
        combined     = _combined(reach, quality, latency),
        similarity   = sim,
    )
    _SCORE_CACHE[cache_key] = s
    return s


# alias used by graph.py (baseline) and main.py (A/B evaluation)
score = rank_score