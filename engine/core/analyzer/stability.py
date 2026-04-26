"""
Multi-run sampling with variance and confidence metrics
"""

import math
from dataclasses import dataclass

from core.chains.prompt_chain import ModelBackend, run_variants_parallel
from core.evaluator.embedder import pairwise_similarity
from core.evaluator.scorer import compute_reachability
from utils.create_logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenConfidence:
    token:     str
    logprob:   float
    certainty: float


@dataclass
class StabilityResult:
    outputs:          list[str]
    avg_reachability: float
    variance:         float
    avg_similarity:   float
    stability_score:  float
    token_confidence: list[TokenConfidence]   # from the first run
    recommendation:   str = ""


def analyze(
    prompt: str,
    input_text: str,
    task: str,
    backend: ModelBackend,
    n_runs: int = 5,
    temperature: float = 0.7,
) -> StabilityResult:
    """
    Runs the prompt N times and measures output stability.

    stability_score = 0.4 * avg_reachability
                    + 0.4 * avg_pairwise_similarity
                    + 0.2 * (1 - normalized_variance)
    """
    results = run_variants_parallel(
        templates=[prompt] * n_runs,
        input_text=input_text,
        task=task,
        backend=backend,
        max_workers=n_runs,
        temperature=temperature,
    )

    outputs        = []
    reachabilities = []
    all_logprobs   = []
    output_lengths = []

    for i, r in enumerate(results):
        if not r.text and not r.logprobs:
            continue
        outputs.append(r.text)
        output_lengths.append(len(r.text))
        reach = compute_reachability(r.logprobs) if r.logprobs else 0.5
        reachabilities.append(reach)
        all_logprobs.append(r.logprobs)
        logger.info(f"stability run={i+1}/{n_runs} reachability={reach:.4f} output_len={len(r.text)}")

    if not outputs:
        return StabilityResult(
            outputs=[""], avg_reachability=0.0, variance=0.0,
            avg_similarity=0.0, stability_score=0.0, token_confidence=[],
            recommendation="🔴 All stability runs failed. Check backend connection or prompt formatting.",
        )

    avg_reachability = round(sum(reachabilities) / len(reachabilities), 4)
    variance = round(
        sum((r - avg_reachability) ** 2 for r in reachabilities) / len(reachabilities), 6
    )
    avg_similarity = pairwise_similarity(outputs)

    normalized_variance = min(1.0, variance / 0.1)
    stability_score = round(
        0.4 * avg_reachability + 0.4 * avg_similarity + 0.2 * (1.0 - normalized_variance), 4
    )

    token_confidence = []
    if all_logprobs and all_logprobs[0]:
        for entry in all_logprobs[0]:
            chosen_prob = math.exp(entry.get("logprob", -10.0))
            top_probs   = [math.exp(t["logprob"]) for t in entry.get("top", [])]
            total       = sum(top_probs) or chosen_prob
            certainty   = round(chosen_prob / total, 4) if total > 0 else 0.5
            token_confidence.append(TokenConfidence(
                token=entry.get("token", ""),
                logprob=round(entry.get("logprob", -10.0), 4),
                certainty=certainty,
            ))

    length_warning = ""
    if output_lengths:
        mn, mx = min(output_lengths), max(output_lengths)
        if mn > 0 and mx / mn > 3.0:
            length_warning = (
                f" ⚠️ Output length varied significantly (min={mn}, max={mx} chars). "
                f"Try lowering temperature to 0.1–0.3 for classify/extract tasks."
            )

    if stability_score >= 0.80:
        recommendation = f"🟢 Highly stable. Ready for production.{length_warning}"
    elif stability_score >= 0.60:
        recommendation = f"🟡 Moderately stable. Consider an optimization cycle.{length_warning}"
    else:
        recommendation = f"🔴 Unstable. High drift/hallucination risk. Optimization strongly recommended.{length_warning}"

    logger.info(
        f"stability complete avg_reachability={avg_reachability:.4f} "
        f"variance={variance:.6f} avg_similarity={avg_similarity:.4f} "
        f"stability_score={stability_score:.4f}"
    )

    return StabilityResult(
        outputs=outputs,
        avg_reachability=avg_reachability,
        variance=variance,
        avg_similarity=avg_similarity,
        stability_score=stability_score,
        token_confidence=token_confidence,
        recommendation=recommendation,
    )