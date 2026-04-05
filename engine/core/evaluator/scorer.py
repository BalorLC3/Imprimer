from dataclasses import dataclass
from chains.prompt_chain import VariantResult


@dataclass
class Score:
    latency_score: float  # 0.0-1.0, higher is faster
    length_score: float   # penalizes to short or too long
    combined: float       # weighted combination


# Latency budget 
LATENCY_BUDGET = 1000.0
# Target response length in characters 
TARGET_LENGTH = 300


def score(result: VariantResult) -> Score:
    """
    Scores a single variant result on two dimensions:
    latency (did it respond quickly?) and length (did it respond fully?).

    This is intentionally simple for the MVP.
    Real scoring would use LLM-as-judge for quality.
    """
    latency_score = max(0.0, 1.0 - (result.latency / LATENCY_BUDGET))
    # Length score: penalize responses far from target length
    length_ratio = len(result.text) / TARGET_LENGTH
    if length_ratio < 1.0:
        length_score = length_ratio          # too short, penalize proportionally
    else:
        length_score = 1.0 / length_ratio    # too long, penalize inverse

    # Combined: weight latency 40%, length 60%
    combined = (0.4 * latency_score) + (0.6 * length_score)

    return Score(
        latency_score=round(latency_score, 3),
        length_score=round(length_score, 3),
        combined=round(combined, 3)
    )