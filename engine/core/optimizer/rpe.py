"""
Reflective Prompt Engineering helpers

  _generate_variants_with_residual : LLM-driven generation with RiOT residual injection
  extract_residual_content         : RiOT — structurally-aware constraint extraction
  _compute_ssc                     : Semantic Self-Consistency (diagnostic, stability.py only)
"""

import json
import re
from typing import Optional

from core.chains.prompt_chain import ModelBackend, run_variants_parallel, call_llm
from core.evaluator.scorer import compute_reachability
from core.evaluator.embedder import pairwise_similarity
from utils.create_logger import get_logger

logger = get_logger(__name__)

_SSC_RUNS        = 2
_SSC_TEMPERATURE = 0.8

_MIN_PROMPT_LEN = 20

_LABEL_PATTERNS = re.compile(
    r"^(version\s*\d+|v\d+|prompt\s*\d+|option\s*\d+|variant\s*\d+|#\s*\d+|improved version)$",
    re.IGNORECASE,
)

# Imperative verbs that open format/constraint instructions
_IMPERATIVE_VERBS = frozenset({
    "output", "return", "respond", "write", "give", "provide", "format",
    "use", "keep", "limit", "start", "ensure", "include", "exclude",
    "avoid", "do", "don't", "never", "always", "only", "omit", "strip",
})

# Negation markers
_NEGATION = frozenset({"no", "not", "never", "don't", "do not", "avoid", "without", "except"})

_SHORT_PROMPT_CHARS = 200
_SHORT_PROMPT_LINES = 3


def _is_valid_prompt(text: str, anchor: str) -> bool:
    stripped = text.strip()
    return (
        len(stripped) >= _MIN_PROMPT_LEN
        and not _LABEL_PATTERNS.match(stripped)
        and stripped != anchor.strip()
    )


def _is_constraint_line(line: str) -> bool:
    """
    Structural detection of lines worth preserving as residual constraints.

    A line is a constraint if it:
      - Starts with an imperative/format verb (output, return, only, never…)
      - Contains a negation marker (no, not, never, avoid…)
      - Contains a colon with short text on the left (format spec: "Label: …")
      - Is short and sentence-like (< 120 chars) — avoids grabbing prose

    This replaces the keyword frozenset: "Limit your answer to the label."
    now matches (starts with "Limit" → imperative verb family, < 120 chars).
    """
    s = line.strip()
    if not s or len(s) > 120:
        return False

    lower = s.lower()
    first_word = lower.split()[0].rstrip(".,:")

    if first_word in _IMPERATIVE_VERBS:
        return True
    if any(neg in lower for neg in _NEGATION):
        return True
    if ":" in s and s.index(":") < 30:   # "Output: just the label"
        return True

    return False


def extract_residual_content(prompt: str) -> str:
    """
    RiOT residual extractor.

    Short prompts (≤ _SHORT_PROMPT_CHARS or ≤ _SHORT_PROMPT_LINES) are returned
    whole — for small-model outputs that ARE a single constraint sentence.

    Long prompts: structurally scanned for imperative/negative/format lines.
    """
    stripped = prompt.strip()
    if not stripped:
        return ""

    lines = [l.strip() for l in stripped.splitlines() if l.strip()]

    if len(stripped) <= _SHORT_PROMPT_CHARS or len(lines) <= _SHORT_PROMPT_LINES:
        logger.debug(f"riot residual: short prompt ({len(stripped)} chars) — full prompt")
        return stripped

    residual = [l for l in lines if _is_constraint_line(l)]
    logger.debug(f"riot residual: extracted {len(residual)}/{len(lines)} constraint lines")
    return "\n".join(residual)


def _generate_variants_with_residual(
    base_prompt: str,
    feedback: str,
    n_variants: int,
    backend: ModelBackend,
    task: str,
    current_best_prompt: Optional[str] = None,
    residual_content: str = "",
) -> list[str]:
    """
    Calls the GENERATOR model to produce N improved prompt variants.

    RiOT injection: residual_content (proven constraints) is presented as
    lines the model must preserve, preventing semantic drift across cycles.

    Parsing pipeline (robust for models that may not output clean JSON):
      1. Strict JSON array
      2. Quoted-string extraction
      3. Non-empty line fallback
      4. [anchor] on total failure — loop never crashes
    """
    anchor = current_best_prompt or base_prompt

    feedback_block = f"\nPrevious feedback:\n{feedback}\n" if feedback.strip() else ""
    residual_block = (
        f"\nPreserve these constraints in every version:\n{residual_content}\n"
    ) if residual_content.strip() else ""

    generation_prompt = (
        f"You are improving an AI prompt for the task: {task}\n\n"
        f"Current best prompt:\n{anchor}\n"
        f"{feedback_block}"
        f"{residual_block}"
        f"Write {n_variants} improved versions. Each must:\n"
        f"- Keep {{input}} exactly as written\n"
        f"- Change only wording, tone, or instruction style (one change per version)\n"
        f"- Be a complete, usable instruction (not a label or placeholder)\n\n"
        f"Output ONLY a valid JSON array of {n_variants} strings. No other text."
    )

    raw = ""
    try:
        raw = call_llm(
            prompt_text=generation_prompt,
            backend=backend,
            temperature=0.7,
            max_tokens=500,
        )

        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if match:
            try:
                variants = json.loads(match.group())
                valid = [v for v in variants if isinstance(v, str) and _is_valid_prompt(v, anchor)]
                if valid:
                    logger.info(f"rpe: {len(valid)} variants (JSON)")
                    return valid[:n_variants]
            except json.JSONDecodeError:
                pass

        quoted = re.findall(r'"([^"]{10,})"', cleaned)
        valid  = [v.strip() for v in quoted if _is_valid_prompt(v, anchor)]
        if valid:
            logger.info(f"rpe: {len(valid)} variants (quoted fallback)")
            return valid[:n_variants]

        lines = [
            l.strip().lstrip("0123456789.-) ")
            for l in cleaned.splitlines()
            if len(l.strip()) > 15
        ]
        valid = [l for l in lines if _is_valid_prompt(l, anchor)]
        if valid:
            logger.info(f"rpe: {len(valid)} variants (line fallback)")
            return valid[:n_variants]

    except Exception as exc:
        logger.warning(f"variant generation failed: {exc} — returning anchor")

    if raw:
        logger.warning(f"all parsers failed. Raw: {raw[:200]!r}")

    return [anchor]


def _compute_ssc(
    prompt: str,
    input_example: str,
    task: str,
    backend: ModelBackend,
    k: int = _SSC_RUNS,
    temperature: float = _SSC_TEMPERATURE,
) -> tuple[float, float, str]:
    """
    Semantic Self-Consistency — diagnostic only (stability.py).
    Not used in the optimization loop.
    """
    results = run_variants_parallel(
        templates=[prompt] * k,
        input_text=input_example,
        task=task,
        backend=backend,
        temperature=temperature,
        max_workers=k,
    )

    outputs        = [r.text for r in results if r.text.strip()]
    reachabilities = [
        compute_reachability(r.logprobs) if r.logprobs else 0.6
        for r in results
    ]

    if not outputs:
        return 0.0, 0.6, ""

    ssc       = pairwise_similarity(outputs) if len(outputs) > 1 else 0.6
    avg_reach = sum(reachabilities) / len(reachabilities)
    return round(ssc, 4), round(avg_reach, 4), outputs[0]