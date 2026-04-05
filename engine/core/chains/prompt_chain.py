from dataclasses import dataclass, field
import time
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class VariantResult:
    text: str
    latency: float # in ms

    # logprobs is a list of dicts, one per output token
    # Each dict has a 'token' and a 'logprob' (log probability)
    # We use this in the scorer to compute the reachability index 
    # which is the controllability metric from Bhargava et al. 
    # paper "What's the magic word?", if the model does not return
    # logprobs, this stays empty and the scorer fallback to heuristic
    # scorign only
    logprobs: list = field(default_factory=list)

def run_variant(template: str, input_text: str, task: str) -> VariantResult:
    """
    Runs one prompt variant through LangChain and returns the result.

    In Minsky's framing this function activates one candidate "mind" —
    a specific configuration of the model's internal society, and
    observes what it produces. The scorer then measures how well that
    mind was steered toward the target behavior.

    template: a string like "You are a helpful assistant. {task}: {input}"
    input_text: the user's actual content
    task: what kind of task this is (summarize, classify, etc.)

    This function is a pure unit, it knows nothing about gRPC,
    nothing about scoring, nothing about the registry.
    It takes inputs and returns an output. That's it.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["task", "input"]
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        logprobs=True,
        top_logprobs=5 
    )

    chain = prompt | llm

    start = time.time()
    response = chain.invoke({"task": task, "input": input_text})
    elapsed = (time.time() - start) * 1000

    raw_logprobs = []
    try:
        lp_content = response.response_metadata.get("logprobs", {})
        if lp_content and "content" in lp_content:
            for token_data in lp_content["content"]:
                raw_logprobs.append({
                    "token": token_data["token"],
                    "logprob": token_data["logprob"],
                    "top": [
                        {"token": t["token"], "logprob": t["logprob"]}
                        for t in token_data.get("top_logprobs", [])
                    ]
                })
    except (AttributeError, KeyError, TypeError):
        # If logprobs are unavailable the scorer degrades gracefully
        pass

    return VariantResult(
        text=response.content,
        latency_ms=round(elapsed, 2),
        logprobs=raw_logprobs,
    )