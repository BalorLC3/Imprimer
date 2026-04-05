from dataclasses import dataclass
import time
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class VariantResult:
    text: str
    latency: float # in ms


def run_variant(template: str, input_text: str, task: str) -> VariantResult:
    """
    Runs one prompt variant through LangChain and returns the result.

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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    start = time.time()
    response = chain.invoke({"task": task, "input": input_text})
    elapsed = (time.time() - start) * 1000

    return VariantResult(
        text=response.content,
        latency=round(elapsed, 2)
    )