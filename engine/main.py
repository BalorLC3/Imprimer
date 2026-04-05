"""
Imprimer engine, gRPC entrypoint.

This may be described as the cognitive layer.
"""
import grpc
from concurrent import futures

import imprimer_pb2
import imprimer_pb2_grpc

from core.chains.prompt_chain import run_variant
from core.evaluator.scorer import score
from utils.create_logger import get_logger

logger = get_logger(__name__)


class PromptEngineServicer(imprimer_pb2_grpc.PromptEngineServicer):
    """
    Python implementation of the gRPC contract.
    One method per RPC defined in imprimer.proto.
    Go calls EvaluatePrompt → this runs → result returns to Go.
    """

    def EvaluatePrompt(self, request, context):
        logger.info(f"trace={request.trace_id} task={request.task}")

        result_a = run_variant(
            template=request.variant_a,
            input_text=request.input,
            task=request.task,
        )
        result_b = run_variant(
            template=request.variant_b,
            input_text=request.input,
            task=request.task,
        )

        score_a = score(result_a)
        score_b = score(result_b)
        winner = "a" if score_a.combined >= score_b.combined else "b"

        logger.info(
            f"trace={request.trace_id} "
            f"winner={winner} "
            f"score_a={score_a.combined} "
            f"score_b={score_b.combined} "
            f"reachability_a={score_a.reachability} "
            f"reachability_b={score_b.reachability}"
        )

        # Field names must match the proto exactly:
        # latency_a not latency_a_ms — the proto fields are latency_a / latency_b
        return imprimer_pb2.EvaluateResponse(
            trace_id=request.trace_id,
            winner=winner,
            output_a=result_a.text,
            output_b=result_b.text,
            latency_a=result_a.latency_ms,
            latency_b=result_b.latency_ms,
            score_a=score_a.combined,
            score_b=score_b.combined,
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    imprimer_pb2_grpc.add_PromptEngineServicer_to_server(
        PromptEngineServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("Imprimer engine listening on :50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()