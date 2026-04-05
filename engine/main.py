'''
   gRPC Entrypoint
'''
import grpc
from concurrent import futures
import imprimer_pb2
import imprimer_pb2_grpc

from core.chains.prompt_chain import run_variant
from core.evaluator.scorer import score
from utils.create_logger import get_logger

class PromptEngineServicer(imprimer_pb2_grpc.PromptEngineServicer):
    """
    This class is the Python side of the gRPC contract. One method per
    RPC defined in the .proto file.
    Go calls EvaluatePrompt -> this method runs -> result goes back to Go.
    """
    def EvaluatePrompt(self, request, context):
      # request is strongly-typed object, not a dict
      # request.trace_id, request.variant_a, etc. all available
         result_a = run_variant(
            template=request.variant_a,
            input_text=request.input,
            task=request.task
         )
         result_b = run_variant(
            template=request.variant_b,
            input_text=request.input,
            task=request.task
         )

         score_a = score(result_a)
         score_b = score(result_b)
         winner = "a" if score_a.combined >= score_b.combined else "b"

         return imprimer_pb2.EvaluateResponse(
            trace_id=request.trace_id,
            winner=winner,
            output_a=result_a.text,
            output_b=result_b.text,
            latency_a_ms=result_a.latency_ms,
            latency_b_ms=result_b.latency_ms,
            score_a=score_a.combined,
            score_b=score_b.combined,
        )
   

def serve():
     logger = get_logger(__name__)
     server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
     imprimer_pb2_grpc.add_PromptEngineServicer_to_server(
          PromptEngineServicer(), server
     )
     server.add_insecure_port("[::]:50051") # Go will connect to this
     server.start()
     logger.info("Python gRPC server listening on :50051")
     server.wait_for_termination()

   
if __name__ == "__main__":
     serve()
