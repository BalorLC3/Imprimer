import pytest
from unittest.mock import patch, MagicMock
from engine.core.optimizer.graph import optimize, ModelBackend

# fixture for common test parameters
@pytest.fixture
def optimization_params():
    return {
        "task": "Translate to French",
        "base_prompt": "Translate: {input_text}",
        "input_example": "Hello world",
        "expected_output": "Bonjour le monde",
        "n_variants": 2,
        "max_iterations": 2,
        "backend": ModelBackend.OLLAMA
    }

# mocking external dependencies to ensure the graph logic is tested in isolation
@patch("graph_service.run_variant")
@patch("graph_service.rank_score")
@patch("graph_service._graph.stream")
def test_optimize_yields_correct_structure(mock_stream, mock_rank, mock_run, optimization_params):
    """
    Verifies that the optimize generator yields the expected dictionary structure.
    """

    mock_rank.return_value = MagicMock(combined=0.5, reachability=0.5)
    

    mock_stream.return_value = [
        {
            "controller": {
                "current_iteration": 1,
                "best_reachability": 0.8,
                "current_cycle_reachability": 0.8,
                "target_reached": False,
                "iterations_completed": 1,
                "best_prompt": "Revised Prompt",
                "last_feedback": "Improved clarity"
            }
        }
    ]

    # execute the generator
    gen = optimize(**optimization_params)
    results = list(gen)

    # assertions
    assert len(results) > 0
    first_yield = results[0]
    
    expected_keys = [
        "best_prompt", "best_reachability", "improvement", 
        "current_iteration", "target_reached"
    ]
    for key in expected_keys:
        assert key in first_yield
    
    assert first_yield["current_iteration"] == 1
    assert first_yield["best_reachability"] == 0.8

def test_optimize_baseline_failure_handling(optimization_params):
    """
    Tests if the generator yields a failure dictionary when the baseline fails.
    """
    with patch("graph_service.run_variant", side_effect=Exception("Connection Refused")):
        gen = optimize(**optimization_params)
        results = list(gen)
        
        assert len(results) == 1
        assert "Baseline evaluation failed" in results[0]["feedback"]
        assert results[0]["best_score"] == 0.0

