"""External cross-encoder boundary tests."""

from unittest.mock import Mock

import pytest

from config import Settings
from services.reranker_client import RerankerClient


def _settings() -> Settings:
    return Settings(_env_file=None, inference_attempts=2, inference_backoff_seconds=0)


def test_pairs_use_original_query_and_model_config() -> None:
    """FlagEmbedding receives query/document pairs and configurable options."""
    model = Mock()
    model.compute_score.return_value = [0.1, 0.9]
    factory = Mock(return_value=model)
    client = RerankerClient(settings=_settings(), model_factory=factory)
    assert client.score("query", ["first", "second"]) == [0.1, 0.9]
    assert model.compute_score.call_args.args[0] == [["query", "first"], ["query", "second"]]
    assert factory.call_args.kwargs["model_name_or_path"] == "BAAI/bge-reranker-v2-m3"
    assert factory.call_args.kwargs["local_files_only"] is True
    client.score("query", ["first", "second"])
    factory.assert_called_once()


def test_scalar_single_result_and_empty_input() -> None:
    """FlagEmbedding's singleton scalar output and empty batches are supported."""
    model = Mock()
    model.compute_score.return_value = 0.7
    client = RerankerClient(settings=_settings(), model_factory=Mock(return_value=model))
    assert client.score("q", []) == []
    model.compute_score.assert_not_called()
    assert client.score("q", ["doc"]) == [0.7]


@pytest.mark.parametrize("scores", [[1], [float("nan"), 1], [float("inf"), 1]])
def test_bad_score_batches_degrade(scores: list[float]) -> None:
    """Incomplete or non-finite scoring must not silently reorder papers."""
    model = Mock()
    model.compute_score.return_value = scores
    assert RerankerClient(settings=_settings(), model_factory=Mock(return_value=model)).score("q", ["a", "b"]) is None
    assert model.compute_score.call_count == 2


def test_model_failure_retries_then_degrades() -> None:
    """Network or unavailable-model failures are bounded."""
    factory = Mock(side_effect=OSError("unavailable"))
    assert RerankerClient(settings=_settings(), model_factory=factory).score("q", ["a"]) is None
    assert factory.call_count == 2
