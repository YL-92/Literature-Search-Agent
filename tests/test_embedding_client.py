"""Embedding service tests without network or heavyweight models."""

from unittest.mock import Mock

import numpy as np
import pytest

from config import Settings
from services.embedding_client import EmbeddingClient


def _settings() -> Settings:
    return Settings(_env_file=None, inference_attempts=2, inference_backoff_seconds=0)


def test_embedding_is_normalized_and_model_is_loaded_once() -> None:
    """The injected factory receives configuration, and vectors are unit length."""
    model = Mock()
    model.encode.return_value = [[3.0, 4.0], [0.0, 5.0]]
    factory = Mock(return_value=model)
    client = EmbeddingClient(settings=_settings(), model_factory=factory)
    vectors = client.encode(["first", "second"])
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0, 1]])
    client.encode(["first", "second"])
    factory.assert_called_once()
    assert factory.call_args.kwargs["model_name_or_path"] == "BAAI/bge-m3"
    assert factory.call_args.kwargs["local_files_only"] is True
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True


def test_empty_input_does_not_load_model() -> None:
    """Empty batches are handled without loading a model."""
    factory = Mock()
    assert EmbeddingClient(settings=_settings(), model_factory=factory).encode([]).shape == (0, 0)
    factory.assert_not_called()


@pytest.mark.parametrize("output", [[[0, 0]], [[float("nan"), 1]], [[1, 2], [2, 1]], [1, 2]])
def test_invalid_vectors_degrade(output: list) -> None:
    """Zero, non-finite, and incorrectly shaped vectors cannot enter FAISS."""
    model = Mock()
    model.encode.return_value = output
    assert EmbeddingClient(settings=_settings(), model_factory=Mock(return_value=model)).encode(["x"]) is None
    assert model.encode.call_count == 2


def test_download_failure_retries_and_recovers() -> None:
    """Failed model initialization does not poison subsequent retries."""
    model = Mock()
    model.encode.return_value = [[1, 0]]
    factory = Mock(side_effect=[OSError("network"), model])
    assert EmbeddingClient(settings=_settings(), model_factory=factory).encode(["x"]) is not None
    assert factory.call_count == 2
