"""Configuration tests for the retrieval milestone."""

import pytest
from pydantic import ValidationError

from config import Settings


def test_retrieval_defaults_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrieval defaults are centralized and environment-overridable."""
    settings = Settings(_env_file=None)
    assert settings.rrf_k == 60
    assert settings.rerank_top_k == 5
    assert settings.embedding_model == "BAAI/bge-m3"
    monkeypatch.setenv("LITSEARCH_EMBEDDING_MODEL", "local/model")
    monkeypatch.setenv("LITSEARCH_RETRIEVAL_CANDIDATES", "12")
    assert Settings(_env_file=None).embedding_model == "local/model"
    assert Settings(_env_file=None).retrieval_candidates == 12


@pytest.mark.parametrize("field,value", [
    ("rrf_k", 0), ("rerank_top_k", 0), ("retrieval_candidates", 0),
    ("model_batch_size", 0), ("model_max_length", 0),
    ("inference_attempts", 0), ("inference_backoff_seconds", -1),
])
def test_invalid_retrieval_settings(field: str, value: int) -> None:
    """Invalid limits fail at configuration loading, not during inference."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
