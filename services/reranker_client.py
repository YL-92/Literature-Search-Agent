"""Lazy FlagEmbedding cross-encoder service boundary."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from config import Settings, get_settings
from services.inference import attempt_inference


class ScoringProvider(Protocol):
    """Contract for independent query/document relevance scoring."""

    def score(self, query: str, texts: Sequence[str]) -> list[float] | None:
        """Return scores in input order, or None if scoring is unavailable."""
        ...


def _load_model(
    *, model_name_or_path: str, cache_dir: str, local_files_only: bool,
    device: str, use_fp16: bool,
) -> Any:
    # The local-only resolution happens here, not in rag/ or CLI.
    # https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download
    from FlagEmbedding import FlagReranker

    if local_files_only and not Path(model_name_or_path).is_dir():
        from huggingface_hub import snapshot_download
        model_name_or_path = snapshot_download(
            repo_id=model_name_or_path, cache_dir=cache_dir, local_files_only=True,
        )
    # https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/reranker/encoder_only/base.py
    return FlagReranker(
        model_name_or_path, use_fp16=use_fp16, devices=device, cache_dir=cache_dir,
    )


class RerankerClient:
    """Score with bge-reranker, leaving ranking policy to the RAG layer."""

    def __init__(
        self, *, settings: Settings | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Accept a lazy factory for offline, deterministic tests."""
        self.settings = settings or get_settings()
        self._factory = model_factory or _load_model
        self._model: Any = None

    def score(self, query: str, texts: Sequence[str]) -> list[float] | None:
        """Return finite relevance scores or None after bounded retries."""
        if not texts:
            return []

        def operation() -> list[float]:
            if self._model is None:
                self._model = self._factory(
                    model_name_or_path=self.settings.reranker_model,
                    cache_dir=str(Path(self.settings.cache_dir) / "models"),
                    local_files_only=self.settings.model_local_files_only,
                    device=self.settings.model_device,
                    use_fp16=self.settings.model_use_fp16,
                )
            # FlagEmbedding accepts pairs and returns list or scalar for one pair.
            raw = self._model.compute_score(
                [[query, text] for text in texts],
                batch_size=self.settings.model_batch_size,
                max_length=self.settings.model_max_length,
                normalize=True,
            )
            scores = np.asarray(raw, dtype=float).reshape(-1)
            if len(scores) != len(texts) or not np.isfinite(scores).all():
                raise ValueError("Invalid reranker score batch")
            return scores.tolist()

        return attempt_inference(operation, self.settings, "reranker")
