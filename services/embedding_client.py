"""使用可注入模型构建的懒惰 bge-m3 稠密嵌入。"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from config import Settings, get_settings
from services.inference import attempt_inference


class EmbeddingProvider(Protocol):
    """可检索兼容嵌入服务合同。"""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32] | None:
        """每段文本返回一个向量，服务不可用时返回 None。"""
        ...


class EmbeddingClient:
    """仅在需要时加载 SentenceTransformer；没有虚假向量时降级。"""

    def __init__(
        self, *, settings: Settings | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Accept a lazy factory so offline tests never download models."""
        self.settings = settings or get_settings()
        self._factory = model_factory
        self._model: Any = None

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32] | None:
        """Encode texts into normalized float32 vectors with bounded retries."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        def operation() -> NDArray[np.float32]:
            if self._model is None:
                factory = self._factory
                if factory is None:
                    # Official encode / constructor options:
                    # https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html
                    from sentence_transformers import SentenceTransformer
                    factory = SentenceTransformer
                self._model = factory(
                    model_name_or_path=self.settings.embedding_model,
                    device=self.settings.model_device,
                    cache_folder=str(Path(self.settings.cache_dir) / "models"),
                    local_files_only=self.settings.model_local_files_only,
                )
                self._model.max_seq_length = self.settings.model_max_length
            vectors = np.array(self._model.encode(
                list(texts), batch_size=self.settings.model_batch_size,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            ), dtype=np.float32, copy=True)
            if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
                raise ValueError("Embedding batch has invalid shape")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Embedding batch has invalid values")
            return np.ascontiguousarray(vectors / norms)

        return attempt_inference(operation, self.settings, "embeddings")
