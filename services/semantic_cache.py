"""Small persistent semantic-query cache for workflow outputs."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config import Settings, get_settings


class SemanticCache:
    """Persist normalized query/limit results under the configured cache directory."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        """Create the cache directory lazily and use the configured TTL."""
        self.settings = settings or get_settings()
        self.directory = Path(self.settings.cache_dir) / "semantic"
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, query: str, limit: int) -> dict[str, Any] | None:
        """Return a non-expired JSON payload or ``None``."""
        path = self._path(query, limit)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            created_at = float(payload["created_at"])
            if (
                self.settings.semantic_cache_ttl_seconds <= 0
                or time.time() - created_at > self.settings.semantic_cache_ttl_seconds
            ):
                return None
            value = payload["value"]
            return value if isinstance(value, dict) else None
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            return None

    def set(self, query: str, limit: int, value: dict[str, Any]) -> None:
        """Persist a JSON-serializable payload with the current timestamp."""
        path = self._path(query, limit)
        path.write_text(
            json.dumps(
                {"created_at": time.time(), "value": value},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _path(self, query: str, limit: int) -> Path:
        """Derive a stable path from normalized query and result limit."""
        normalized = " ".join(query.casefold().split())
        digest = hashlib.sha256(
            f"{normalized}\0{limit}".encode("utf-8")
        ).hexdigest()
        return self.directory / f"{digest}.json"
