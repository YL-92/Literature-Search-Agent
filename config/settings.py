"""Environment-backed application settings.环境支持的应用程序设置。"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行时配置已从环境变量和 ``.env`` 加载。"""

    model_config = SettingsConfigDict(
        env_prefix="LITSEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "litsearch-agent"
    log_level: str = "INFO"
    openai_api_key: str | None = None
    openai_base_url: HttpUrl | None = None
    openai_model: str | None = None
    semantic_scholar_api_key: str | None = None
    semantic_scholar_api_url: str | None = None
    semantic_scholar_enabled: bool = False
    scholar_provider: Literal["openalex"] = "openalex"
    openalex_base_url: str = "https://api.openalex.org"
    openalex_mailto: str | None = None
    openalex_rps: float = Field(default=8.0, gt=0, le=10)
    openalex_backoff_seconds: float = Field(default=0.5, ge=0)
    crossref_base_url: str = "https://api.crossref.org/v1"
    crossref_mailto: str | None = None
    crossref_backoff_seconds: float = Field(default=0.5, ge=0)
    semantic_cache_ttl_seconds: int = Field(default=86400, ge=0)
    semantic_cache_enabled: bool = True
    stream_chunk_size: int = Field(default=120, ge=1)
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    request_timeout: float = 20.0
    request_max_retries: int = 3
    cache_dir: str = ".cache/litsearch"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    model_device: str = "cpu"
    model_use_fp16: bool = False
    model_batch_size: int = Field(default=8, ge=1)
    model_max_length: int = Field(default=512, ge=1)
    model_local_files_only: bool = True
    inference_attempts: int = Field(default=3, ge=1)
    inference_backoff_seconds: float = Field(default=0.5, ge=0)
    retrieval_candidates: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=5, ge=1)
    hyde_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的应用设置实例。"""
    return Settings()
