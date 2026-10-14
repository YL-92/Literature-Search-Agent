"""Bounded retry and safe diagnostic tests."""

from unittest.mock import Mock

from config import Settings
from services.inference import attempt_inference


def test_retry_then_success() -> None:
    """Transient service failures retry without delaying unit tests."""
    operation = Mock(side_effect=[OSError("offline"), 7])
    settings = Settings(_env_file=None, inference_backoff_seconds=0)
    assert attempt_inference(operation, settings, "test") == 7
    assert operation.call_count == 2


def test_failure_is_explicit_and_does_not_log_secrets(caplog) -> None:
    """Exhausted retries return None and never echo provider error payloads."""
    operation = Mock(side_effect=RuntimeError("secret-in-provider-error"))
    settings = Settings(_env_file=None, inference_attempts=2, inference_backoff_seconds=0)
    assert attempt_inference(operation, settings, "test") is None
    assert operation.call_count == 2
    assert "secret-in-provider-error" not in caplog.text
    assert "RuntimeError" in caplog.text
