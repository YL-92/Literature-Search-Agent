"""Acceptance tests for the M0 project scaffold."""

from __future__ import annotations

from config.settings import Settings


def test_settings_load_defaults_without_api_keys() -> None:
    """Settings should be usable locally without external service credentials."""
    settings = Settings(_env_file=None)

    assert settings.app_name
    assert settings.log_level == "INFO"
    assert settings.openai_api_key is None


def test_settings_can_be_overridden_explicitly() -> None:
    """Settings should accept explicit values, as they would from environment."""
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_base_url="https://example.invalid/v1",
    )

    assert settings.openai_api_key == "test-key"
    assert str(settings.openai_base_url) == "https://example.invalid/v1"
