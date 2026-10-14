"""HyDE generation tests against an injected OpenAI-compatible transport."""

from types import SimpleNamespace
from unittest.mock import Mock

from config import Settings
from services.llm_client import LLMClient


def _settings(**kwargs) -> Settings:
    return Settings(
        _env_file=None, openai_model="test-model",
        inference_attempts=2, inference_backoff_seconds=0, **kwargs,
    )


def _response(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_hyde_requests_hypothetical_text_with_configured_model() -> None:
    """The generated abstract is retrieval text, not a cited research claim."""
    sdk = Mock()
    sdk.chat.completions.create.return_value = _response("  Hypothetical abstract.  ")
    assert LLMClient(settings=_settings(), client=sdk).hypothetical_abstract("agents") == "Hypothetical abstract."
    args = sdk.chat.completions.create.call_args.kwargs
    assert args["model"] == "test-model"
    assert args["messages"][-1]["content"] == "agents"
    assert "hypothetical" in args["messages"][0]["content"].lower()


def test_hyde_retries_malformed_or_failed_responses() -> None:
    """Blank and failed responses degrade to the original query."""
    sdk = Mock()
    sdk.chat.completions.create.side_effect = [OSError("offline"), _response(None)]
    assert LLMClient(settings=_settings(), client=sdk).hypothetical_abstract("agents") == "agents"
    assert sdk.chat.completions.create.call_count == 2


def test_hyde_recovers_on_retry() -> None:
    """A later valid response is returned after a transient error."""
    sdk = Mock()
    sdk.chat.completions.create.side_effect = [ValueError("bad"), _response("Recovered")]
    assert LLMClient(settings=_settings(), client=sdk).hypothetical_abstract("agents") == "Recovered"


def test_no_configuration_or_blank_query_makes_no_request() -> None:
    """No credentials are required for ordinary local retrieval."""
    client = LLMClient(settings=Settings(_env_file=None, openai_api_key=None, openai_model=None))
    assert client.hypothetical_abstract("agents") == "agents"
    sdk = Mock()
    assert LLMClient(settings=_settings(), client=sdk).hypothetical_abstract("  ") == ""
    sdk.chat.completions.create.assert_not_called()
