"""Structured paper-analysis service tests."""

from types import SimpleNamespace
from unittest.mock import Mock

from config import Settings
from services.analysis_client import AnalysisClient
from services.semanticscholar_client import SemanticPaper


def _paper() -> SemanticPaper:
    """Create a minimal paper with evidence for tests."""
    return SemanticPaper(
        paper_id="paper-1",
        title="A Retrieval Study",
        abstract="We propose a retrieval method and evaluate it on three datasets.",
        year=2024,
    )


def _tool_response(arguments: str) -> SimpleNamespace:
    """Build an OpenAI-compatible function-call response fixture."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments=arguments)
                        )
                    ]
                )
            )
        ]
    )


def _settings() -> Settings:
    """Create deterministic configured LLM settings."""
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_model="test-model",
        inference_attempts=2,
        inference_backoff_seconds=0,
    )


def test_analysis_uses_structured_function_call_and_fixed_paper_id() -> None:
    """Analysis must request structured output and preserve host provenance."""
    client = Mock()
    client.chat.completions.create.return_value = _tool_response(
        '{"paper_id":"paper-1","methods":[{"text":"retrieval method",'
        '"paper_id":"wrong","evidence_quote":"retrieval method"}],'
        '"experiments":[],"conclusions":[],"reason":{"text":"retrieval method",'
        '"paper_id":"paper-1","evidence_quote":"retrieval method"}}'
    )

    result = AnalysisClient(settings=_settings(), client=client).analyze(_paper())

    assert result is not None
    assert result.paper_id == "paper-1"
    assert result.methods[0].paper_id == "paper-1"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"]["function"]["name"] == "extract_paper_findings"
    assert kwargs["tools"][0]["function"]["parameters"]["type"] == "object"


def test_analysis_retries_invalid_json_then_falls_back() -> None:
    """Malformed model output cannot break the workflow."""
    client = Mock()
    client.chat.completions.create.side_effect = [
        _tool_response('{"paper_id":'),
        _tool_response('{"paper_id":'),
    ]

    result = AnalysisClient(settings=_settings(), client=client).analyze(_paper())

    assert result is not None
    assert result.mode == "extractive"
    assert result.paper_id == "paper-1"
    assert client.chat.completions.create.call_count == 2


def test_unconfigured_analysis_is_local_and_traceable() -> None:
    """Offline fallback uses only paper evidence and keeps paper provenance."""
    result = AnalysisClient(settings=Settings(_env_file=None)).analyze(_paper())

    assert result is not None
    assert result.mode == "extractive"
    assert result.conclusions
    assert all(item.paper_id == "paper-1" for item in result.conclusions)
