"""OpenAlex adapter tests with mocked HTTP transport."""

from unittest.mock import Mock

import requests

from config import Settings
from services.openalex_client import OpenAlexClient, reconstruct_abstract


def _settings(tmp_path, **kwargs) -> Settings:
    return Settings(
        _env_file=None,
        openalex_base_url="https://openalex.test",
        openalex_mailto="test@example.com",
        openalex_rps=8,
        cache_dir=str(tmp_path),
        request_max_retries=2,
        inference_backoff_seconds=0,
        openalex_backoff_seconds=0,
        **kwargs,
    )


def _response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def test_reconstruct_abstract_handles_gaps_and_empty_values() -> None:
    """Inverted-index positions are restored into ordinary reading order."""
    assert reconstruct_abstract({"world": [1], "Hello": [0], "!": [2]}) == "Hello world !"
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_search_maps_openalex_fields_and_caches_response(tmp_path) -> None:
    """Search uses select/per-page and maps authors and abstract fields."""
    session = Mock()
    session.get.return_value = _response({
        "results": [{
            "id": "https://openalex.org/W1",
            "display_name": "A Paper",
            "publication_year": 2024,
            "cited_by_count": 7,
            "doi": "https://doi.org/10.1/test",
            "authorships": [{"author": {"display_name": "Researcher"}}],
            "abstract_inverted_index": {"A": [0], "paper": [1]},
        }]
    })
    client = OpenAlexClient(session=session, settings=_settings(tmp_path))

    papers = client.search_papers("retrieval", limit=10)
    cached = client.search_papers("retrieval", limit=10)

    assert papers == cached
    assert papers[0].paper_id == "https://openalex.org/W1"
    assert papers[0].abstract == "A paper"
    assert papers[0].authors == ("Researcher",)
    params = session.get.call_args.kwargs["params"]
    assert params["per-page"] == 10
    assert params["mailto"] == "test@example.com"
    assert "referenced_works" not in params["select"]
    assert isinstance(params["select"], str)
    assert session.get.call_count == 1


def test_get_and_backward_citations_use_referenced_works(tmp_path) -> None:
    """Details expose referenced_works as backward citation edges."""
    session = Mock()
    session.get.return_value = _response({
        "id": "https://openalex.org/W1",
        "display_name": "Seed",
        "publication_year": 2024,
        "cited_by_count": 3,
        "doi": None,
        "authorships": [],
        "abstract_inverted_index": None,
        "referenced_works": [
            "https://openalex.org/W2",
            "https://openalex.org/W3",
        ],
    })
    client = OpenAlexClient(session=session, settings=_settings(tmp_path))

    paper = client.get_paper("W1")
    backward = client.backward_citations("W1")

    assert paper is not None
    assert [item.paper_id for item in paper.references] == [
        "https://openalex.org/W2", "https://openalex.org/W3"
    ]
    assert [item.paper_id for item in backward] == [
        "https://openalex.org/W2", "https://openalex.org/W3"
    ]
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["params"]["filter"] == (
        "ids.openalex:https://openalex.org/W2|https://openalex.org/W3"
    )
    assert session.get.call_args.kwargs["params"]["per-page"] == 2


def test_forward_citations_uses_referenced_works_filter(tmp_path) -> None:
    """Forward lookup filters records by the seed in referenced_works."""
    session = Mock()
    session.get.return_value = _response({"results": [{
        "id": "https://openalex.org/W9",
        "display_name": "Citing",
        "publication_year": 2025,
        "cited_by_count": 0,
        "doi": None,
        "authorships": [],
        "primary_location": None,
    }]})
    client = OpenAlexClient(session=session, settings=_settings(tmp_path))

    result = client.forward_citations("https://openalex.org/W1")

    assert result[0].paper_id == "https://openalex.org/W9"
    assert session.get.call_args.kwargs["params"]["filter"] == "referenced_works:W1"


def test_retry_on_429_and_cache_only_successful_responses(tmp_path) -> None:
    """429 retries, then a successful response is cached."""
    first = Mock(status_code=429)
    first.raise_for_status.side_effect = requests.HTTPError("rate limited")
    second = _response({"results": []})
    session = Mock()
    session.get.side_effect = [first, second]
    client = OpenAlexClient(session=session, settings=_settings(tmp_path))

    assert client.search_papers("query") == []
    assert client.search_papers("query") == []
    assert session.get.call_count == 2


def test_retry_delay_grows_exponentially(tmp_path) -> None:
    """429 and server errors use exponential backoff after rate-limit spacing."""
    from unittest.mock import patch

    first = Mock(status_code=429)
    first.raise_for_status.side_effect = requests.HTTPError("rate limited")
    second = Mock(status_code=503)
    second.raise_for_status.side_effect = requests.HTTPError("unavailable")
    third = _response({"results": []})
    session = Mock()
    session.get.side_effect = [first, second, third]
    client = OpenAlexClient(
        session=session, settings=_settings(tmp_path),
        max_retries=3, backoff_seconds=0.25,
    )
    with patch.object(client, "_wait_for_rate_limit"), \
         patch("services.openalex_client.time.sleep") as sleep:
        assert client.search_papers("retry") == []
    assert [call.args[0] for call in sleep.call_args_list] == [0.25, 0.5]


def test_http_failure_is_explicit_for_fallback(tmp_path) -> None:
    """After retries the adapter exposes failure state without raising."""
    session = Mock()
    session.get.side_effect = requests.ConnectionError("offline")
    client = OpenAlexClient(session=session, settings=_settings(tmp_path))

    assert client.search_papers("query") == []
    assert client.last_error is not None
    assert session.get.call_count == 2


def test_rate_limiter_spaces_requests_to_configured_rps(tmp_path) -> None:
    """At 8 RPS, consecutive network starts remain at least 125 ms apart."""
    from unittest.mock import patch

    client = OpenAlexClient(settings=_settings(tmp_path))
    with patch("services.openalex_client.time.monotonic", side_effect=[0.0, 0.0, 0.01, 0.02]), \
         patch("services.openalex_client.time.sleep") as sleep:
        client._wait_for_rate_limit()
        client._wait_for_rate_limit()
    sleep.assert_called_once()
    assert sleep.call_args.args[0] == 0.115


def test_doi_identifier_is_normalized_for_detail_lookup(tmp_path) -> None:
    """A DOI is forwarded to the works endpoint as an explicit doi: identifier."""
    session = Mock()
    session.get.return_value = _response({"id": "https://openalex.org/W1"})
    client = OpenAlexClient(session=session, settings=_settings(tmp_path))

    client.get_paper("10.1234/example/with-slash")

    assert session.get.call_args.args[0].endswith(
        "/works/doi:10.1234/example/with-slash"
    )
