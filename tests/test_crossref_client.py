"""Optional Crossref DOI metadata fallback tests."""

from unittest.mock import Mock

import requests

from config import Settings
from services.crossref_client import CrossrefClient


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        cache_dir=str(tmp_path),
        crossref_base_url="https://crossref.test/v1",
        crossref_mailto="test@example.com",
        request_max_retries=2,
        crossref_backoff_seconds=0,
    )


def test_get_work_normalizes_crossref_message(tmp_path) -> None:
    """DOI metadata maps to the provider-neutral Paper model."""
    session = Mock()
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {
            "DOI": "10.1234/example",
            "title": ["A Crossref Paper"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "published": {"date-parts": [[2024, 1, 2]]},
            "abstract": "<jats:p>An <b>abstract</b>.</jats:p>",
        }
    }
    session.get.return_value = response

    paper = CrossrefClient(session=session, settings=_settings(tmp_path)).get_work(
        "https://doi.org/10.1234/example"
    )

    assert paper is not None
    assert paper.paper_id == "doi:10.1234/example"
    assert paper.title == "A Crossref Paper"
    assert paper.authors == ("Ada Lovelace",)
    assert paper.year == 2024
    assert paper.abstract == "An abstract."
    assert session.get.call_args.kwargs["params"]["mailto"] == "test@example.com"


def test_crossref_retries_and_degrades(tmp_path) -> None:
    """A DOI metadata outage returns None after bounded retries."""
    session = Mock()
    session.get.side_effect = requests.Timeout("offline")

    client = CrossrefClient(session=session, settings=_settings(tmp_path))

    assert client.get_work("10.1234/example") is None
    assert client.last_error == "Timeout"
    assert session.get.call_count == 2


def test_invalid_doi_does_not_make_a_request(tmp_path) -> None:
    """Blank and malformed identifiers are rejected locally."""
    session = Mock()
    client = CrossrefClient(session=session, settings=_settings(tmp_path))

    assert client.get_work("not-a-doi") is None
    session.get.assert_not_called()
