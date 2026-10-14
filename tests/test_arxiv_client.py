"""Tests for the arXiv service adapter."""

from __future__ import annotations

from unittest.mock import Mock

import requests

from services.arxiv_client import ArxivClient, Paper


def test_search_normalizes_atom_feed_results() -> None:
    """Search should map official Atom fields to stable paper records."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.text = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.01234v2</id>
        <title>  A Study of Agents </title>
        <summary>  We study agents. </summary>
        <published>2024-01-03T00:00:00Z</published>
        <updated>2024-01-05T00:00:00Z</updated>
        <author><name>A. Researcher</name></author>
        <category term="cs.AI"/>
        <link title="pdf" href="https://arxiv.org/pdf/2401.01234v2"/>
        <arxiv:journal_ref>Journal of Tests</arxiv:journal_ref>
        <arxiv:doi>10.1234/test</arxiv:doi>
      </entry>
    </feed>"""
    session = Mock()
    session.get.return_value = response

    papers = ArxivClient(session=session).search("agents")

    assert papers == [
        Paper(
            paper_id="2401.01234",
            title="A Study of Agents",
            abstract="We study agents.",
            authors=("A. Researcher",),
            published="2024-01-03T00:00:00Z",
            updated="2024-01-05T00:00:00Z",
            categories=("cs.AI",),
            pdf_url="https://arxiv.org/pdf/2401.01234v2",
            doi="10.1234/test",
            journal_ref="Journal of Tests",
            source="arxiv",
        )
    ]
    session.get.assert_called_once()
    assert session.get.call_args.kwargs["params"]["max_results"] == 10


def test_search_retries_then_degrades_to_empty_results() -> None:
    """A transient network failure should retry and return an empty list."""
    session = Mock()
    session.get.side_effect = requests.Timeout("temporary outage")
    client = ArxivClient(session=session, max_retries=2, backoff_seconds=0)

    assert client.search("agents") == []
    assert session.get.call_count == 2

