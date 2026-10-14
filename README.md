# Literature Search Agent

A Python 3.11 command-line application for research-paper discovery, analysis,
and citation tracing.

## M0 scaffold

Create and activate a virtual environment, install the project with its
development dependencies, then run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python cli.py --help
```

Copy `.env.example` to `.env` and provide credentials only when using services
that require them. Configuration is read through Pydantic Settings.

## M4 workflow

The CLI now runs a traceable workflow:

```powershell
python cli.py search "retrieval augmented generation" --limit 5
python cli.py search "citation graph provenance" --limit 5 --json
```

The workflow uses OpenAlex as the primary scholarly source, with Semantic
Scholar disabled by default. It performs retrieval, lightweight relevance
filtering, structured paper analysis, and citation verification. Every accepted recommendation
contains a `paper_id`, an evidence quote, and an explicit analysis mode.
The standard search command also emits a comparison table, a source-tagged
review, and numeric `faithfulness` / `context_recall` metrics; `--review` and
`--eval` remain accepted as explicit switches.
Repeated normalized queries use the local semantic-query cache, and `--stream`
prints the review progressively:

```powershell
python cli.py search "retrieval augmented generation" --limit 5 --stream
python cli.py search "  Retrieval   Augmented Generation " --limit 5 --json
```

The second command reuses the first query's cached workflow result when the
cache TTL has not expired. The JSON output includes `cache_hit`.
When the LLM is unavailable, the workflow uses a clearly labeled extractive
fallback rather than inventing findings. Network and model failures degrade to
structured empty or partial results.

OpenAlex configuration is available in `.env`:

```text
LITSEARCH_SCHOLAR_PROVIDER=openalex
LITSEARCH_OPENALEX_MAILTO=your@email.com
LITSEARCH_OPENALEX_RPS=8
LITSEARCH_SEMANTIC_SCHOLAR_ENABLED=false
```

## End-to-end demo

From the project root:

```powershell
python cli.py search "citation graph provenance" --limit 5 --stream
```

This runs OpenAlex retrieval, filtering, paper analysis, citation verification,
source-tagged review generation, and the two evaluation metrics. If no LLM
credentials are configured, the output explicitly uses the extractive
fallback; no unsupported conclusions are added.

## Regression question set

The project includes 120 Chinese research questions across 12 categories:
`data/test_questions.jsonl`. Validate its size, uniqueness, and category
coverage with:

```powershell
python scripts/validate_question_set.py
```

The question-set benchmark distinguishes real RAGAS results from the
deterministic provenance proxy. It will not report a fabricated RAGAS
improvement when no model-generated benchmark outputs are available.

## Web frontend

The lightweight FastAPI frontend uses only `frontend/index.html` and browser
JavaScript; there is no npm or React build step:

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. The page provides paper search, SSE review
streaming, clickable source markers, OpenAlex citation graph loading, and the
evaluation metrics bar.
