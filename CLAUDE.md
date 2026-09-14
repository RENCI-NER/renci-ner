# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

renci-ner is a Python library providing a unified interface for biomedical Named Entity Recognition (NER) and entity linking. It chains NER services, linkers, and transformers via a fluent API with full provenance tracking.

## Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/nameres/test_nameres.py

# Lint
uv run ruff check

# Format
uv run ruff format

# Build
uv build
```

## Architecture

### Core Data Model (`src/renci_ner/core.py`)

- **Annotation** — base data class: text, id, label, type, start/end offsets, provenance, `based_on` chain
- **NormalizedAnnotation** — extends Annotation with `biolink_type` (must start with `"biolink:"`) and CURIE id
- **AnnotatedText** — container for text + annotations; provides `reannotate()` and `transform()` for chaining
- **Annotator** / **Transformer** — interfaces that services implement
- **Pipeline** / **MultiAnnotator** — compose services; see Service Pipeline below
- **AnnotationProvenance** — tracks service name, URL, version for each step

### Service Pipeline

The fluent chaining API: `annotator.annotate(text).reannotate(linker).transform(normalizer)`,
or equivalently `Pipeline(annotator, linker, normalizer).annotate(text)`. Steps are services or
`(service, props)` tuples; `MultiAnnotator(*linkers)` gathers candidates from several linkers for
a re-ranker. Both are Annotators themselves, so they nest.

**NER** (`services/ner/`):
- **BioMegatron** — neural NER for biomedical concepts, returns raw Annotations with biolink types

**Linkers** (`services/linkers/`):
- **NameRes** — Solr-based entity linker using Babel cliques
- **BabelSAPBERTAnnotator** — SAPBERT embeddings-based linker

**Transformers**:
- **NodeNorm** (`services/normalization/`) — normalizes identifiers to preferred CURIEs via Translator Node Normalizer
- **BagelAnnotator** (`services/linkers/`) — LLM-based re-ranker; picks among the `NormalizedAnnotation` candidates at each span; requires `BAGEL_USERNAME`/`BAGEL_PASSWORD` env vars

### File formats (`src/renci_ner/formats/`)

`reader_for_file(filename)` picks a `Format` by suffix (`.txt`, `.jsonl`; `.gz` is transparent) whose `read()` yields `AnnotatedText`s with `location=[filename, "row=N", column]`; `writer_for_format(name).write(texts, file)` writes them. JSONL writes `to_dict()` and reads back only text and location.

### Key Patterns

- All services call external HTTP APIs with a default 120s timeout
- `renci_ner.utils`: `make_session(retries)` gives a retrying session to pass as `requests_session`; `forbidden(response, text, data)` logs HTTP 403s (RENCI ingress) so services return an empty result instead of aborting, and raises on any other error
- `reannotate()` preserves/adjusts start/end offsets through the chain; 0 results keeps original annotation
- `AnnotatedText.location` is an opaque passthrough; services build results with `dataclasses.replace()` so it survives
- `to_dict()` on all core classes gives JSON-serializable dicts tagged with `@type`
- Services fetch their version from `/openapi.json` at the service URL
- Caching: `Annotator.annotate()` caches per (text, props) in a per-instance LRU and services implement `_annotate(text, props)`; NodeNorm caches per (identifier, conflation flags); `skip_cache: True` bypasses. Cached results are shared, so never modify an AnnotatedText a service returned
- Tests use `pytest.skip()` when remote services are unavailable
- Python 3.11+ required (uses `typing.Self`)

## Linting

Ruff with rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade). E501 is ignored. Line length 88.
