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
- **AnnotationProvenance** — tracks service name, URL, version for each step

### Service Pipeline

The fluent chaining API: `annotator.annotate(text).reannotate(linker).transform(normalizer)`

**NER** (`services/ner/`):
- **BioMegatron** — neural NER for biomedical concepts, returns raw Annotations with biolink types

**Linkers** (`services/linkers/`):
- **NameRes** — Solr-based entity linker using Babel cliques
- **BabelSAPBERTAnnotator** — SAPBERT embeddings-based linker
- **BagelAnnotator** — LLM-based re-ranker that combines results from multiple annotators; requires `BAGEL_USERNAME`/`BAGEL_PASSWORD` env vars

**Transformers** (`services/normalization/`):
- **NodeNorm** — normalizes identifiers to preferred CURIEs via Translator Node Normalizer

### Key Patterns

- All services call external HTTP APIs with a default 120s timeout
- `reannotate()` preserves/adjusts start/end offsets through the chain; 0 results keeps original annotation
- Services fetch their version from `/openapi.json` at the service URL
- Bagel uses `@functools.cache` for memoization
- Tests use `pytest.skip()` when remote services are unavailable
- Python 3.11+ required (uses `typing.Self`)

## Linting

Ruff with rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade). E501 is ignored. Line length 88.
