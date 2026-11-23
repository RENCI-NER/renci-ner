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
- **Text** — un-annotated text + `location`; `Annotator.annotate()` turns it into an AnnotatedText. Deliberately not a subclass of AnnotatedText (or vice versa), so the type says whether a text is annotated
- **AnnotatedText** — container for text + annotations; provides `reannotate()` and `transform()` for chaining
- **Record[T]** — one line/row of input: `location`, every field in `cells`, and the fields to annotate in `texts` (Texts before, AnnotatedTexts after `record.annotate(annotator)`)
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
- **BagelAnnotator** (`services/linkers/`) — LLM-based re-ranker; picks among the `NormalizedAnnotation` candidates at each span; requires `BAGEL_USERNAME`/`BAGEL_PASSWORD` settings

### File formats (`src/renci_ner/formats/`)

One module per format (`text.py`, `jsonl.py`, `delimited.py`), with what they share (`open_text()`, `as_records()`, the `Format` interface) in `base.py`; `__init__.py` holds the `FORMATS` registry and re-exports the public names, so import from `renci_ner.formats`. A new format gets its own module and a `FORMATS` entry.
`reader_for_file(filename, **kwargs)` picks a `Format` by suffix (`.txt`, `.jsonl`, `.csv`, `.tsv`; `.gz` is transparent) whose `read()` yields one `Record[Text]` per line or row, at `[filename, "row=N"]`, with its texts at `[filename, "row=N", column]`; callers annotate with `record.annotate(pipeline)`. `writer_for_format(name).write(records, file)` writes Records (bare Texts/AnnotatedTexts become one-text Records via `as_records()`) to a handle, e.g. from `open_text()`. TXT skips blank lines on reading and refuses texts with line breaks on writing. JSONL writes one `Record.to_dict()` per line (cells included); it reads back location, cells and texts but not annotations (#25), and still reads single-text lines, keeping their own location.
`DelimitedFile` reads a record per data row (rows with nothing to annotate too), with a text per non-empty cell of the chosen columns (`columns_include`/`columns_exclude`, passed through `reader_for_file()`). The CSV/TSV writer streams record by record with the header from the first record, so a later record with new columns raises (#30); it writes each record's cells (renaming any that clash with `ANNOTATION_COLUMNS` to `input_<name>`), `annotation_row`, then one row per annotation.

### Key Patterns

- All services call external HTTP APIs with a default 120s timeout
- `renci_ner.utils`: `make_session(retries)` gives a retrying session to pass as `requests_session`; `forbidden(response, text, data)` logs HTTP 403s (RENCI ingress) so services return an empty result instead of aborting, and raises on any other error
- Settings (Bagel credentials, `*_URL` overrides for each service) come from `renci_ner.config.get_setting()`: the environment, then the nearest `.env` (template: `env.default`). It never loads `.env` into `os.environ`, since this is a library. Read settings at construction time, not import time
- `reannotate()` preserves/adjusts start/end offsets through the chain; 0 results keeps original annotation
- `AnnotatedText.location` is an opaque passthrough; services build results with `dataclasses.replace()` so it survives, and `annotate()` accepts a `Text` (un-annotated text with a location) and copies its location onto the (cached, shared) result; it refuses an `AnnotatedText`, whose annotations it would drop (use `reannotate()`)
- `to_dict()` on all core classes gives JSON-serializable dicts tagged with `@type`
- Services fetch their version from `/openapi.json` at the service URL
- Caching: `Annotator.annotate()` caches per (text, props) in a per-instance LRU and services implement `_annotate(text, props)`; NodeNorm caches per (identifier, conflation flags); `skip_cache: True` bypasses. Cached results are shared, so never modify an AnnotatedText a service returned
- Tests use `pytest.skip()` when a remote service CI can't reach (BioMegatron, Bagel) is unavailable. The SAPBERT test fails instead on purpose, so CI catches `sap-qdrant` being down; don't make it skip. During a known outage, mark it `xfail` for request errors only: `xfail_strict = true` in `pyproject.toml` makes it fail again once the service is back, which is the cue to remove the marker
- Python 3.11+ required (uses `typing.Self`)

## Linting

Ruff with rules: E (pycodestyle), F (pyflakes), I (isort), UP (pyupgrade). E501 is ignored. Line length 88.
