"""
Readers and writers for AnnotatedText in plain files.

A reader yields one Text per unit of text (a line, a cell) with a `location`
of `[filename, "row=N", column]`, where column is "text" for formats without columns.
(A JSONL line that already has a location keeps it, so annotated output read back in
still points at the original input.) A CSV/TSV reader also yields a Row before each
row's texts, holding every cell of that row, so the writer can put the whole row back;
annotate only the Texts and pass Rows through. A writer takes these items and
a file handle; the TXT and JSONL writers skip Rows. Files ending in .gz are gzipped.
"""

import functools
from collections.abc import Callable

from renci_ner.formats.base import Format, Row, format_suffix, open_text
from renci_ner.formats.delimited import ANNOTATION_COLUMNS, DelimitedFile
from renci_ner.formats.jsonl import JsonlFile
from renci_ner.formats.text import TextFile

__all__ = [
    "ANNOTATION_COLUMNS",
    "FORMATS",
    "DelimitedFile",
    "Format",
    "JsonlFile",
    "Row",
    "TextFile",
    "format_suffix",
    "open_text",
    "reader_for_file",
    "writer_for_format",
]


FORMATS: dict[str, Callable[..., Format]] = {
    ".txt": TextFile,
    ".jsonl": JsonlFile,
    ".csv": functools.partial(DelimitedFile, dialect="excel"),
    ".tsv": functools.partial(DelimitedFile, dialect="excel-tab"),
}


def _format_class(suffix: str, what: str) -> Callable[..., Format]:
    """Look up a Format by suffix (e.g. ".txt"); `what` describes the request for errors."""
    if suffix not in FORMATS:
        raise ValueError(f"{what}: expected one of {sorted(FORMATS)}")
    return FORMATS[suffix]


def reader_for_file(filename: str, **kwargs) -> Format:
    """
    Pick a Format to read a file by its suffix (ignoring .gz). kwargs go to its
    constructor, such as columns_include/columns_exclude for CSV and TSV.
    """
    return _format_class(format_suffix(filename), f"Don't know how to read {filename}")(
        filename, **kwargs
    )


def writer_for_format(name: str) -> Format:
    """Pick a Format to write with by name ("txt", "jsonl", ...)."""
    suffix = "." + name.lower().lstrip(".")
    return _format_class(suffix, f"Unknown format {name!r}")()
