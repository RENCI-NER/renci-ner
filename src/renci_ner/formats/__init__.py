"""
Readers and writers for Records in plain files.

A reader yields one Record[Text] per line or row: its `location` is
`[filename, "row=N"]`, its `cells` hold every field (so a writer can put back what
wasn't annotated), and its `texts` hold the fields to annotate, each a Text at
`[filename, "row=N", column]`, where column is "text" for formats without columns.
Annotate a record with `record.annotate(pipeline)`, and give the results to a writer,
which takes Records (or bare Texts and AnnotatedTexts) and a file handle. Files ending
in .gz are gzipped.
"""

import functools
from collections.abc import Callable

from renci_ner.core import Record, Text
from renci_ner.formats.base import Format, as_records, format_suffix, open_text
from renci_ner.formats.delimited import ANNOTATION_COLUMNS, DelimitedFile
from renci_ner.formats.jsonl import JsonlFile
from renci_ner.formats.text import TextFile

__all__ = [
    "ANNOTATION_COLUMNS",
    "FORMATS",
    "DelimitedFile",
    "Format",
    "JsonlFile",
    "Record",
    "Text",
    "TextFile",
    "as_records",
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
