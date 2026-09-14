"""
Readers and writers for AnnotatedText in plain files.

A reader yields one AnnotatedText per unit of text (a line, a cell) with a `location`
of `[filename, "row=N", column]`, where column is "text" for formats without columns.
A writer takes AnnotatedTexts and a file handle. Files ending in .gz are gzipped.
"""

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from renci_ner.core import AnnotatedText


def open_text(filename: str, mode: str = "rt"):
    """Open a text file, transparently gzipped if the name ends in .gz."""
    if Path(filename).suffix.lower() == ".gz":
        return gzip.open(filename, mode, encoding="utf-8")
    return open(filename, mode, encoding="utf-8")


def format_suffix(filename: str) -> str:
    """The lowercase suffix that decides the format, ignoring a trailing .gz."""
    suffixes = [s.lower() for s in Path(filename).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


class Format:
    """A file format that can read and/or write AnnotatedTexts."""

    def __init__(self, filename: str):
        self.filename = filename

    def read(self) -> Iterator[AnnotatedText]:
        raise NotImplementedError

    def write(self, texts: Iterable[AnnotatedText], file) -> None:
        raise NotImplementedError


class TextFile(Format):
    """One text per line. Writing emits only the text, one per line."""

    def read(self) -> Iterator[AnnotatedText]:
        with open_text(self.filename) as f:
            for row, line in enumerate(f, start=1):
                yield AnnotatedText(
                    line.rstrip("\n"), location=[self.filename, f"row={row}", "text"]
                )

    def write(self, texts: Iterable[AnnotatedText], file) -> None:
        for text in texts:
            file.write(text.text + "\n")


class JsonlFile(Format):
    """
    One AnnotatedText.to_dict() per line. Reading takes only `text` and `location`
    from each line (annotations are not read back yet), so it can be used as input.
    """

    def read(self) -> Iterator[AnnotatedText]:
        with open_text(self.filename) as f:
            for row, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                data = json.loads(line)
                yield AnnotatedText(
                    data["text"],
                    location=data.get("location")
                    or [self.filename, f"row={row}", "text"],
                )

    def write(self, texts: Iterable[AnnotatedText], file) -> None:
        for text in texts:
            file.write(json.dumps(text.to_dict()) + "\n")


FORMATS: dict[str, type[Format]] = {".txt": TextFile, ".jsonl": JsonlFile}


def reader_for_file(filename: str, **kwargs) -> Format:
    """Pick a Format for a filename by its suffix (ignoring .gz); kwargs go to its constructor."""
    suffix = format_suffix(filename)
    if suffix not in FORMATS:
        raise ValueError(
            f"Don't know how to read {filename}: expected one of {sorted(FORMATS)}"
        )
    return FORMATS[suffix](filename, **kwargs)


def writer_for_format(name: str, filename: str = "-", **kwargs) -> Format:
    """Pick a Format by name ("txt", "jsonl", ...); the filename is only informational."""
    suffix = "." + name.lower().lstrip(".")
    if suffix not in FORMATS:
        raise ValueError(f"Unknown format {name!r}: expected one of {sorted(FORMATS)}")
    return FORMATS[suffix](filename, **kwargs)
