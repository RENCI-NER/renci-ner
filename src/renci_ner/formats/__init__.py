"""
Readers and writers for AnnotatedText in plain files.

A reader yields one AnnotatedText per unit of text (a line, a cell) with a `location`
of `[filename, "row=N", column]`, where column is "text" for formats without columns.
A writer takes AnnotatedTexts and a file handle. Files ending in .gz are gzipped.
"""

import csv
import functools
import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from renci_ner.core import AnnotatedText


def open_text(filename: str, mode: str = "rt", newline: str | None = None):
    """Open a text file, transparently gzipped if the name ends in .gz."""
    if Path(filename).suffix.lower() == ".gz":
        return gzip.open(filename, mode, encoding="utf-8", newline=newline)
    return open(filename, mode, encoding="utf-8", newline=newline)


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


ANNOTATION_COLUMNS = [
    "annotation_column",
    "annotation_text",
    "annotation_label",
    "annotation_id",
    "annotation_type",
    "annotation_start",
    "annotation_end",
    "annotation_prov",
]


class DelimitedFile(Format):
    """
    CSV (default) or TSV (by .tsv suffix or dialect="excel-tab"). Reading yields one
    AnnotatedText per non-empty cell in the chosen columns, with
    location=[filename, "row=N", column]. Writing puts the input cells of a row first,
    then one output row per annotation (a text with no annotations still gets one row),
    so a file can be read, annotated and written back with its columns intact.
    """

    def __init__(
        self,
        filename: str,
        columns_include: list[str] = None,
        columns_exclude: list[str] = None,
        dialect: str = None,
    ):
        super().__init__(filename)
        self.dialect = dialect or (
            "excel-tab" if format_suffix(filename) == ".tsv" else "excel"
        )
        self.columns_include = list(columns_include or [])
        self.columns_exclude = set(columns_exclude or [])

    def read(self) -> Iterator[AnnotatedText]:
        with open_text(self.filename, newline="") as f:
            reader = csv.DictReader(f, dialect=self.dialect)
            columns = self.columns_include or [
                c for c in reader.fieldnames or [] if c not in self.columns_exclude
            ]
            for row, cells in enumerate(reader, start=1):
                for column in columns:
                    text = cells.get(column) or ""
                    if text.strip():
                        yield AnnotatedText(
                            text, location=[self.filename, f"row={row}", column]
                        )

    @staticmethod
    def _row_and_column(text: AnnotatedText) -> tuple:
        """Where a text goes: everything but the last location element is the row."""
        if len(text.location) >= 2:
            return tuple(text.location[:-1]), text.location[-1]
        return (id(text),), "text"

    def write(self, texts: Iterable[AnnotatedText], file) -> None:
        texts = list(texts)
        rows: dict[tuple, dict[str, str]] = {}
        columns: list[str] = []
        for text in texts:
            row, column = self._row_and_column(text)
            rows.setdefault(row, {})[column] = text.text
            if column not in columns:
                columns.append(column)

        writer = csv.DictWriter(
            file, fieldnames=columns + ANNOTATION_COLUMNS, dialect=self.dialect
        )
        writer.writeheader()
        for text in texts:
            row, column = self._row_and_column(text)
            cells = rows[row] | {"annotation_column": column}
            if not text.annotations:
                writer.writerow(cells)
            for ann in text.annotations:
                writer.writerow(
                    cells
                    | {
                        "annotation_text": ann.text,
                        "annotation_label": ann.label,
                        "annotation_id": ann.id,
                        "annotation_type": ann.type,
                        "annotation_start": ann.start,
                        "annotation_end": ann.end,
                        "annotation_prov": " > ".join(
                            f"{p.name} {p.version}" for p in ann.provenances
                        ),
                    }
                )


FORMATS = {
    ".txt": TextFile,
    ".jsonl": JsonlFile,
    ".csv": functools.partial(DelimitedFile, dialect="excel"),
    ".tsv": functools.partial(DelimitedFile, dialect="excel-tab"),
}


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
