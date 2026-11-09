"""
Readers and writers for AnnotatedText in plain files.

A reader yields one AnnotatedText per unit of text (a line, a cell) with a `location`
of `[filename, "row=N", column]`, where column is "text" for formats without columns.
(A JSONL line that already has a location keeps it, so annotated output read back in
still points at the original input.) A writer takes AnnotatedTexts and a file handle.
Files ending in .gz are gzipped.
"""

import csv
import functools
import gzip
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from renci_ner.core import AnnotatedText


def open_text(filename: str, mode: str = "r", newline: str | None = None):
    """
    Open a UTF-8 text file, transparently gzipped if the name ends in .gz.

    The mode is always text: "r" and "w" mean "rt" and "wt" here, as they do for open()
    (gzip.open would otherwise treat them as binary). Reading uses "utf-8-sig", so a
    byte-order mark from Windows tools or Excel is dropped rather than ending up in the
    first text (or a CSV header, or breaking json.loads on the first JSONL line);
    writing never adds one. Pass newline="" for the csv module.
    """
    if "b" in mode:
        raise ValueError(f"open_text() opens text files only, not mode {mode!r}")
    if "t" not in mode:
        mode += "t"
    encoding = "utf-8-sig" if "r" in mode else "utf-8"
    if Path(filename).suffix.lower() == ".gz":
        return gzip.open(filename, mode, encoding=encoding, newline=newline)
    return open(filename, mode, encoding=encoding, newline=newline)


def format_suffix(filename: str) -> str:
    """The lowercase suffix that decides the format, ignoring a trailing .gz."""
    suffixes = [s.lower() for s in Path(filename).suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    return suffixes[-1] if suffixes else ""


class Format:
    """
    A file format that can read and/or write AnnotatedTexts. A reader is constructed
    with the file it reads; a writer needs no filename, since write() takes an open
    file handle (see open_text()).
    """

    def __init__(self, filename: str | None = None):
        self.filename = filename

    def _open(self, newline: str | None = None):
        """Open this reader's file; readers use this rather than open_text() directly."""
        if self.filename is None:
            raise ValueError(f"{type(self).__name__} was not given a file to read")
        return open_text(self.filename, newline=newline)

    def read(self) -> Iterator[AnnotatedText]:
        raise NotImplementedError

    def write(self, texts: Iterable[AnnotatedText], file) -> None:
        raise NotImplementedError


class TextFile(Format):
    """
    One text per line. Reading skips blank lines (so they aren't sent to the annotators),
    but row numbers still count them. Writing emits only the text, one per line.
    """

    def read(self) -> Iterator[AnnotatedText]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                text = line.rstrip("\n")
                if not text.strip():
                    continue
                yield AnnotatedText(
                    text, location=[self.filename, f"row={row}", "text"]
                )

    def write(self, texts: Iterable[AnnotatedText], file) -> None:
        for text in texts:
            # A line break would split the text over several lines, so it would read
            # back as several texts and shift every later row. Refuse rather than
            # silently rewrite the text; use JSONL for texts with line breaks.
            # ("\r" counts because reading in text mode treats it as a line end.)
            if "\n" in text.text or "\r" in text.text:
                raise ValueError(
                    f"Can't write a text containing a line break as TXT; use JSONL "
                    f"instead (location {text.location}): {text.text[:100]!r}"
                )
            file.write(text.text + "\n")


class JsonlFile(Format):
    """
    One AnnotatedText.to_dict() per line. Reading takes only `text` and `location`
    from each line (annotations are not read back yet), so it can be used as input.
    """

    def read(self) -> Iterator[AnnotatedText]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{self.filename}, row {row}"
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {where}: {e}") from e
                if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                    raise ValueError(
                        f'Expected a JSON object with a string "text" at {where}, '
                        f"got: {line.strip()[:100]}"
                    )
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
        filename: str | None = None,
        columns_include: list[str] = None,
        columns_exclude: list[str] = None,
        dialect: str = None,
    ):
        super().__init__(filename)
        is_tsv = filename is not None and format_suffix(filename) == ".tsv"
        self.dialect = dialect or ("excel-tab" if is_tsv else "excel")
        self.columns_include = list(columns_include or [])
        self.columns_exclude = set(columns_exclude or [])

    def read(self) -> Iterator[AnnotatedText]:
        with self._open(newline="") as f:
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
