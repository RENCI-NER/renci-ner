"""
Readers and writers for AnnotatedText in plain files.

A reader yields one AnnotatedText per unit of text (a line, a cell) with a `location`
of `[filename, "row=N", column]`, where column is "text" for formats without columns.
(A JSONL line that already has a location keeps it, so annotated output read back in
still points at the original input.) A CSV/TSV reader also yields a Row before each
row's texts, holding every cell of that row, so the writer can put the whole row back;
annotate only the AnnotatedTexts and pass Rows through. A writer takes these items and
a file handle; the TXT and JSONL writers skip Rows. Files ending in .gz are gzipped.
"""

import csv
import functools
import gzip
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Row:
    """
    One row of a CSV/TSV input, with every cell (excluded and blank ones too), so the
    writer can reproduce the row however few of its cells were annotated. `location` is
    [filename, "row=N"]: the location of the row's AnnotatedTexts minus their column.
    A Row is not a text: don't annotate it, just pass it on to the writer.
    """

    location: list[str]
    cells: dict[str, str]


class Format:
    """
    A file format that can read and/or write AnnotatedTexts (and, for CSV/TSV, Rows). A
    reader is constructed with the file it reads; a writer needs no filename, since
    write() takes an open file handle (see open_text()).
    """

    def __init__(self, filename: str | None = None):
        self.filename = filename

    def _open(self, newline: str | None = None):
        """Open this reader's file; readers use this rather than open_text() directly."""
        if self.filename is None:
            raise ValueError(f"{type(self).__name__} was not given a file to read")
        return open_text(self.filename, newline=newline)

    def read(self) -> Iterator[AnnotatedText | Row]:
        raise NotImplementedError

    def write(self, texts: Iterable[AnnotatedText | Row], file) -> None:
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

    def write(self, texts: Iterable[AnnotatedText | Row], file) -> None:
        for text in texts:
            if not isinstance(text, AnnotatedText):
                continue
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

    def write(self, texts: Iterable[AnnotatedText | Row], file) -> None:
        for text in texts:
            if isinstance(text, AnnotatedText):
                file.write(json.dumps(text.to_dict()) + "\n")


ANNOTATION_COLUMNS = [
    "annotation_row",
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
    CSV (default) or TSV (by .tsv suffix or dialect="excel-tab"). Reading yields, for
    each data row, a Row with all its cells, then one AnnotatedText per non-empty cell
    in the chosen columns, with location=[filename, "row=N", column].

    Reading refuses a header that repeats a name, and a row with non-blank cells beyond
    the header, rather than lose those cells.

    Writing puts every cell of a row first (from its Row), then `annotation_row` (N) and
    one output row per annotation. A text with no annotations still gets one row, and so
    does a row with no texts, so a file can be read, annotated and written back with all
    its rows and columns. An input column named like one of ANNOTATION_COLUMNS (as when
    this output is annotated again) is written as "input_<name>". Texts without a Row
    (from TXT or JSONL input, or built by hand) get a row made from their own text: texts
    with a reader's location, [filename, "row=N", column], share a row per filename and
    N, with a column named by the location; any other text gets a row of its own, in a
    "text" column.

    The writer holds every item in memory until the end, because the header must list
    every column before the first row is written, and items from several inputs (a TXT
    and a CSV, or CSVs with different headers) may add columns at any point. See #30
    for ways to stream large inputs.
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
        if columns_include and columns_exclude:
            raise ValueError(
                "Pass columns_include or columns_exclude, not both: it isn't clear "
                "whether an included column that is also excluded should be read"
            )
        # A column named twice would be read, annotated and written twice.
        self.columns_include = list(dict.fromkeys(columns_include or []))
        self.columns_exclude = set(columns_exclude or [])

    def read(self) -> Iterator[AnnotatedText | Row]:
        with self._open(newline="") as f:
            reader = csv.DictReader(f, dialect=self.dialect)
            fieldnames = reader.fieldnames or []
            # DictReader keeps only the last cell of a repeated column name, so the
            # others would vanish from both the texts and the output.
            repeated = sorted({c for c in fieldnames if fieldnames.count(c) > 1})
            if repeated:
                raise ValueError(
                    f"{self.filename} has more than one column named {repeated}; "
                    f"rename them so no cells are lost"
                )
            # A column name that isn't in the file is almost always a typo: an
            # include would yield nothing, and an exclude would silently do nothing.
            missing = [
                c
                for c in [*self.columns_include, *sorted(self.columns_exclude)]
                if c not in fieldnames
            ]
            if missing:
                raise ValueError(
                    f"{self.filename} has no column(s) {missing}; its columns are "
                    f"{fieldnames}"
                )
            columns = self.columns_include or [
                c for c in fieldnames if c not in self.columns_exclude
            ]
            for row, cells in enumerate(reader, start=1):
                # Cells beyond the header have no column to go in, so they would be
                # neither annotated nor written. Blank ones (trailing delimiters) are
                # harmless; anything else is refused rather than dropped.
                extra = [c for c in cells.get(None) or [] if c.strip()]
                if extra:
                    raise ValueError(
                        f"{self.filename} row {row} has more cells than the header "
                        f"has columns ({fieldnames}); the extra cells are {extra}"
                    )
                # A short row's missing cells come back as None; make them blank.
                yield Row(
                    [self.filename, f"row={row}"],
                    {c: cells.get(c) or "" for c in fieldnames},
                )
                for column in columns:
                    text = cells.get(column) or ""
                    if text.strip():
                        yield AnnotatedText(
                            text, location=[self.filename, f"row={row}", column]
                        )

    @staticmethod
    def _row_and_column(text: AnnotatedText) -> tuple:
        """
        Where a text goes: a (row key, column) pair. Only a location shaped like a
        reader's, [filename, "row=N", column] in strings, joins the other texts of
        that row. Anything else (from JSONL, which keeps whatever location a line
        has, or built by hand) gets a row of its own in a "text" column: guessing
        that some other shape ends in a column could merge unrelated texts into one
        row, and its elements might not even be hashable.
        """
        location = text.location
        if (
            len(location) == 3
            and all(isinstance(part, str) for part in location)
            and location[1].startswith("row=")
        ):
            return (location[0], location[1]), location[2]
        return object(), "text"

    def write(self, texts: Iterable[AnnotatedText | Row], file) -> None:
        # Held in full so the header can list every column; see the class docstring.
        items = list(texts)
        # Every row, keyed by its location minus the column, in order of first appearance,
        # with its cells and the texts written against it.
        cells: dict[tuple, dict[str, str]] = {}
        texts_by_row: dict[tuple, list[tuple[str, AnnotatedText]]] = {}
        for item in items:
            if isinstance(item, Row):
                key = tuple(item.location)
                cells[key] = dict(item.cells)
                texts_by_row.setdefault(key, [])
        rows_from_input = set(cells)
        for item in items:
            if isinstance(item, AnnotatedText):
                key, column = self._row_and_column(item)
                texts_by_row.setdefault(key, []).append((column, item))
                if key not in rows_from_input:
                    cells.setdefault(key, {})[column] = item.text

        columns: list[str] = []
        for row_cells in cells.values():
            columns += [c for c in row_cells if c not in columns]

        names = self._output_names(columns)
        writer = csv.DictWriter(
            file,
            fieldnames=list(names.values()) + ANNOTATION_COLUMNS,
            dialect=self.dialect,
        )
        writer.writeheader()
        for key, row_texts in texts_by_row.items():
            row_cells = {names[c]: v for c, v in cells.get(key, {}).items()} | {
                "annotation_row": self._row_number(key)
            }
            if not row_texts:
                writer.writerow(row_cells)
            for column, text in row_texts:
                text_cells = row_cells | {
                    "annotation_column": names.get(column, column)
                }
                if not text.annotations:
                    writer.writerow(text_cells)
                for ann in text.annotations:
                    writer.writerow(
                        text_cells
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

    @staticmethod
    def _output_names(columns: list[str]) -> dict[str, str]:
        """
        The output name of each input column. An input column named like one of
        ANNOTATION_COLUMNS (as in this writer's own output, read back in) is prefixed
        with "input_" until the name is free; otherwise the header would repeat the
        name and this run's values would overwrite the input's.
        """
        taken = set(columns) | set(ANNOTATION_COLUMNS)
        names = {}
        for column in columns:
            name = column
            if name in ANNOTATION_COLUMNS:
                while name in taken:
                    name = "input_" + name
                taken.add(name)
            names[column] = name
        return names

    @staticmethod
    def _row_number(key: tuple) -> str:
        """N from a row key ending in "row=N", or "" for a text with a row of its own."""
        if isinstance(key, tuple) and key[-1].startswith("row="):
            return key[-1].removeprefix("row=")
        return ""


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
