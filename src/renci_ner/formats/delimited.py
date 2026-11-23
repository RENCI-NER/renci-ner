"""CSV and TSV: one text per cell, written back with every row and column."""

import csv
from collections.abc import Iterable, Iterator
from itertools import chain

from renci_ner.core import AnnotatedText
from renci_ner.formats.base import Format, Row, format_suffix

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
    this output is annotated again) is written as "input_<name>".

    The writer takes either a reader's items, each Row followed by its texts, which it
    writes row by row as they arrive, or texts alone (from TXT or JSONL input, or built
    by hand), which it holds until the end (see #30). A text alone gets a row made from
    its own text: texts with a reader's location, [filename, "row=N", column], share a
    row per filename and N, with a column named by the location; any other text gets a
    row of its own, in a "text" column.
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
        items = iter(texts)
        first = next(items, None)
        if isinstance(first, Row):
            self._write_rows(first, items, file)
        else:
            self._write_texts([] if first is None else chain([first], items), file)

    def _write_rows(self, first: Row, items: Iterator, file) -> None:
        """
        Write a reader's items, each Row followed by its texts, as they arrive: a row
        is written as soon as the next Row (or the end) shows that all its texts are
        in. So only one row is held at a time, and a run that fails partway leaves the
        rows before it written. The header comes from the first Row, so a later Row
        may not add columns (rows of one file, or of files with the same header, don't).
        """
        writer, names = self._start(file, list(first.cells))
        row, row_texts = first, []
        for item in items:
            if isinstance(item, Row):
                self._write_row(writer, names, row, row_texts)
                new = [c for c in item.cells if c not in names]
                if new:
                    raise ValueError(
                        f"The Row at {item.location} has column(s) {new} that the "
                        f"first Row ({first.location}) doesn't, and the header is "
                        f"already written; write files with different columns to "
                        f"separate outputs"
                    )
                row, row_texts = item, []
            elif isinstance(item, AnnotatedText):
                key, column = self._row_and_column(item)
                if key != tuple(row.location):
                    raise ValueError(
                        f"The text at {item.location} doesn't belong to the Row "
                        f"before it ({row.location}); pass each Row followed by its "
                        f"texts, as a reader yields them, or only texts"
                    )
                row_texts.append((column, item))
        self._write_row(writer, names, row, row_texts)

    def _write_texts(self, items: Iterable, file) -> None:
        """
        Write texts without Rows (from TXT or JSONL input, or built by hand). Each
        row's cells are its own texts, in columns named by their locations, so the
        header isn't known until every text has been seen: these are held in memory
        and written at the end.
        """
        rows: dict = {}
        for item in items:
            if isinstance(item, Row):
                raise ValueError(
                    f"The Row at {item.location} comes after a text without one; pass "
                    f"each Row followed by its texts, as a reader yields them, or only "
                    f"texts"
                )
            if isinstance(item, AnnotatedText):
                key, column = self._row_and_column(item)
                rows.setdefault(key, []).append((column, item))
        columns: list[str] = []
        for row_texts in rows.values():
            columns += [c for c, _ in row_texts if c not in columns]
        writer, names = self._start(file, columns)
        for key, row_texts in rows.items():
            location = list(key) if isinstance(key, tuple) else []
            cells = {column: text.text for column, text in row_texts}
            self._write_row(writer, names, Row(location, cells), row_texts)

    def _start(self, file, columns: list[str]) -> tuple[csv.DictWriter, dict]:
        """Write the header for these input columns; return the writer and names."""
        names = self._output_names(columns)
        writer = csv.DictWriter(
            file,
            fieldnames=list(names.values()) + ANNOTATION_COLUMNS,
            dialect=self.dialect,
        )
        writer.writeheader()
        return writer, names

    def _write_row(
        self,
        writer: csv.DictWriter,
        names: dict[str, str],
        row: Row,
        row_texts: list[tuple[str, AnnotatedText]],
    ) -> None:
        """
        Write one input row: its cells and `annotation_row`, then one output row per
        annotation. A text with no annotations, and a row with no texts, still get one.
        """
        row_cells = {names[c]: v for c, v in row.cells.items()} | {
            "annotation_row": self._row_number(row.location)
        }
        if not row_texts:
            writer.writerow(row_cells)
        for column, text in row_texts:
            text_cells = row_cells | {"annotation_column": names.get(column, column)}
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
    def _row_number(location: list[str]) -> str:
        """N from a Row location ending in "row=N", or "" for a row without one."""
        if location and str(location[-1]).startswith("row="):
            return location[-1].removeprefix("row=")
        return ""
