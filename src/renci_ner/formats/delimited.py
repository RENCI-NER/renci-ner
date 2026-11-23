"""CSV and TSV: one text per cell, written back with every row and column."""

import csv
from collections.abc import Iterable, Iterator

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
