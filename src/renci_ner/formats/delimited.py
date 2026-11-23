"""CSV and TSV: one text per cell, written back with every row and column."""

import csv
from collections.abc import Iterable, Iterator

from renci_ner.core import AnnotatedText, Record, Text
from renci_ner.formats.base import Format, as_records, format_suffix

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
    CSV (default) or TSV (by .tsv suffix or dialect="excel-tab"). Reading yields a
    Record per data row: location [filename, "row=N"], every cell of the row (excluded
    and blank ones too), and a Text for each non-empty cell in the chosen columns, at
    [filename, "row=N", column]. A row with nothing to annotate still gets a Record, so
    the writer can put it back.

    Reading refuses a header that repeats a name, and a row with non-blank cells beyond
    the header, rather than lose those cells.

    Writing puts every cell of a record first, then `annotation_row` (N) and one output
    row per annotation. A text with no annotations still gets one row, and so does a
    record with no texts, so a file can be read, annotated and written back with all its
    rows and columns. A record from TXT or JSONL input has a single "text" cell. An
    input column named like one of ANNOTATION_COLUMNS (as when this output is annotated
    again) is written as "input_<name>".

    The writer holds every record in memory until the end, because the header must
    list every column before the first row is written, and records from several inputs
    (a TXT and a CSV, or CSVs with different headers) may add columns at any point. See
    #30 for ways to stream large inputs.
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

    def read(self) -> Iterator[Record[Text]]:
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
                location = [self.filename, f"row={row}"]
                yield Record(
                    location,
                    {c: cells.get(c) or "" for c in fieldnames},
                    {
                        column: Text(cells[column], location=[*location, column])
                        for column in columns
                        if (cells.get(column) or "").strip()
                    },
                )

    def write(self, records: Iterable[Record | Text | AnnotatedText], file) -> None:
        # Held in full so the header can list every column; see the class docstring.
        records = list(as_records(records))
        columns: list[str] = []
        for record in records:
            columns += [c for c in record.cells if c not in columns]

        names = self._output_names(columns)
        writer = csv.DictWriter(
            file,
            fieldnames=list(names.values()) + ANNOTATION_COLUMNS,
            dialect=self.dialect,
        )
        writer.writeheader()
        for record in records:
            row_cells = {names[c]: v for c, v in record.cells.items()} | {
                "annotation_row": self._row_number(record.location)
            }
            if not record.texts:
                writer.writerow(row_cells)
            for column, text in record.texts.items():
                text_cells = row_cells | {
                    "annotation_column": names.get(column, column)
                }
                # A Text has no annotations yet, so it gets one row like an
                # AnnotatedText with none.
                annotations = getattr(text, "annotations", [])
                if not annotations:
                    writer.writerow(text_cells)
                for ann in annotations:
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
        """N from a record location ending in "row=N", or "" if it doesn't."""
        if (
            location
            and isinstance(location[-1], str)
            and location[-1].startswith("row=")
        ):
            return location[-1].removeprefix("row=")
        return ""
