#
# csv.py - Supports reading and writing AnnotatedText objects in CSV format.
#
import csv
import gzip
import json
import logging
from collections import defaultdict
from io import TextIOBase
from pathlib import Path

from renci_ner.core import AnnotatedText
from renci_ner.formats import Format, detect_gzip


class DelimitedFile(Format):
    def __init__(
        self,
        filename: str,
        columns_include: list[str] = None,
        columns_exclude: list[str] = None,
        dialect: str = None,
        gzipped: bool = None,
    ):
        file_path = Path(filename)
        self.gzipped = detect_gzip(filename, gzipped)

        if dialect is None:
            # Infer dialect from the suffix (ignoring any .gz): .tsv is tab-delimited,
            # everything else defaults to comma-delimited.
            suffixes = [s.lower() for s in file_path.suffixes]
            if suffixes and suffixes[-1] == ".gz":
                suffixes.pop()
            last_suffix = suffixes[-1] if suffixes else ""
            dialect = "excel-tab" if last_suffix == ".tsv" else "excel"
        self.dialect = dialect

        self.columns_include = (
            set(columns_include) if columns_include is not None else set()
        )
        self.columns_exclude = (
            set(columns_exclude) if columns_exclude is not None else set()
        )

        self.file_path = file_path
        self.filename = filename

        self.column_names = None
        self.row_count = None

        self.logger = logging.getLogger(__name__)

    def read_file(self, root_location=None):
        """
        Reads a delimited file and generates a sequence of AnnotatedText objects, one
        per non-empty cell in the included columns.

        :param root_location: An optional string that specifies a root location
            to be associated with the AnnotatedText objects' location metadata. If None,
            it uses the filename of the input file.
        :return: A generator that yields AnnotatedText objects containing data and
            associated metadata derived from the processed file.
        """
        if root_location is None:
            self.logger.info(
                f"Reading {self.filename} as a DelimitedFile with dialect {self.dialect}."
            )
            root_location = self.filename
        else:
            self.logger.info(
                f"Reading {self.filename} (root location: {root_location}) as a DelimitedFile with dialect {self.dialect}."
            )
        count_texts = 0
        self.row_count = 0
        self.column_names = []
        opener = gzip.open(self.file_path, "rt") if self.gzipped else open(self.file_path)
        with opener as csvfile:
            reader = csv.DictReader(csvfile, dialect=self.dialect)
            if self.columns_include:
                columns_to_include = list(self.columns_include)
            else:
                columns_to_include = [
                    col
                    for col in (reader.fieldnames or [])
                    if col not in self.columns_exclude
                ]
            self.column_names = list(columns_to_include)
            for row in reader:
                self.row_count += 1

                for column in columns_to_include:
                    text = row[column]
                    if text.strip() == "":
                        continue
                    yield AnnotatedText(
                        row[column],
                        location=[
                            root_location,
                            self.__class__.__name__,
                            f"row={self.row_count}",
                            column,
                        ],
                    )
                    count_texts += 1

        self.logger.info(
            f"Generated {count_texts} AnnotatedText objects from {self.row_count} rows in {self.filename}."
        )

    def get_col_name(self, location):
        if isinstance(location, list) and len(location) > 0:
            return location[-1]
        else:
            # We pretend we have a single column called "text".
            return "text"

    def write_file(self, texts: list[AnnotatedText], file: TextIOBase, **kwargs):
        """Write annotated texts to a CSV file."""

        col_names = self.column_names
        if col_names is None:
            col_names = []

        # Because the rows could (theoretically) be present in any order, we need to load all the rows into memory
        # before we can write it out.
        texts_by_row = defaultdict(list)
        rownum = 0
        for text in texts:
            locations = text.location
            colname = self.get_col_name(locations)
            if colname not in col_names:
                col_names.append(colname)

            if len(locations) > 1 and locations[-2].startswith("row="):
                rownum = int(locations[-2][4:])
            else:
                rownum += 1
            texts_by_row[rownum].append(text)

        col_names.extend(
            [
                "annotation_column",
                # 'annotation_location',
                "annotation_text",
                "annotation_id",
                "annotation_type",
                "annotation_prov",
            ]
        )
        writer = csv.DictWriter(file, fieldnames=col_names, dialect=self.dialect)
        writer.writeheader()

        for rownum in sorted(texts_by_row.keys()):
            texts_in_row = texts_by_row[rownum]
            row_values = {}

            written_colnames = set()
            for text in texts_in_row:
                colname = self.get_col_name(text.location)
                if colname in written_colnames:
                    raise RuntimeError(f"Duplicate column name: {colname}")
                written_colnames.add(colname)
                row_values[colname] = text.text

            for text in texts_in_row:
                row_values_with_annotation = row_values.copy()
                colname = self.get_col_name(text.location)

                for ann in text.annotations:
                    row_values_with_annotation["annotation_column"] = colname
                    # row_values_with_annotation["annotation_location"] = json.dumps(text.location)
                    row_values_with_annotation["annotation_text"] = ann.text
                    row_values_with_annotation["annotation_id"] = ann.id
                    row_values_with_annotation["annotation_type"] = ann.type

                    provenances = ann.provenances
                    row_values_with_annotation["annotation_prov"] = json.dumps(
                        [prov.to_dict() for prov in provenances]
                    )

                    writer.writerow(row_values_with_annotation)

                    # Reset the row values so subsequent annotations in the same row
                    # don't duplicate the cell values already written.
                    row_values_with_annotation = {
                        colname: "" for colname in written_colnames
                    }
