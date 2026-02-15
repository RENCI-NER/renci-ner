#
# csv.py - Supports reading and writing AnnotatedText objects in CSV format.
#
import csv
import json
import logging
from collections import defaultdict
from io import TextIOBase
from pathlib import Path

from renci_ner.core import AnnotatedText
from renci_ner.formats import Format


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
        suffixes = file_path.suffixes

        # Check if it's gzipped.
        if len(suffixes) > 0 and suffixes[-1].lower() == ".gz":
            suffixes.pop()
            self.gzipped = True
        if gzipped is not None:
            self.gzipped = gzipped

        # If we don't have a dialect, try to guess it from the file suffix.
        if dialect is not None:
            self.dialect = dialect
        else:
            if len(suffixes) > 0:
                last_suffix = suffixes[-1].lower()
                if last_suffix == ".csv":
                    self.dialect = "excel"
                elif last_suffix == ".tsv":
                    self.dialect = "excel-tab"
                else:
                    raise ValueError(
                        f"Unsupported file type for DelimitedFile: {filename}"
                    )
            else:
                # If all else fails, use the default dialect.
                self.dialect = "excel"

        # Set up the column filters.
        self.columns_include = (
            set(columns_include) if columns_include is not None else set()
        )
        self.columns_exclude = (
            set(columns_exclude) if columns_exclude is not None else set()
        )

        # Set up the filenames.
        self.file_path = file_path
        self.filename = filename

        # Set up the column and row information.
        self.column_names = None
        self.row_count = None

        # Set up logging.
        self.logger = logging.getLogger(__name__)

    def read_file(self, include_empty=False, combine_columns=False, root_location=None):
        """
        Reads a delimited file and generates a sequence of AnnotatedText objects based
        on its rows and columns. The function supports options to combine multiple
        columns or process them individually, while also allowing control over which
        columns to include or exclude.

        :param include_empty: A boolean flag. If True, includes cells with empty values
            in the generated AnnotatedText objects. If False, skips such rows. Defaults
            to False.
        :param combine_columns: A boolean flag. If True, combines the specified columns
            into a single AnnotatedText object per row. If False, yields individual
            AnnotatedText objects for each column. Defaults to False.
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
        with open(self.file_path) as csvfile:
            reader = csv.DictReader(csvfile, dialect=self.dialect)
            for row in reader:
                self.row_count += 1

                if self.columns_include:
                    # Only include the columns we're interested in.
                    columns_to_include = self.columns_include
                else:
                    # Include all columns.
                    columns_to_include = list(reader.fieldnames)

                    # Do we have any columns to exclude?
                    if self.columns_exclude:
                        for column in self.columns_exclude:
                            columns_to_include.remove(column)

                if combine_columns:
                    text = ""
                    for column in columns_to_include:
                        text += row[column] + "\n"
                        if column not in self.column_names:
                            self.column_names.append(column)
                    text = "\n".join([row[column] for column in columns_to_include])
                    if not include_empty and text.strip() == "":
                        continue
                    yield AnnotatedText(
                        text,
                        location=[
                            root_location,
                            self.__class__.__name__,
                            f"row={self.row_count}",
                            "combined_columns",
                        ],
                    )
                    count_texts += 1
                else:
                    for column in columns_to_include:
                        if column not in self.column_names:
                            self.column_names.append(column)
                        text = row[column]
                        if not include_empty and text.strip() == "":
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

    def write_file(
        self, texts: list[AnnotatedText], file: TextIOBase, duplicate_values=False
    ):
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

            # Step 1. Go through texts_in_row and write out the values.
            written_colnames = set()
            for text in texts_in_row:
                colname = self.get_col_name(text.location)
                if colname in written_colnames:
                    raise RuntimeError(f"Duplicate column name: {colname}")
                written_colnames.add(colname)
                row_values[colname] = text.text

            # Step 2. Write out this row as many times as necessary along with all the annotations.
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

                    if not duplicate_values:
                        # If we're not writing duplicate values, reset the row values so subsequent annotations
                        # don't duplicate those values.
                        row_values_with_annotation = {
                            colname: "" for colname in written_colnames
                        }
