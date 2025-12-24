#
# csv.py - Supports reading and writing AnnotatedText objects in CSV format.
#
import csv
import logging
from pathlib import Path

from renci_ner.core import AnnotatedText


class DelimitedFile:
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
        if suffixes[-1].lower() == ".gz":
            suffixes.pop()
            self.gzipped = True
        if gzipped is not None:
            self.gzipped = gzipped

        # If we don't have a dialect, try to guess it from the file suffix.
        if dialect is not None:
            self.dialect = "excel"
        else:
            last_suffix = suffixes[-1].lower()
            if last_suffix == ".csv":
                self.dialect = "excel"
            elif last_suffix == ".tsv":
                self.dialect = "excel-tab"
            else:
                raise ValueError(f"Unsupported file type for DelimitedFile: {filename}")

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
        count_rows = 0
        with open(self.file_path) as csvfile:
            reader = csv.DictReader(csvfile, dialect=self.dialect)
            for row in reader:
                count_rows += 1

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
                    text = "\n".join([row[column] for column in columns_to_include])
                    if not include_empty and text.strip() == "":
                        continue
                    yield AnnotatedText(
                        text,
                        location=[
                            root_location,
                            self.__class__.__name__,
                            f"row={count_rows}",
                        ],
                    )
                    count_texts += 1
                else:
                    for column in columns_to_include:
                        text = row[column]
                        if not include_empty and text.strip() == "":
                            continue
                        yield AnnotatedText(
                            row[column],
                            location=[
                                root_location,
                                self.__class__.__name__,
                                f"row={count_rows}",
                                column,
                            ],
                        )
                        count_texts += 1

        self.logger.info(
            f"Generated {count_texts} AnnotatedText objects from {count_rows} rows in {self.filename}."
        )
