#
# txt.py - Supports reading and writing AnnotatedText objects in TXT format.
#
import gzip
import logging
from pathlib import Path

from renci_ner.core import AnnotatedText


class TextFile:
    def __init__(self, filename: str, encoding='utf-8', gzipped: bool = None, strip_lines: bool = True):
        file_path = Path(filename)
        suffixes = file_path.suffixes

        # Check if it's gzipped.
        if suffixes[-1].lower() == '.gz':
            suffixes.pop()
            self.gzipped = True
        else:
            self.gzipped = False

        if gzipped is not None:
            self.gzipped = gzipped

        # Set up the filenames.
        self.file_path = file_path
        self.filename = filename
        self.encoding = encoding
        self.strip_lines = strip_lines

        # Set up logging.
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return f"TextFile(filename={self.filename}, encoding={self.encoding}, gzipped={self.gzipped})"

    def read_file(self):
        self.logger.info(f"Reading {self.filename} as a text file.")
        count_rows = 0

        if self.gzipped:
            textfile = gzip.open(self.file_path, 'rt', encoding=self.encoding)
        else:
            textfile = open(self.file_path, encoding=self.encoding)

        with textfile:
            for row in textfile:
                count_rows += 1

                text = row
                if self.strip_lines:
                    text = row.strip()

                yield AnnotatedText(text, location=[
                    self.filename,
                    self.__class__.__name__,
                    f"row={count_rows}"
                ])

        self.logger.info(f"Generated {count_rows} AnnotatedText objects from {count_rows} rows in {self}.")
