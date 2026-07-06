#
# txt.py - Supports reading and writing AnnotatedText objects in TXT format.
#
import gzip
import logging
from io import TextIOBase
from pathlib import Path

from renci_ner.core import AnnotatedText
from renci_ner.formats import Format, detect_gzip


class TextFile(Format):
    def __init__(
        self,
        filename: str,
        encoding="utf-8",
        gzipped: bool = None,
        strip_lines: bool = True,
    ):
        file_path = Path(filename)
        self.gzipped = detect_gzip(filename, gzipped)

        self.file_path = file_path
        self.filename = filename
        self.encoding = encoding
        self.strip_lines = strip_lines

        self.logger = logging.getLogger(__name__)

    def __str__(self):
        return f"TextFile(filename={self.filename}, encoding={self.encoding}, gzipped={self.gzipped})"

    def read_file(self):
        self.logger.info(f"Reading {self.filename} as a text file.")
        count_rows = 0

        if self.gzipped:
            textfile = gzip.open(self.file_path, "rt", encoding=self.encoding)
        else:
            textfile = open(self.file_path, encoding=self.encoding)

        with textfile:
            for row in textfile:
                count_rows += 1

                text = row
                if self.strip_lines:
                    text = row.strip()

                yield AnnotatedText(
                    text,
                    location=[
                        self.filename,
                        self.__class__.__name__,
                        f"row={count_rows}",
                        "text",
                    ],
                )

        self.logger.info(
            f"Generated {count_rows} AnnotatedText objects from {count_rows} rows in {self.filename}."
        )

    def write_file(
        self, texts: list[AnnotatedText], file: TextIOBase, **kwargs
    ) -> None:
        for text in texts:
            file.write(text.text + "\n")
