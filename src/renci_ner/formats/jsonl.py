#
# jsonl.py - Supports reading and writing AnnotatedText objects in JSONL format.
#
import gzip
import json
import logging
from io import TextIOBase
from pathlib import Path

from renci_ner.core import AnnotatedText
from renci_ner.formats import Format


class JsonlFile(Format):
    def __init__(self, filename: str, gzipped: bool = None):
        file_path = Path(filename)
        suffixes = list(file_path.suffixes)

        if len(suffixes) > 0 and suffixes[-1].lower() == ".gz":
            suffixes.pop()
            self.gzipped = True
        else:
            self.gzipped = False

        if gzipped is not None:
            self.gzipped = gzipped

        self.file_path = file_path
        self.filename = filename
        self.logger = logging.getLogger(__name__)

    def read_file(self, **kwargs):
        self.logger.info(f"Reading {self.filename} as a JSONL file.")
        count_rows = 0

        if self.gzipped:
            f = gzip.open(self.file_path, "rt", encoding="utf-8")
        else:
            f = open(self.file_path, encoding="utf-8")

        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                count_rows += 1
                data = json.loads(line)
                yield AnnotatedText(
                    data.get("text", ""),
                    location=data.get(
                        "location",
                        [
                            self.filename,
                            self.__class__.__name__,
                            f"row={count_rows}",
                            "text",
                        ],
                    ),
                )

        self.logger.info(
            f"Generated {count_rows} AnnotatedText objects from {self.filename}."
        )

    def write_file(
        self, texts: list[AnnotatedText], file: TextIOBase, **kwargs
    ) -> None:
        for annotated_text in texts:
            file.write(json.dumps(annotated_text.to_dict()) + "\n")
