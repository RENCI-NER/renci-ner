"""
What every format shares: opening (possibly gzipped) text files, picking a format by
suffix, the Row record a CSV/TSV reader yields alongside its texts, and the Format
interface.
"""

import gzip
from collections.abc import Iterable, Iterator
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
