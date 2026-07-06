from abc import ABC, abstractmethod
from collections.abc import Generator
from io import TextIOBase
from pathlib import Path

from renci_ner.core import AnnotatedText


def detect_gzip(filename: str, gzipped: bool = None) -> bool:
    """Whether a file is gzipped: explicit override wins, else True if the name ends in .gz."""
    if gzipped is not None:
        return gzipped
    return Path(filename).suffix.lower() == ".gz"


class Format(ABC):
    """Base class for reading/writing AnnotatedText objects in a file format."""

    @abstractmethod
    def read_file(self, **kwargs) -> Generator[AnnotatedText]:
        """Read the file, yielding AnnotatedText objects."""
        ...

    @abstractmethod
    def write_file(
        self, texts: list[AnnotatedText], file: TextIOBase, **kwargs
    ) -> None:
        """Write AnnotatedText objects to a file handle."""
        ...


def reader_for_file(
    filename: str,
    columns_include: list[str] = None,
    columns_exclude: list[str] = None,
    gzipped: bool = False,
) -> Format:
    """Detect format from filename suffix and return the appropriate Format instance.

    :param filename: Path to the input file.
    :param columns_include: Column names to include (for delimited formats).
    :param columns_exclude: Column names to exclude (for delimited formats).
    :param gzipped: Whether the file is gzipped.
    :return: A Format subclass instance for reading the file.
    """
    from renci_ner.formats.csv import DelimitedFile
    from renci_ner.formats.jsonl import JsonlFile
    from renci_ner.formats.txt import TextFile

    filepath = Path(filename)
    file_gzipped = detect_gzip(filename, gzipped or None)

    # Strip .gz suffix for format detection.
    suffixes = [s.lower() for s in filepath.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    last_suffix = suffixes[-1] if suffixes else ""

    if last_suffix == ".csv":
        return DelimitedFile(
            filename,
            columns_include=columns_include,
            columns_exclude=columns_exclude,
            gzipped=file_gzipped,
            dialect="excel",
        )
    elif last_suffix == ".tsv":
        return DelimitedFile(
            filename,
            columns_include=columns_include,
            columns_exclude=columns_exclude,
            gzipped=file_gzipped,
            dialect="excel-tab",
        )
    elif last_suffix == ".txt":
        return TextFile(filename, gzipped=file_gzipped)
    elif last_suffix == ".jsonl":
        return JsonlFile(filename, gzipped=file_gzipped)
    else:
        raise ValueError(
            f"Could not determine a file type for {filename} based on the suffixes {filepath.suffixes}."
        )


def writer_for_format(output_format: str, output_filename: str = None) -> Format:
    """Return a Format instance suitable for writing in the given format.

    :param output_format: The output format string ("csv", "tsv", or "jsonl").
    :param output_filename: The output filename (used for DelimitedFile metadata).
    :return: A Format subclass instance for writing.
    """
    from renci_ner.formats.csv import DelimitedFile
    from renci_ner.formats.jsonl import JsonlFile

    match output_format.lower():
        case "csv":
            return DelimitedFile(output_filename or "-")
        case "tsv":
            return DelimitedFile(output_filename or "-", dialect="excel-tab")
        case "jsonl":
            return JsonlFile(output_filename or "-")
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")
