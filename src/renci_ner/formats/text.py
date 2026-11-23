"""Plain text: one text per line."""

from collections.abc import Iterable, Iterator

from renci_ner.core import AnnotatedText, Record, Text
from renci_ner.formats.base import Format, as_records


class TextFile(Format):
    """
    One text per line. Reading yields a Record per line, with the line as both its only
    cell and its only text, both called "text"; blank lines are skipped (so they aren't
    sent to the annotators), but row numbers still count them. Writing emits only the
    texts, each on its own line, so a CSV record with two annotated cells gives two.
    """

    def read(self) -> Iterator[Record[Text]]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                text = line.rstrip("\n")
                if not text.strip():
                    continue
                location = [self.filename, f"row={row}"]
                yield Record(
                    location,
                    {"text": text},
                    {"text": Text(text, location=[*location, "text"])},
                )

    def write(self, records: Iterable[Record | Text | AnnotatedText], file) -> None:
        for record in as_records(records):
            for text in record.texts.values():
                # A line break would split the text over several lines, so it would
                # read back as several texts and shift every later row. Refuse rather
                # than silently rewrite the text; use JSONL for texts with line breaks.
                # ("\r" counts because reading in text mode treats it as a line end.)
                if "\n" in text.text or "\r" in text.text:
                    raise ValueError(
                        f"Can't write a text containing a line break as TXT; use JSONL "
                        f"instead (location {text.location}): {text.text[:100]!r}"
                    )
                file.write(text.text + "\n")
