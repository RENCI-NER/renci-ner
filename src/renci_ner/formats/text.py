"""Plain text: one text per line."""

from collections.abc import Iterable, Iterator

from renci_ner.core import AnnotatedText, Text
from renci_ner.formats.base import Format, Row


class TextFile(Format):
    """
    One text per line. Reading skips blank lines (so they aren't sent to the annotators),
    but row numbers still count them. Writing emits only the text, one per line.
    """

    def read(self) -> Iterator[Text]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                text = line.rstrip("\n")
                if not text.strip():
                    continue
                yield Text(text, location=[self.filename, f"row={row}", "text"])

    def write(self, texts: Iterable[AnnotatedText | Row], file) -> None:
        for text in texts:
            if not isinstance(text, Text | AnnotatedText):
                continue
            # A line break would split the text over several lines, so it would read
            # back as several texts and shift every later row. Refuse rather than
            # silently rewrite the text; use JSONL for texts with line breaks.
            # ("\r" counts because reading in text mode treats it as a line end.)
            if "\n" in text.text or "\r" in text.text:
                raise ValueError(
                    f"Can't write a text containing a line break as TXT; use JSONL "
                    f"instead (location {text.location}): {text.text[:100]!r}"
                )
            file.write(text.text + "\n")
