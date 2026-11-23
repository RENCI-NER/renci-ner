"""JSON Lines: one AnnotatedText.to_dict() per line."""

import json
from collections.abc import Iterable, Iterator

from renci_ner.core import AnnotatedText
from renci_ner.formats.base import Format, Row


class JsonlFile(Format):
    """
    One AnnotatedText.to_dict() per line. Reading takes only `text` and `location`
    from each line (annotations are not read back yet), so it can be used as input.
    """

    def read(self) -> Iterator[AnnotatedText]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{self.filename}, row {row}"
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {where}: {e}") from e
                if not isinstance(data, dict) or not isinstance(data.get("text"), str):
                    raise ValueError(
                        f'Expected a JSON object with a string "text" at {where}, '
                        f"got: {line.strip()[:100]}"
                    )
                yield AnnotatedText(
                    data["text"],
                    location=data.get("location")
                    or [self.filename, f"row={row}", "text"],
                )

    def write(self, texts: Iterable[AnnotatedText | Row], file) -> None:
        for text in texts:
            if isinstance(text, AnnotatedText):
                file.write(json.dumps(text.to_dict()) + "\n")
