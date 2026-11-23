"""JSON Lines: one Record.to_dict() per line."""

import json
from collections.abc import Iterable, Iterator

from renci_ner.core import AnnotatedText, Record, Text
from renci_ner.formats.base import Format, as_records


class JsonlFile(Format):
    """
    One Record.to_dict() per line, so every cell of a CSV row survives, annotated or
    not. Reading takes each record's location, cells and the text of each field to
    annotate (annotations are not read back yet, see #25), so annotated output can be
    used as input again. A line that is a single text, {"text": ..., "location": ...},
    such as an AnnotatedText.to_dict() from older output, is read as a one-text Record;
    its text keeps the line's own location, so it still points at the original input.
    """

    def read(self) -> Iterator[Record[Text]]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{self.filename}, row {row}"
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {where}: {e}") from e
                location = [self.filename, f"row={row}"]
                if isinstance(data, dict) and data.get("@type") == "renci_ner:Record":
                    yield self._record(data, location, where)
                elif isinstance(data, dict) and isinstance(data.get("text"), str):
                    text = Text(
                        data["text"],
                        location=data.get("location") or [*location, "text"],
                    )
                    yield Record(location, {"text": text.text}, {"text": text})
                else:
                    raise ValueError(
                        f'Expected a Record or a JSON object with a string "text" at '
                        f"{where}, got: {line.strip()[:100]}"
                    )

    @staticmethod
    def _record(data: dict, location: list[str], where: str) -> Record[Text]:
        """Rebuild a Record[Text] from Record.to_dict(), checking its shape."""
        cells, texts = data.get("cells"), data.get("texts")
        if not (
            isinstance(cells, dict)
            and isinstance(texts, dict)
            and all(
                isinstance(t, dict) and isinstance(t.get("text"), str)
                for t in texts.values()
            )
        ):
            raise ValueError(
                f'Expected a Record with "cells" and "texts" (each with a string '
                f'"text") at {where}'
            )
        location = data.get("location") or location
        return Record(
            location,
            cells,
            {
                name: Text(t["text"], location=t.get("location") or [*location, name])
                for name, t in texts.items()
            },
        )

    def write(self, records: Iterable[Record | Text | AnnotatedText], file) -> None:
        for record in as_records(records):
            file.write(json.dumps(record.to_dict()) + "\n")
