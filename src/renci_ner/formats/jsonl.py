"""JSON Lines: one Record.to_dict() per line."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import replace

from renci_ner.core import AnnotatedText, Record, Text, from_dict
from renci_ner.formats.base import Format, as_records


class JsonlFile(Format):
    """
    One Record.to_dict() per line, so every cell of a CSV row survives, annotated or
    not. A line that is a single text, such as {"text": ..., "location": ...} or an
    AnnotatedText.to_dict() from older output, is read as a one-text Record; a record
    or text keeps the location it was written with, so it still points at the original
    input, and one without gets [filename, "row=N"].

    By default, reading yields Record[Text] for use as input: annotations in the file
    are dropped, since annotate() would replace them anyway. With annotated=True it
    yields Record[AnnotatedText] with every annotation read back (via
    renci_ner.core.from_dict()), for comparing runs; a text that was never annotated
    becomes an AnnotatedText with no annotations.
    """

    def __init__(self, filename: str | None = None, annotated: bool = False):
        super().__init__(filename)
        self.annotated = annotated

    def read(self) -> Iterator[Record]:
        with self._open() as f:
            for row, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{self.filename}, row {row}"
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {where}: {e}") from e
                try:
                    yield self._record(data, [self.filename, f"row={row}"])
                except ValueError as e:
                    raise ValueError(
                        f"Expected a Record or a text at {where} ({e}), got: "
                        f"{line.strip()[:100]}"
                    ) from e

    def _record(self, data, location: list[str]) -> Record:
        """The Record a line holds, with default locations filled in."""
        # As input, only each text's text and location matter, so annotations are
        # dropped before parsing: a malformed one shouldn't stop a file being re-used.
        prepare = (
            _typed if self.annotated else lambda t: _without_annotations(_typed(t))
        )
        if isinstance(data, dict) and data.get("@type") == "renci_ner:Record":
            texts = data.get("texts")
            if isinstance(texts, dict):
                data = data | {"texts": {k: prepare(t) for k, t in texts.items()}}
        record = from_dict(prepare(data))
        if isinstance(record, Text | AnnotatedText):
            record = Record(location, {"text": record.text}, {"text": record})
            # A single text keeps its own location; the record is this line.
            location = record.location
        elif isinstance(record, Record):
            location = record.location or location
        else:
            raise ValueError(f"a {data['@type']} is neither a Record nor a text")
        return Record(
            location,
            record.cells,
            {
                name: self._convert(
                    replace(text, location=text.location or [*location, name])
                )
                for name, text in record.texts.items()
            },
        )

    def _convert(self, text: Text | AnnotatedText) -> Text | AnnotatedText:
        """A Text for input, or an AnnotatedText when reading annotations back."""
        if self.annotated:
            if isinstance(text, AnnotatedText):
                return text
            return AnnotatedText(text.text, [], text.location)
        return Text(text.text, text.location)

    def write(self, records: Iterable[Record | Text | AnnotatedText], file) -> None:
        for record in as_records(records):
            file.write(json.dumps(record.to_dict()) + "\n")


def _typed(data):
    """A hand-written {"text": ...} with no @type is a Text."""
    if (
        isinstance(data, dict)
        and "@type" not in data
        and isinstance(data.get("text"), str)
    ):
        return {"@type": "renci_ner:Text"} | data
    return data


def _without_annotations(data):
    """An AnnotatedText.to_dict() as a Text.to_dict(), leaving its annotations unread."""
    if isinstance(data, dict) and data.get("@type") == "renci_ner:AnnotatedText":
        return {
            "@type": "renci_ner:Text",
            "text": data.get("text"),
            "location": data.get("location"),
        }
    return data
