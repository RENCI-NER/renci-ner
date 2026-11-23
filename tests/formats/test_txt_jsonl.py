import gzip
import io
import json
import re

import pytest

from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance, Record, Text
from renci_ner.formats import (
    JsonlFile,
    TextFile,
    open_text,
    reader_for_file,
    writer_for_format,
)

PROV = AnnotationProvenance("Test", "http://example.com", "0.1")


def texts_of(records):
    """Every text in the records, in order."""
    return [t.text for r in records for t in r.texts.values()]


def annotated():
    return AnnotatedText(
        "The brain.",
        [
            Annotation(
                "brain",
                "UBERON:0000955",
                "brain",
                "biolink:AnatomicalEntity",
                4,
                9,
                PROV,
            )
        ],
        location=["in.txt", "row=1", "text"],
    )


def test_txt_round_trip(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("The brain.\n\nThe heart.\n  \n\n")

    # Blank lines are skipped, but still counted in the row numbers.
    records = list(TextFile(str(path)).read())
    assert texts_of(records) == ["The brain.", "The heart."]
    assert records[1].location == [str(path), "row=3"]
    assert records[1].cells == {"text": "The heart."}
    assert records[1].texts == {
        "text": Text("The heart.", [str(path), "row=3", "text"])
    }

    out = io.StringIO()
    writer_for_format("txt").write(records, out)
    assert out.getvalue() == "The brain.\nThe heart.\n"


@pytest.mark.parametrize("text", ["Line one\nLine two", "Line one\rLine two"])
def test_txt_refuses_line_breaks(text):
    with pytest.raises(ValueError, match="line break"):
        writer_for_format("txt").write([AnnotatedText(text)], io.StringIO())


def test_jsonl_write_and_read(tmp_path):
    record = Record(
        ["in.csv", "row=1"], {"id": "1", "text": "The brain."}, {"text": annotated()}
    )
    out = io.StringIO()
    JsonlFile().write([record], out)
    (line,) = out.getvalue().splitlines()
    data = json.loads(line)
    assert data["@type"] == "renci_ner:Record"
    assert data["cells"] == {"id": "1", "text": "The brain."}
    assert data["texts"]["text"]["annotations"][0]["id"] == "UBERON:0000955"

    # Read back alongside a bare text line and an older one-AnnotatedText line.
    path = tmp_path / "out.jsonl"
    old_line = json.dumps(annotated().to_dict())
    path.write_text(
        out.getvalue() + "\n" + json.dumps({"text": "bare"}) + "\n" + old_line + "\n"
    )
    records = list(reader_for_file(str(path)).read())
    assert texts_of(records) == ["The brain.", "bare", "The brain."]
    # A record keeps its own location and cells, so it still points at the input;
    # its texts are Texts again, since annotations are not read back (#25).
    assert records[0].location == ["in.csv", "row=1"]
    assert records[0].cells == {"id": "1", "text": "The brain."}
    assert records[0].texts["text"] == Text("The brain.", ["in.txt", "row=1", "text"])
    # A single-text line becomes a one-text record at its row of the JSONL file;
    # its text keeps the line's own location if it has one.
    assert records[1].location == [str(path), "row=3"]
    assert records[1].texts["text"].location == [str(path), "row=3", "text"]
    assert records[2].texts["text"].location == ["in.txt", "row=1", "text"]


@pytest.mark.parametrize(
    "line",
    [
        '["a"]',
        '"text"',
        '{"txt": "a"}',
        '{"text": 1}',
        "{not json",
        '{"@type": "renci_ner:Record", "cells": {}, "texts": {"a": {"txt": 1}}}',
        '{"@type": "renci_ner:Record", "texts": {}}',
    ],
)
def test_jsonl_bad_lines_name_file_and_row(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "fine"}) + "\n" + line + "\n")
    with pytest.raises(ValueError, match=re.escape(f"{path}, row 2")):
        list(reader_for_file(str(path)).read())


def test_gzip_by_suffix(tmp_path):
    path = tmp_path / "in.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("gzipped brain\n")
    reader = reader_for_file(str(path))
    assert isinstance(reader, TextFile)
    assert texts_of(reader.read()) == ["gzipped brain"]


def test_open_text_modes(tmp_path):
    for name in ("out.txt", "out.txt.gz"):
        path = str(tmp_path / name)
        # "w" and "r" are text modes for .gz files too, as they are for open().
        with open_text(path, "w") as f:
            f.write("brain\n")
        with open_text(path, "r") as f:
            assert f.read() == "brain\n"
    with pytest.raises(ValueError):
        open_text(str(tmp_path / "out.txt"), "rb")


def test_byte_order_mark_is_dropped(tmp_path):
    txt = tmp_path / "bom.txt"
    txt.write_bytes("﻿The brain.\n".encode())
    assert texts_of(reader_for_file(str(txt)).read()) == ["The brain."]

    jsonl = tmp_path / "bom.jsonl"
    jsonl.write_bytes(("﻿" + json.dumps({"text": "The brain."}) + "\n").encode())
    assert texts_of(reader_for_file(str(jsonl)).read()) == ["The brain."]

    # Writing never adds a byte-order mark.
    with open_text(str(txt), "w") as f:
        f.write("The brain.\n")
    assert txt.read_bytes() == b"The brain.\n"


def test_unknown_formats():
    with pytest.raises(ValueError):
        reader_for_file("data.xlsx")
    with pytest.raises(ValueError):
        writer_for_format("xlsx")


def test_writer_cannot_read():
    with pytest.raises(ValueError, match="not given a file"):
        list(writer_for_format("txt").read())


def test_jsonl_record_without_locations(tmp_path):
    # A hand-written record line with no locations gets them from its row of the file.
    path = tmp_path / "in.jsonl"
    line = {
        "@type": "renci_ner:Record",
        "cells": {"id": "7", "name": "brain"},
        "texts": {"name": {"text": "brain"}},
    }
    path.write_text(json.dumps(line) + "\n")
    (record,) = reader_for_file(str(path)).read()
    assert record.location == [str(path), "row=1"]
    assert record.cells == {"id": "7", "name": "brain"}
    assert record.texts == {"name": Text("brain", [str(path), "row=1", "name"])}


def test_jsonl_reads_annotations_back(tmp_path):
    record = Record(
        ["in.csv", "row=1"],
        {"id": "1", "text": "The brain.", "note": "no texts here"},
        {"text": annotated()},
    )
    path = tmp_path / "out.jsonl"
    with open(path, "w") as f:
        JsonlFile().write(
            [
                record,
                Record(
                    ["in.csv", "row=2"],
                    {"text": "Plain."},
                    {"text": Text("Plain.", ["in.csv", "row=2", "text"])},
                ),
            ],
            f,
        )
        # An older one-AnnotatedText line, and a bare one.
        f.write(json.dumps(annotated().to_dict()) + "\n")
        f.write(json.dumps({"text": "bare"}) + "\n")

    first, plain, old, bare = reader_for_file(str(path), annotated=True).read()
    # Everything comes back as it was written, annotations included.
    assert first == record
    assert first.texts["text"].annotations[0].id == "UBERON:0000955"
    # A text that was never annotated comes back as an AnnotatedText with none.
    assert plain.texts["text"] == AnnotatedText(
        "Plain.", [], ["in.csv", "row=2", "text"]
    )
    assert old.texts["text"] == annotated()
    assert bare.texts["text"] == AnnotatedText("bare", [], [str(path), "row=4", "text"])

    # Read as input, the same file gives Texts and drops the annotations.
    as_input = list(reader_for_file(str(path)).read())
    assert all(isinstance(t, Text) for r in as_input for t in r.texts.values())
    assert as_input[0].texts["text"] == Text("The brain.", ["in.txt", "row=1", "text"])


def test_jsonl_bad_annotations_name_file_and_row(tmp_path):
    path = tmp_path / "bad.jsonl"
    bad = annotated().to_dict()
    del bad["annotations"][0]["provenance"]
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=re.escape(f"{path}, row 1") + ".*provenance"):
        list(reader_for_file(str(path), annotated=True).read())

    # As input, the annotations aren't read, so the file can still be used.
    (record,) = reader_for_file(str(path)).read()
    assert record.texts["text"].text == "The brain."
