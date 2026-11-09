import gzip
import io
import json
import re

import pytest

from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance
from renci_ner.formats import (
    JsonlFile,
    TextFile,
    open_text,
    reader_for_file,
    writer_for_format,
)

PROV = AnnotationProvenance("Test", "http://example.com", "0.1")


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
    texts = list(TextFile(str(path)).read())
    assert [t.text for t in texts] == ["The brain.", "The heart."]
    assert texts[1].location == [str(path), "row=3", "text"]

    out = io.StringIO()
    writer_for_format("txt").write(texts, out)
    assert out.getvalue() == "The brain.\nThe heart.\n"


@pytest.mark.parametrize("text", ["Line one\nLine two", "Line one\rLine two"])
def test_txt_refuses_line_breaks(text):
    with pytest.raises(ValueError, match="line break"):
        writer_for_format("txt").write([AnnotatedText(text)], io.StringIO())


def test_jsonl_write_and_read(tmp_path):
    out = io.StringIO()
    JsonlFile("-").write([annotated()], out)
    (line,) = out.getvalue().splitlines()
    data = json.loads(line)
    assert data["@type"] == "renci_ner:AnnotatedText"
    assert data["annotations"][0]["id"] == "UBERON:0000955"

    path = tmp_path / "out.jsonl"
    path.write_text(out.getvalue() + "\n" + json.dumps({"text": "bare"}) + "\n")
    texts = list(reader_for_file(str(path)).read())
    assert [t.text for t in texts] == ["The brain.", "bare"]
    assert texts[0].location == ["in.txt", "row=1", "text"]
    assert texts[1].location == [str(path), "row=3", "text"]


@pytest.mark.parametrize(
    "line", ['["a"]', '"text"', '{"txt": "a"}', '{"text": 1}', "{not json"]
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
    assert [t.text for t in reader.read()] == ["gzipped brain"]


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
    assert [t.text for t in reader_for_file(str(txt)).read()] == ["The brain."]

    jsonl = tmp_path / "bom.jsonl"
    jsonl.write_bytes(("﻿" + json.dumps({"text": "The brain."}) + "\n").encode())
    assert [t.text for t in reader_for_file(str(jsonl)).read()] == ["The brain."]

    # Writing never adds a byte-order mark.
    with open_text(str(txt), "w") as f:
        f.write("The brain.\n")
    assert txt.read_bytes() == b"The brain.\n"


def test_unknown_formats():
    with pytest.raises(ValueError):
        reader_for_file("data.xlsx")
    with pytest.raises(ValueError):
        writer_for_format("xlsx")
