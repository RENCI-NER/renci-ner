import gzip
import io
import json

import pytest

from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance
from renci_ner.formats import (
    JsonlFile,
    TextFile,
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
    path.write_text("The brain.\n\nThe heart.\n")

    texts = list(TextFile(str(path)).read())
    assert [t.text for t in texts] == ["The brain.", "", "The heart."]
    assert texts[2].location == [str(path), "row=3", "text"]

    out = io.StringIO()
    writer_for_format("txt").write(texts, out)
    assert out.getvalue() == path.read_text()


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


def test_gzip_by_suffix(tmp_path):
    path = tmp_path / "in.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("gzipped brain\n")
    reader = reader_for_file(str(path))
    assert isinstance(reader, TextFile)
    assert [t.text for t in reader.read()] == ["gzipped brain"]


def test_unknown_formats():
    with pytest.raises(ValueError):
        reader_for_file("data.xlsx")
    with pytest.raises(ValueError):
        writer_for_format("xlsx")
