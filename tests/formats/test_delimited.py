import csv
import gzip
import io

import pytest

from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance, Annotator
from renci_ner.formats import (
    DelimitedFile,
    Row,
    reader_for_file,
    writer_for_format,
)

NER = AnnotationProvenance("NER", "http://ner.example/", "1")
LINK = AnnotationProvenance("Linker", "http://linker.example/", "2")

CSV = "id,name,description\n1,Brain,The brain is an organ.\n2,,\n3,Heart,\n"


def texts_of(items):
    """The AnnotatedTexts among a reader's items, leaving out its Rows."""
    return [t for t in items if isinstance(t, AnnotatedText)]


def test_read_cells_with_locations(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    items = list(reader_for_file(str(path), columns_exclude=["id"]).read())
    assert [(t.text, t.location[1:]) for t in texts_of(items)] == [
        ("Brain", ["row=1", "name"]),
        ("The brain is an organ.", ["row=1", "description"]),
        ("Heart", ["row=3", "name"]),
    ]
    # Every row comes first as a Row with all its cells, excluded and blank ones too.
    rows = [r for r in items if isinstance(r, Row)]
    assert [r.location for r in rows] == [[str(path), f"row={n}"] for n in (1, 2, 3)]
    assert rows[1].cells == {"id": "2", "name": "", "description": ""}
    assert isinstance(items[0], Row) and items[1].location[:2] == rows[0].location

    only = list(DelimitedFile(str(path), columns_include=["description"]).read())
    assert [t.text for t in texts_of(only)] == ["The brain is an organ."]


def test_tsv_by_suffix(tmp_path):
    path = tmp_path / "in.tsv"
    path.write_text("a\tb\nx y\tz\n")
    reader = reader_for_file(str(path))
    assert reader.dialect == "excel-tab"
    assert [t.text for t in texts_of(reader.read())] == ["x y", "z"]


def test_write_rows_per_annotation(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    items = list(reader_for_file(str(path), columns_exclude=["id"]).read())
    texts = texts_of(items)

    # Annotate "Brain" with two candidates, the description with one, and "Heart" with none.
    def link(text, start, curie):
        ner = Annotation(
            text[start : start + 5],
            "I1",
            "",
            "biolink:AnatomicalEntity",
            start,
            start + 5,
            NER,
        )
        return Annotation(
            ner.text,
            curie,
            ner.text.lower(),
            "biolink:AnatomicalEntity",
            start,
            start + 5,
            LINK,
            based_on=[ner],
        )

    texts[0].annotations = [
        link("Brain", 0, "UBERON:0000955"),
        link("Brain", 0, "UMLS:C0006104"),
    ]
    texts[1].annotations = [link("The brain is an organ.", 4, "UBERON:0000955")]

    out = io.StringIO()
    writer_for_format("csv").write(items, out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))

    # Every input column comes back, including the excluded id, in the input's order.
    assert list(rows[0].keys())[:5] == [
        "id",
        "name",
        "description",
        "annotation_row",
        "annotation_column",
    ]
    assert [
        (r["id"], r["annotation_row"], r["annotation_column"], r["annotation_id"])
        for r in rows
    ] == [
        ("1", "1", "name", "UBERON:0000955"),
        ("1", "1", "name", "UMLS:C0006104"),
        ("1", "1", "description", "UBERON:0000955"),
        # Row 2 has nothing to annotate, but still gets a row.
        ("2", "2", "", ""),
        ("3", "3", "name", ""),
    ]
    # Input cells of the row are repeated on every output row for that row.
    assert rows[2]["name"] == "Brain"
    assert rows[2]["description"] == "The brain is an organ."
    assert (rows[2]["annotation_start"], rows[2]["annotation_end"]) == ("4", "9")
    assert rows[2]["annotation_label"] == "brain"
    assert rows[2]["annotation_prov"] == "NER 1 > Linker 2"
    # "Heart" is on its own row and had no annotations.
    assert rows[4]["name"] == "Heart" and rows[4]["annotation_text"] == ""


def test_write_texts_without_locations():
    out = io.StringIO()
    writer_for_format("tsv").write([AnnotatedText("a"), AnnotatedText("b")], out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue()), dialect="excel-tab"))
    assert [r["text"] for r in rows] == ["a", "b"]
    assert [r["annotation_row"] for r in rows] == ["", ""]


def test_write_texts_from_txt_input(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("brain\n\nheart\n")
    out = io.StringIO()
    writer_for_format("csv").write(reader_for_file(str(path)).read(), out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))
    assert [(r["text"], r["annotation_row"]) for r in rows] == [
        ("brain", "1"),
        ("heart", "3"),
    ]


def test_txt_and_jsonl_writers_skip_rows(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    items = list(reader_for_file(str(path), columns_exclude=["id"]).read())
    out = io.StringIO()
    writer_for_format("txt").write(items, out)
    assert out.getvalue() == "Brain\nThe brain is an organ.\nHeart\n"
    out = io.StringIO()
    writer_for_format("jsonl").write(items, out)
    assert len(out.getvalue().splitlines()) == 3


def test_rows_cannot_be_annotated():
    class Echo(Annotator):
        def _annotate(self, text, props):
            return AnnotatedText(text, [])

    with pytest.raises(TypeError, match="Row"):
        Echo().annotate(Row(["in.csv", "row=1"], {"text": "brain"}))


def test_unknown_include_column_is_an_error(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    with pytest.raises(ValueError, match="nope"):
        list(DelimitedFile(str(path), columns_include=["nope"]).read())


def test_rows_from_different_files_stay_separate(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("text\nfrom a\n")
    b.write_text("text\nfrom b\n")
    texts = list(reader_for_file(str(a)).read()) + list(reader_for_file(str(b)).read())
    assert [t.location[1] for t in texts_of(texts)] == ["row=1", "row=1"]

    out = io.StringIO()
    writer_for_format("csv").write(texts, out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))
    assert [r["text"] for r in rows] == ["from a", "from b"]


def test_delimiters_quotes_and_newlines_round_trip(tmp_path):
    tricky = 'He said "brain, not heart"\nand left.'
    path = tmp_path / "in.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([["text"], [tricky]])

    row, text = reader_for_file(str(path)).read()
    assert text.text == tricky

    out = io.StringIO()
    writer_for_format("csv").write([row, text], out)
    (row,) = csv.DictReader(io.StringIO(out.getvalue()))
    assert row["text"] == tricky


def test_gzipped_csv(tmp_path):
    path = tmp_path / "in.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("id,text\n1,brain\n")
    reader = reader_for_file(str(path), columns_exclude=["id"])
    assert isinstance(reader, DelimitedFile)
    assert [t.text for t in texts_of(reader.read())] == ["brain"]
