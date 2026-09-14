import csv
import io

from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance
from renci_ner.formats import DelimitedFile, reader_for_file, writer_for_format

NER = AnnotationProvenance("NER", "http://ner.example/", "1")
LINK = AnnotationProvenance("Linker", "http://linker.example/", "2")

CSV = "id,name,description\n1,Brain,The brain is an organ.\n2,,\n3,Heart,\n"


def test_read_cells_with_locations(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    texts = list(reader_for_file(str(path), columns_exclude=["id"]).read())
    assert [(t.text, t.location[1:]) for t in texts] == [
        ("Brain", ["row=1", "name"]),
        ("The brain is an organ.", ["row=1", "description"]),
        ("Heart", ["row=3", "name"]),
    ]
    only = list(DelimitedFile(str(path), columns_include=["description"]).read())
    assert [t.text for t in only] == ["The brain is an organ."]


def test_tsv_by_suffix(tmp_path):
    path = tmp_path / "in.tsv"
    path.write_text("a\tb\nx y\tz\n")
    reader = reader_for_file(str(path))
    assert reader.dialect == "excel-tab"
    assert [t.text for t in reader.read()] == ["x y", "z"]


def test_write_rows_per_annotation(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    texts = list(reader_for_file(str(path), columns_exclude=["id"]).read())

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
    writer_for_format("csv").write(texts, out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue())))

    assert list(rows[0].keys())[:2] == ["name", "description"]
    assert [(r["annotation_column"], r["annotation_id"]) for r in rows] == [
        ("name", "UBERON:0000955"),
        ("name", "UMLS:C0006104"),
        ("description", "UBERON:0000955"),
        ("name", ""),
    ]
    # Input cells of the row are repeated on every output row for that row.
    assert rows[2]["name"] == "Brain"
    assert rows[2]["description"] == "The brain is an organ."
    assert (rows[2]["annotation_start"], rows[2]["annotation_end"]) == ("4", "9")
    assert rows[2]["annotation_label"] == "brain"
    assert rows[2]["annotation_prov"] == "NER 1 > Linker 2"
    # "Heart" is on its own row and had no annotations.
    assert rows[3]["name"] == "Heart" and rows[3]["annotation_text"] == ""


def test_write_texts_without_locations():
    out = io.StringIO()
    writer_for_format("tsv").write([AnnotatedText("a"), AnnotatedText("b")], out)
    rows = list(csv.DictReader(io.StringIO(out.getvalue()), dialect="excel-tab"))
    assert [r["text"] for r in rows] == ["a", "b"]
