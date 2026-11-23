import csv
import gzip
import io

import pytest

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    Annotator,
    Record,
    Text,
)
from renci_ner.formats import DelimitedFile, reader_for_file, writer_for_format

NER = AnnotationProvenance("NER", "http://ner.example/", "1")
LINK = AnnotationProvenance("Linker", "http://linker.example/", "2")

CSV = "id,name,description\n1,Brain,The brain is an organ.\n2,,\n3,Heart,\n"


def texts_of(records):
    """Every text in the records, in order."""
    return [t for r in records for t in r.texts.values()]


def rows_of(output, dialect="excel"):
    return list(csv.DictReader(io.StringIO(output), dialect=dialect))


def link(text, start, curie):
    """A linked annotation of the five characters at `start`, based on an NER one."""
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


class Canned(Annotator):
    """Annotates "Brain" with two candidates, the description with one, else nothing."""

    ANNOTATIONS = {
        "Brain": [
            link("Brain", 0, "UBERON:0000955"),
            link("Brain", 0, "UMLS:C0006104"),
        ],
        "The brain is an organ.": [link("The brain is an organ.", 4, "UBERON:0000955")],
    }

    def _annotate(self, text, props):
        return AnnotatedText(text, self.ANNOTATIONS.get(text, []))


def test_read_records(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    records = list(reader_for_file(str(path), columns_exclude=["id"]).read())

    # One record per data row, with every cell, excluded and blank ones too.
    assert [r.location for r in records] == [[str(path), f"row={n}"] for n in (1, 2, 3)]
    assert records[1].cells == {"id": "2", "name": "", "description": ""}
    # Texts only for non-empty cells in the chosen columns; row 2 has none.
    assert [list(r.texts) for r in records] == [["name", "description"], [], ["name"]]
    assert [(t.text, t.location[1:]) for t in texts_of(records)] == [
        ("Brain", ["row=1", "name"]),
        ("The brain is an organ.", ["row=1", "description"]),
        ("Heart", ["row=3", "name"]),
    ]
    assert all(isinstance(t, Text) for t in texts_of(records))

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
    records = reader_for_file(str(path), columns_exclude=["id"]).read()
    out = io.StringIO()
    writer_for_format("csv").write((r.annotate(Canned()) for r in records), out)
    rows = rows_of(out.getvalue())

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


def test_write_bare_texts():
    out = io.StringIO()
    writer_for_format("tsv").write([AnnotatedText("a"), Text("b")], out)
    rows = rows_of(out.getvalue(), dialect="excel-tab")
    assert [r["text"] for r in rows] == ["a", "b"]
    assert [r["annotation_row"] for r in rows] == ["", ""]


def test_write_records_from_txt_input(tmp_path):
    path = tmp_path / "in.txt"
    path.write_text("brain\n\nheart\n")
    out = io.StringIO()
    writer_for_format("csv").write(reader_for_file(str(path)).read(), out)
    rows = rows_of(out.getvalue())
    assert [(r["text"], r["annotation_row"], r["annotation_column"]) for r in rows] == [
        ("brain", "1", "text"),
        ("heart", "3", "text"),
    ]


def test_txt_and_jsonl_writers_take_csv_records(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    records = list(reader_for_file(str(path), columns_exclude=["id"]).read())
    out = io.StringIO()
    writer_for_format("txt").write(records, out)
    assert out.getvalue() == "Brain\nThe brain is an organ.\nHeart\n"
    out = io.StringIO()
    writer_for_format("jsonl").write(records, out)
    # One line per record, including row 2, which has nothing to annotate.
    assert len(out.getvalue().splitlines()) == 3


def test_csv_through_jsonl_keeps_every_column(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    records = reader_for_file(str(path), columns_exclude=["id"]).read()
    jsonl = tmp_path / "mid.jsonl"
    with open(jsonl, "w") as f:
        writer_for_format("jsonl").write((r.annotate(Canned()) for r in records), f)

    out = io.StringIO()
    writer_for_format("csv").write(reader_for_file(str(jsonl)).read(), out)
    rows = rows_of(out.getvalue())
    # The id column, which was never annotated, and the empty row 2 both survive, and
    # the row numbers still point at in.csv.
    assert [(r["id"], r["annotation_row"], r["annotation_column"]) for r in rows] == [
        ("1", "1", "name"),
        ("1", "1", "description"),
        ("2", "2", ""),
        ("3", "3", "name"),
    ]


def test_unknown_include_column_is_an_error(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    with pytest.raises(ValueError, match="nope"):
        list(DelimitedFile(str(path), columns_include=["nope"]).read())


def test_rows_from_different_files_stay_separate(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("text\nfrom a\n")
    b.write_text("text\nfrom b\n")
    records = list(reader_for_file(str(a)).read()) + list(
        reader_for_file(str(b)).read()
    )
    assert [r.location[1] for r in records] == ["row=1", "row=1"]

    out = io.StringIO()
    writer_for_format("csv").write(records, out)
    assert [r["text"] for r in rows_of(out.getvalue())] == ["from a", "from b"]


def test_delimiters_quotes_and_newlines_round_trip(tmp_path):
    tricky = 'He said "brain, not heart"\nand left.'
    path = tmp_path / "in.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([["text"], [tricky]])

    (record,) = reader_for_file(str(path)).read()
    assert record.texts["text"].text == tricky

    out = io.StringIO()
    writer_for_format("csv").write([record], out)
    (row,) = rows_of(out.getvalue())
    assert row["text"] == tricky


def test_gzipped_csv(tmp_path):
    path = tmp_path / "in.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write("id,text\n1,brain\n")
    reader = reader_for_file(str(path), columns_exclude=["id"])
    assert isinstance(reader, DelimitedFile)
    assert [t.text for t in texts_of(reader.read())] == ["brain"]


def test_extra_cells_are_an_error(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("id,text\n1,brain,extra note\n")
    with pytest.raises(ValueError, match="row 1.*extra note"):
        list(reader_for_file(str(path)).read())


def test_blank_extra_cells_are_ignored(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("id,text\n1,brain,,\n")
    (record,) = reader_for_file(str(path), columns_exclude=["id"]).read()
    assert record.cells == {"id": "1", "text": "brain"}
    assert record.texts["text"].text == "brain"


def test_duplicate_header_names_are_an_error(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("id,note,note\n1,a,b\n")
    with pytest.raises(ValueError, match="note"):
        list(reader_for_file(str(path)).read())


def test_include_and_exclude_together_is_an_error():
    with pytest.raises(ValueError, match="columns_include"):
        DelimitedFile("in.csv", columns_include=["name"], columns_exclude=["id"])


def test_unknown_exclude_column_is_an_error(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    with pytest.raises(ValueError, match="ID"):
        list(DelimitedFile(str(path), columns_exclude=["ID"]).read())


def test_repeated_include_column_is_read_once(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text(CSV)
    reader = DelimitedFile(str(path), columns_include=["description", "description"])
    assert [t.text for t in texts_of(reader.read())] == ["The brain is an organ."]


def test_rewriting_annotated_output_renames_clashing_columns(tmp_path):
    first = tmp_path / "first.csv"
    first.write_text(CSV)
    out = io.StringIO()
    writer_for_format("csv").write(
        reader_for_file(str(first), columns_include=["name"]).read(), out
    )

    # Read that output back in and write it again, annotating its name column.
    second = tmp_path / "second.csv"
    second.write_text(out.getvalue())
    out = io.StringIO()
    writer_for_format("csv").write(
        reader_for_file(str(second), columns_include=["name"]).read(), out
    )
    header = next(csv.reader(io.StringIO(out.getvalue())))
    assert len(header) == len(set(header))
    assert header[:5] == [
        "id",
        "name",
        "description",
        "input_annotation_row",
        "input_annotation_column",
    ]
    rows = rows_of(out.getvalue())
    # The first run's values stay in their renamed columns, and this run's row
    # numbers count the rows of second.csv.
    assert [r["input_annotation_row"] for r in rows] == ["1", "2", "3"]
    assert [r["annotation_row"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["input_annotation_column"] == "name"
    assert rows[0]["annotation_column"] == "name"


def test_annotation_column_names_the_renamed_column():
    record = Record(
        ["in.csv", "row=1"],
        {"annotation_text": "brain"},
        {"annotation_text": Text("brain", ["in.csv", "row=1", "annotation_text"])},
    )
    out = io.StringIO()
    writer_for_format("csv").write([record], out)
    (row,) = rows_of(out.getvalue())
    assert row["input_annotation_text"] == "brain"
    assert row["annotation_column"] == "input_annotation_text"


def test_writers_refuse_other_items():
    with pytest.raises(TypeError, match="Records"):
        writer_for_format("csv").write(["brain"], io.StringIO())
