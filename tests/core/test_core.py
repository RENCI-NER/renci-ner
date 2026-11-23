import json

import pytest

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    Annotator,
    NormalizedAnnotation,
    Record,
    Text,
    from_dict,
)


def test_normalized_annotations_biolink_type():
    """
    Check if we constrain biolink_type to be prefixed with `biolink:`.
    """
    provenance = AnnotationProvenance("Test", "http://example.com", "0.1.0")

    # NormalizedAnnotations without a `biolink:` prefix are not allowed!
    with pytest.raises(ValueError):
        normalized_annotation = NormalizedAnnotation(
            provenance=provenance,
            text="brain",
            id="UBERON:0000955",
            label="brain",
            type="AnatomicalEntity",
            biolink_type="AnatomicalEntity",
            start=0,
            end=4,
        )
    normalized_annotation = NormalizedAnnotation(
        provenance=provenance,
        text="brain",
        id="UBERON:0000955",
        label="brain",
        type="AnatomicalEntity",
        biolink_type="biolink:AnatomicalEntity",
        start=0,
        end=4,
    )
    with pytest.raises(ValueError):
        normalized_annotation.biolink_type = "AnatomicalEntity"


class StubLinker(Annotator):
    """Returns `n` NormalizedAnnotations covering the whole text, like NameRes/SAPBERT."""

    def __init__(self, n=1):
        self.n = n

    @property
    def provenance(self):
        return AnnotationProvenance("StubLinker", "http://stub.example/", "1")

    def annotate(self, text, props=None):
        return AnnotatedText(
            text,
            [
                NormalizedAnnotation(
                    text=text,
                    id=f"STUB:{i}",
                    label=text,
                    type="biolink:NamedThing",
                    biolink_type="biolink:NamedThing",
                    start=0,
                    end=len(text),
                    provenance=self.provenance,
                )
                for i in range(self.n)
            ],
        )


def ner_text():
    ner = AnnotationProvenance("StubNER", "http://ner.example/", "1")
    return AnnotatedText(
        "The brain is part of the nervous system.",
        [
            Annotation("brain", "I1", "", "biolink:AnatomicalEntity", 4, 9, ner),
            Annotation(
                "nervous system", "I2", "", "biolink:AnatomicalEntity", 25, 39, ner
            ),
        ],
        location=["file.txt", "row=1"],
    )


def test_reannotate_offsets_provenance_and_based_on():
    source = ner_text()
    result = source.reannotate(StubLinker(n=2))

    assert result.location == source.location
    assert len(result.annotations) == 4
    for ann in result.annotations:
        # Offsets are relative to the full text again.
        assert source.text[ann.start : ann.end] == ann.text
        # The linker's own provenance is kept, not overwritten by reannotate().
        assert [p.name for p in ann.provenances] == ["StubNER", "StubLinker"]

    first, second = result.annotations[0], result.annotations[1]
    assert first.based_on == second.based_on
    # ... but they must not share the same list object.
    first.based_on.append(first)
    assert len(second.based_on) == 1


class StubPipeline(Annotator):
    """Finds the last word of the text and links it with StubLinker: a nested pipeline."""

    def annotate(self, text, props=None):
        ner = AnnotationProvenance("StubInnerNER", "http://inner.example/", "1")
        start = text.rfind(" ") + 1
        word = text[start:]
        inner = AnnotatedText(
            text, [Annotation(word, "W1", "", "", start, len(text), ner)]
        )
        return inner.reannotate(StubLinker(n=2))


def test_reannotate_nested_pipeline_offsets():
    source = ner_text()
    result = source.reannotate(StubPipeline())

    # "brain" has no space, so the whole thing is the last word; "nervous system" -> "system".
    assert [ann.text for ann in result.annotations] == [
        "brain",
        "brain",
        "system",
        "system",
    ]
    for ann in result.annotations:
        assert [p.name for p in ann.provenances] == [
            "StubNER",
            "StubInnerNER",
            "StubLinker",
        ]
        # Every annotation in the chain, including the ones the nested pipeline
        # recorded, must point at its own text in the full text.
        for step in [*ann.based_on, ann]:
            assert source.text[step.start : step.end] == step.text

    # The nested pipeline shares its inner annotation between both results; it must
    # only have been shifted once.
    assert result.annotations[2].based_on[1].start == 33


def test_reannotate_no_results_keeps_annotation():
    source = ner_text()
    result = source.reannotate(StubLinker(n=0))
    assert result.annotations == source.annotations


def test_from_annotation_copies_props():
    provenance = AnnotationProvenance("Test", "http://example.com", "0.1.0")
    original = Annotation(
        "brain", "I1", "", "biolink:AnatomicalEntity", 4, 9, provenance
    )
    original.props["score"] = 1
    normalized = NormalizedAnnotation.from_annotation(
        original,
        provenance,
        curie="UBERON:0000955",
        biolink_type="biolink:AnatomicalEntity",
    )
    normalized.props["types"] = ["biolink:AnatomicalEntity"]
    assert normalized.based_on == [original]
    assert original.props == {"score": 1}


def test_to_dict():
    result = ner_text().reannotate(StubLinker(n=1))
    d = result.to_dict()
    assert d["@type"] == "renci_ner:AnnotatedText"
    assert d["location"] == ["file.txt", "row=1"]
    ann = d["annotations"][0]
    assert ann["@type"] == "renci_ner:NormalizedAnnotation"
    assert ann["biolink_type"] == "biolink:NamedThing"
    assert ann["provenance"] == {
        "@type": "renci_ner:AnnotationProvenance",
        "name": "StubLinker",
        "url": "http://stub.example/",
        "version": "1",
    }
    assert ann["based_on"][0]["@type"] == "renci_ner:Annotation"
    assert "biolink_type" not in ann["based_on"][0]
    # Everything in there must survive a JSON round trip.
    assert json.loads(json.dumps(d)) == d


def test_str_is_compact():
    result = ner_text().reannotate(StubLinker(n=1))
    assert str(result.annotations[0]) == (
        "NormalizedAnnotation('brain' [4:9] -> STUB:0 'brain' biolink:NamedThing)"
    )
    assert "based_on" not in str(result)


class Upper(Annotator):
    """Annotates the whole text, labelled in upper case."""

    provenance = AnnotationProvenance("Upper", "http://upper.example/", "1")

    def _annotate(self, text, props):
        return AnnotatedText(
            text,
            [
                Annotation(
                    text,
                    "X:1",
                    text.upper(),
                    "biolink:NamedThing",
                    0,
                    len(text),
                    self.provenance,
                )
            ],
        )


def csv_record():
    return Record(
        ["in.csv", "row=1"],
        {"id": "1", "name": "Brain", "note": ""},
        {"name": Text("Brain", ["in.csv", "row=1", "name"])},
    )


def test_record_annotate_keeps_location_and_cells():
    record = csv_record()
    result = record.annotate(Upper())
    assert result.location == record.location
    assert result.cells == record.cells
    (text,) = result.texts.values()
    assert isinstance(text, AnnotatedText)
    assert text.location == ["in.csv", "row=1", "name"]
    assert text.annotations[0].label == "BRAIN"
    # The original record is unchanged.
    assert isinstance(record.texts["name"], Text)


def test_record_cannot_be_annotated_twice():
    # Its texts are AnnotatedTexts by then, which annotate() refuses.
    with pytest.raises(TypeError, match="reannotate"):
        csv_record().annotate(Upper()).annotate(Upper())


def test_annotate_refuses_a_record():
    with pytest.raises(TypeError, match="record.annotate"):
        Upper().annotate(csv_record())


def test_record_from_text():
    text = Text("brain", ["in.txt", "row=3", "text"])
    record = Record.from_text(text)
    assert record.location == ["in.txt", "row=3", "text"]
    assert record.cells == {"text": "brain"}
    assert record.texts == {"text": text}


def test_record_to_dict():
    d = csv_record().annotate(Upper()).to_dict()
    assert d["@type"] == "renci_ner:Record"
    assert d["location"] == ["in.csv", "row=1"]
    assert d["cells"] == {"id": "1", "name": "Brain", "note": ""}
    assert d["texts"]["name"]["@type"] == "renci_ner:AnnotatedText"
    assert d["texts"]["name"]["annotations"][0]["label"] == "BRAIN"
    assert json.loads(json.dumps(d)) == d


def test_from_dict_round_trips():
    linked = ner_text().reannotate(StubLinker(n=2))
    assert isinstance(linked.annotations[0], NormalizedAnnotation)
    record = Record(
        ["in.csv", "row=1"], {"id": "1", "text": linked.text}, {"text": linked}
    )
    for obj in [
        linked.annotations[0].provenance,
        linked.annotations[0].based_on[0],
        linked.annotations[0],
        linked,
        Text("brain", ["in.txt", "row=1", "text"]),
        record,
        csv_record(),
    ]:
        # Through JSON too, as a JSONL file would carry it.
        rebuilt = from_dict(json.loads(json.dumps(obj.to_dict())))
        assert type(rebuilt) is type(obj)
        assert rebuilt == obj


@pytest.mark.parametrize(
    "data, message",
    [
        (["not", "a", "dict"], "JSON object"),
        ({"text": "no type"}, "Unknown @type"),
        ({"@type": "renci_ner:Text"}, "Invalid renci_ner:Text"),
        ({"@type": "renci_ner:Text", "text": 1}, "must be a string"),
        (
            {
                "@type": "renci_ner:AnnotatedText",
                "text": "a",
                "annotations": [{"@type": "renci_ner:Text", "text": "a"}],
            },
            "Expected an Annotation",
        ),
        (
            {
                "@type": "renci_ner:Record",
                "cells": {},
                "texts": {
                    "a": {
                        "@type": "renci_ner:AnnotationProvenance",
                        "name": "n",
                        "url": "u",
                        "version": "v",
                    }
                },
            },
            "Texts or AnnotatedTexts",
        ),
    ],
)
def test_from_dict_errors(data, message):
    with pytest.raises(ValueError, match=message):
        from_dict(data)
