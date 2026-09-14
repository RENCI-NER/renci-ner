import json

import pytest

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    Annotator,
    NormalizedAnnotation,
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
