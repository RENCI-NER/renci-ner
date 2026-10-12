"""Offline tests for Pipeline and MultiAnnotator using stub services."""

import pytest

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    Annotator,
    MultiAnnotator,
    NormalizedAnnotation,
    Pipeline,
    Transformer,
)

TEXT = "The brain is part of the nervous system."


class StubNER(Annotator):
    provenance = AnnotationProvenance("StubNER", "http://ner.example/", "1")

    def annotate(self, text, props=None):
        spans = [("brain", 4, 9), ("nervous system", 25, 39)]
        return AnnotatedText(
            text,
            [
                Annotation(
                    t, f"I{i}", "", "biolink:AnatomicalEntity", s, e, self.provenance
                )
                for i, (t, s, e) in enumerate(spans)
                if t in text
            ],
        )


class StubLinker(Annotator):
    def __init__(self, name, n=1):
        self.name = name
        self.n = n

    @property
    def provenance(self):
        return AnnotationProvenance(self.name, f"http://{self.name}.example/", "1")

    def annotate(self, text, props=None):
        limit = min(self.n, (props or {}).get("limit", self.n))
        return AnnotatedText(
            text,
            [
                NormalizedAnnotation(
                    text=text,
                    id=f"{self.name}:{i}",
                    label=text,
                    type="biolink:NamedThing",
                    biolink_type="biolink:NamedThing",
                    start=0,
                    end=len(text),
                    provenance=self.provenance,
                )
                for i in range(limit)
            ],
        )


class Upper(Transformer):
    """Uppercases labels, recording the props it was given."""

    provenance = AnnotationProvenance("Upper", "http://upper.example/", "1")

    def __init__(self):
        self.seen_props = []

    def transform(self, annotated_text, props=None):
        self.seen_props.append(props)
        return AnnotatedText(
            annotated_text.text,
            [
                NormalizedAnnotation.from_annotation(
                    ann, self.provenance, label=ann.label.upper()
                )
                for ann in annotated_text.annotations
            ],
            location=annotated_text.location,
        )


def test_pipeline_dispatches_annotators_and_transformers():
    upper = Upper()
    pipeline = Pipeline(
        StubNER(), (StubLinker("nameres"), {"limit": 1}), (upper, {"x": 1})
    )
    result = pipeline.annotate(TEXT)

    assert upper.seen_props == [{"x": 1}]
    assert [a.text for a in result.annotations] == ["brain", "nervous system"]
    assert [a.label for a in result.annotations] == ["BRAIN", "NERVOUS SYSTEM"]
    for ann in result.annotations:
        assert TEXT[ann.start : ann.end] == ann.text
        assert [p.name for p in ann.provenances] == ["StubNER", "nameres", "Upper"]


def test_pipeline_equals_fluent_form():
    ner, linker, upper = StubNER(), StubLinker("nameres", n=3), Upper()
    fluent = ner.annotate(TEXT).reannotate(linker, {"limit": 2}).transform(upper)
    piped = Pipeline(ner, (linker, {"limit": 2}), upper).annotate(TEXT)
    assert piped == fluent
    assert len(piped.annotations) == 4


def test_multi_annotator_concatenates_and_keeps_provenance():
    multi = MultiAnnotator(
        (StubLinker("sapbert", n=2), {"limit": 2}), StubLinker("nameres")
    )
    result = Pipeline(StubNER(), multi).annotate(TEXT)

    # For each of the two NER spans: two sapbert candidates, then one nameres candidate.
    assert [a.id for a in result.annotations] == [
        "sapbert:0",
        "sapbert:1",
        "nameres:0",
        "sapbert:0",
        "sapbert:1",
        "nameres:0",
    ]
    assert [p.name for p in result.annotations[2].provenances] == ["StubNER", "nameres"]
    assert (result.annotations[2].start, result.annotations[2].end) == (4, 9)


def test_nested_pipeline_keeps_full_based_on_chain():
    inner = Pipeline(StubLinker("nameres"), Upper())
    result = Pipeline(StubNER(), inner).annotate(TEXT)
    ann = result.annotations[0]
    assert [p.name for p in ann.provenances] == ["StubNER", "nameres", "Upper"]
    assert (ann.start, ann.end) == (4, 9)
    assert ann.label == "BRAIN"


def test_pipeline_rejects_bad_steps():
    with pytest.raises(TypeError):
        Pipeline(Upper())
    with pytest.raises(TypeError):
        Pipeline(StubNER(), "nodenorm")
