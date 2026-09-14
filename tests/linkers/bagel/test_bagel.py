import pytest
from requests import HTTPError

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    MultiAnnotator,
    NormalizedAnnotation,
    Pipeline,
)
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator, BagelResult
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron


def test_transform_groups_candidates_by_span(monkeypatch):
    """Offline: Bagel picks among linked candidates per span and passes unlinked spans through."""
    ner = AnnotationProvenance("NER", "http://ner.example/", "1")
    linker = AnnotationProvenance("Linker", "http://linker.example/", "1")
    text = "The brain is part of the nervous system."
    brain = Annotation("brain", "I1", "", "biolink:AnatomicalEntity", 4, 9, ner)
    unlinked = Annotation(
        "nervous system", "I2", "", "biolink:AnatomicalEntity", 25, 39, ner
    )

    def candidate(curie, label):
        return NormalizedAnnotation(
            text="brain",
            id=curie,
            label=label,
            type="biolink:AnatomicalEntity",
            biolink_type="biolink:AnatomicalEntity",
            start=4,
            end=9,
            provenance=linker,
            based_on=[brain],
        )

    annotated = AnnotatedText(
        text,
        [
            candidate("UBERON:0000955", "brain"),
            candidate("UMLS:C0006104", "Brain"),
            unlinked,
        ],
        location=["doc"],
    )

    # Build a BagelAnnotator without touching the network.
    bagel = BagelAnnotator.__new__(BagelAnnotator)
    bagel.openapi_version = "test"
    queries = []

    class FakeNodeNorm:
        def normalize(self, identifiers, props=None):
            queries.append(("nodenorm", identifiers))
            return {
                "UBERON:0000955": {
                    "id": {"identifier": "UBERON:0000955", "description": "The brain."},
                    "type": ["biolink:GrossAnatomicalStructure"],
                }
            }

    def fake_query_bagel(entity_text, context_text, possible_matches, props_json):
        queries.append(
            ("bagel", entity_text, sorted(m.identifier for m in possible_matches))
        )
        # Bagel prefers the UMLS one; the limit should keep only it.
        return [
            BagelResult(
                identifier="UMLS:C0006104",
                label="Brain",
                entity_type="biolink:AnatomicalEntity",
                synonym_type="exact",
            ),
            BagelResult(
                identifier="UBERON:0000955",
                label="brain",
                entity_type="biolink:GrossAnatomicalStructure",
                description="The brain.",
                synonym_type="narrow",
            ),
        ]

    bagel.nodenorm = FakeNodeNorm()
    monkeypatch.setattr(bagel, "query_bagel", fake_query_bagel)

    result = bagel.transform(annotated, {"limit": 1})

    assert queries == [
        ("nodenorm", ["UBERON:0000955", "UMLS:C0006104"]),
        ("bagel", "brain", ["UBERON:0000955", "UMLS:C0006104"]),
    ]
    assert result.location == ["doc"]
    assert len(result.annotations) == 2
    winner, passthrough = result.annotations
    assert isinstance(winner, NormalizedAnnotation)
    assert (winner.id, winner.label, winner.start, winner.end) == (
        "UMLS:C0006104",
        "Brain",
        4,
        9,
    )
    assert winner.props == {"description": ""}
    assert [p.name for p in winner.provenances] == ["NER", "Linker", "Bagel"]
    assert winner.based_on[-1].id == "UMLS:C0006104"
    assert passthrough is unlinked


def test_check():
    """Check that Bagel can be used at the end of a pipeline."""

    try:
        biomegatron = BioMegatron()
        babel_sapbert = BabelSAPBERTAnnotator()
        nameres = NameRes()
        bagel = BagelAnnotator()
    except HTTPError as err:
        pytest.skip(f"A service is not available: {err}")
        return

    text = "In orbital cellulitis, an acute bacterial infection travels through the ophthalmic vein into the eye socket."
    pipeline = Pipeline(
        biomegatron,
        MultiAnnotator((babel_sapbert, {"limit": 10}), (nameres, {"limit": 10})),
        (
            bagel,
            {
                # After doing the Bagel-ing, only choose the single best result from Bagel.
                "limit": 1,
                # Make this as predictable/repeatable as possible.
                "temperature": 0.1,
                "top_p": 0.1,
            },
        ),
    )
    result = pipeline.annotate(text)
    assert result.text == text
    assert len(result.annotations) == 4

    # Every winner came from BioMegatron, then one of the two linkers, then Bagel.
    for ann in result.annotations:
        assert isinstance(ann, NormalizedAnnotation)
        assert text[ann.start : ann.end] == ann.text
        names = [p.name for p in ann.provenances]
        assert names[0] == "BioMegatron"
        assert names[1] in {"BabelSAPBERT", "NameRes"}
        assert names[2] == "Bagel"
        assert ann.based_on[-1].id == ann.id

    def summary(ann):
        return (ann.text, ann.id, ann.label, ann.type)

    # "orbital cellulitis" consistently returns the same result.
    assert summary(result.annotations[0]) == (
        "orbital cellulitis",
        "MONDO:0006881",
        "orbital cellulitis",
        "biolink:Disease",
    )
    assert (
        result.annotations[0]
        .props["description"]
        .startswith("Inflammation of the eye tissues posterior to the orbital septum")
    )

    # The rest can vary between runs because Bagel's LLM re-ranker can pick different winners.
    assert summary(result.annotations[1]) in [
        (
            "acute bacterial infection",
            "UMLS:C0275518",
            "Acute infectious disease",
            "biolink:Disease",
        ),
        (
            "acute bacterial infection",
            "UMLS:C4697762",
            "acute infectious process",
            "biolink:PhenotypicFeature",
        ),
    ]
    assert summary(result.annotations[2]) in [
        (
            "ophthalmic vein",
            "UBERON:0011191",
            "ophthalmic vein",
            "biolink:GrossAnatomicalStructure",
        ),
        (
            "ophthalmic vein",
            "UBERON:0011193",
            "inferior ophthalmic vein",
            "biolink:GrossAnatomicalStructure",
        ),
    ]
    assert summary(result.annotations[3]) in [
        ("eye socket", "UMLS:C3846141", "Eye/Orbit", "biolink:AnatomicalEntity"),
        (
            "eye socket",
            "UMLS:C2371860",
            "Structure of eye socket",
            "biolink:AnatomicalEntity",
        ),
    ]
