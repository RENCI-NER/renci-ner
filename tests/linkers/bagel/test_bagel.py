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
    except (HTTPError, ValueError) as err:
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


def offline_bagel(bagel_results, normalized=None):
    """A BagelAnnotator that never touches the network and answers every span with bagel_results."""
    bagel = BagelAnnotator.__new__(BagelAnnotator)
    bagel.openapi_version = "test"

    class FakeNodeNorm:
        def normalize(self, identifiers, props=None):
            return normalized or {}

    bagel.nodenorm = FakeNodeNorm()
    bagel.query_bagel = lambda entity, context, matches, props_json: bagel_results
    return bagel


def candidate(provenance, curie, label="brain"):
    ner = Annotation(
        "brain",
        "I1",
        "",
        "biolink:AnatomicalEntity",
        4,
        9,
        AnnotationProvenance("NER", "http://ner.example/", "1"),
    )
    return NormalizedAnnotation(
        text="brain",
        id=curie,
        label=label,
        type="biolink:AnatomicalEntity",
        biolink_type="biolink:AnatomicalEntity",
        start=4,
        end=9,
        provenance=provenance,
        based_on=[ner],
    )


SAPBERT = AnnotationProvenance("BabelSAPBERT", "http://sapbert.example/", "1")
NAMERES = AnnotationProvenance("NameRes", "http://nameres.example/", "1")


def test_no_bagel_results_keeps_the_span():
    """If Bagel returns nothing (e.g. a 403, or it rejects every candidate), the span must not vanish."""
    annotated = AnnotatedText("The brain.", [candidate(SAPBERT, "UBERON:0000955")])
    result = offline_bagel([]).transform(annotated)
    assert result.annotations == annotated.annotations


def test_shared_curie_credits_the_first_linker():
    """The same CURIE from two linkers is one candidate; the winner is based_on the first one seen."""
    annotated = AnnotatedText(
        "The brain.",
        [candidate(SAPBERT, "UBERON:0000955"), candidate(NAMERES, "UBERON:0000955")],
    )
    sent = []
    bagel = offline_bagel(
        [
            BagelResult(
                identifier="UBERON:0000955",
                label="brain",
                entity_type="biolink:AnatomicalEntity",
                synonym_type="exact",
            )
        ]
    )
    real_query = bagel.query_bagel
    bagel.query_bagel = lambda entity, context, matches, props_json: sent.append(
        matches
    ) or real_query(entity, context, matches, props_json)

    (winner,) = bagel.transform(annotated, {"limit": 1}).annotations
    assert len(sent[0]) == 1
    assert [p.name for p in winner.provenances] == ["NER", "BabelSAPBERT", "Bagel"]


def test_unknown_identifier_from_bagel_keeps_the_ner_chain():
    annotated = AnnotatedText("The brain.", [candidate(SAPBERT, "UBERON:0000955")])
    bagel = offline_bagel(
        [
            BagelResult(
                identifier="MONDO:9999999",
                label="made up",
                entity_type="biolink:Disease",
                synonym_type="exact",
            )
        ]
    )
    (winner,) = bagel.transform(annotated, {"limit": 1}).annotations
    assert winner.id == "MONDO:9999999"
    assert [p.name for p in winner.provenances] == ["NER", "Bagel"]
    assert (winner.start, winner.end, winner.text) == (4, 9, "brain")


def test_candidate_without_nodenorm_entry_keeps_its_own_type():
    """What Bagel is sent: NodeNorm's type/description when known, else the candidate's biolink_type."""
    annotated = AnnotatedText(
        "The brain.",
        [
            candidate(SAPBERT, "UBERON:0000955"),
            candidate(NAMERES, "UMLS:C0006104", "Brain"),
        ],
    )
    sent = []
    bagel = offline_bagel(
        [],
        normalized={
            "UBERON:0000955": {
                "id": {"identifier": "UBERON:0000955", "description": "The brain."},
                "type": ["biolink:GrossAnatomicalStructure"],
            }
        },
    )
    bagel.query_bagel = (
        lambda entity, context, matches, props_json: sent.append(matches) or []
    )
    bagel.transform(annotated)
    by_id = {m.identifier: m for m in sent[0]}
    assert (
        by_id["UBERON:0000955"].entity_type,
        by_id["UBERON:0000955"].description,
    ) == (
        "biolink:GrossAnatomicalStructure",
        "The brain.",
    )
    assert (by_id["UMLS:C0006104"].entity_type, by_id["UMLS:C0006104"].description) == (
        "biolink:AnatomicalEntity",
        "",
    )
