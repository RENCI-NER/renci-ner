import pytest
from requests import HTTPError

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    NormalizedAnnotation,
)
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm


def test_multiple_annotators():
    """
    Test multiple annotators and transformers on the same text.
    """

    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    nameres = NameRes()
    sapbert = BabelSAPBERTAnnotator()
    nodenorm = NodeNorm()

    text = "The brain is part of the nervous system."
    result_nameres = (
        biomegatron.annotate(text).reannotate(nameres, {"limit": 1}).transform(nodenorm)
    )
    result_sapbert = (
        biomegatron.annotate(text).reannotate(sapbert, {"limit": 1}).transform(nodenorm)
    )

    # Check NameRes results, which we expect to be identical to the SAPBERT results.
    assert result_nameres.text == text
    assert result_nameres.text == result_sapbert.text
    assert len(result_nameres.annotations) == 2
    assert len(result_nameres.annotations) == len(result_sapbert.annotations)

    # If we did the start/end math correctly, we should be able to recover the texts.
    assert (
        text[result_nameres.annotations[0].start : result_nameres.annotations[0].end]
        == "brain"
    )
    assert (
        text[result_nameres.annotations[1].start : result_nameres.annotations[1].end]
        == "nervous system"
    )

    # Check the results. We need to delete some properties that are likely to change often.
    del result_nameres.annotations[0].props["clique_identifier_count"]
    del result_nameres.annotations[0].props["score"]
    del result_nameres.annotations[0].props["synonyms"]
    del result_nameres.annotations[1].props["clique_identifier_count"]
    del result_nameres.annotations[1].props["score"]
    del result_nameres.annotations[1].props["synonyms"]

    assert result_nameres == AnnotatedText(
        "The brain is part of the nervous system.",
        [
            NormalizedAnnotation(
                text="brain",
                id="UBERON:0000955",
                label="brain",
                type="biolink:GrossAnatomicalStructure",
                biolink_type="biolink:GrossAnatomicalStructure",
                start=4,
                end=9,
                provenance=nameres.provenance,
                based_on=[
                    Annotation(
                        text="brain",
                        id="I1-",
                        label="",
                        type="biolink:AnatomicalEntity",
                        start=4,
                        end=9,
                        provenance=biomegatron.provenance,
                        based_on=[],
                        props={},
                    )
                ],
                props={
                    "highlighting": {},
                    "taxa": [],
                    "types": [
                        "biolink:GrossAnatomicalStructure",
                        "biolink:AnatomicalEntity",
                        "biolink:PhysicalEssence",
                        "biolink:OrganismalEntity",
                        "biolink:SubjectOfInvestigation",
                        "biolink:BiologicalEntity",
                        "biolink:ThingWithTaxon",
                        "biolink:NamedThing",
                        "biolink:Entity",
                        "biolink:PhysicalEssenceOrOccurrent",
                    ],
                },
            ),
            NormalizedAnnotation(
                text="nervous system",
                id="UBERON:0001016",
                label="nervous system",
                type="biolink:AnatomicalEntity",
                biolink_type="biolink:AnatomicalEntity",
                start=25,
                end=39,
                provenance=nameres.provenance,
                based_on=[
                    Annotation(
                        text="nervous system",
                        id="I6-",
                        label="",
                        type="biolink:AnatomicalEntity",
                        start=25,
                        end=39,
                        provenance=biomegatron.provenance,
                        based_on=[],
                        props={},
                    )
                ],
                props={
                    "highlighting": {},
                    "taxa": [],
                    "types": [
                        "biolink:AnatomicalEntity",
                        "biolink:PhysicalEssence",
                        "biolink:OrganismalEntity",
                        "biolink:SubjectOfInvestigation",
                        "biolink:BiologicalEntity",
                        "biolink:ThingWithTaxon",
                        "biolink:NamedThing",
                        "biolink:Entity",
                        "biolink:PhysicalEssenceOrOccurrent",
                    ],
                },
            ),
        ],
    )

    # Make sure that all the annotations are identical between NodeNorm and NameRes.
    for nameres_annotation, sapbert_annotation in zip(
        result_nameres.annotations, result_sapbert.annotations
    ):
        assert nameres_annotation.text == sapbert_annotation.text
        assert nameres_annotation.id == sapbert_annotation.id
        assert nameres_annotation.label == sapbert_annotation.label
        assert nameres_annotation.type == sapbert_annotation.type

        # NameRes and BabelSAPBERT should not need to be normalized.
        assert len(nameres_annotation.provenances) == 2
        assert nameres_annotation.provenances[0] == biomegatron.provenance
        assert nameres_annotation.provenances[1] == nameres.provenance

        # Some SAPBERT annotations may need to be normalized.
        if len(sapbert_annotation.provenances) == 2:
            assert sapbert_annotation.provenances[0] == biomegatron.provenance
            assert sapbert_annotation.provenances[1] == sapbert.provenance
        else:
            assert sapbert_annotation.provenances[0] == biomegatron.provenance
            assert sapbert_annotation.provenances[1] == sapbert.provenance
            assert sapbert_annotation.provenances[2] == nodenorm.provenance
