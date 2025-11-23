import pytest
from requests import HTTPError

from renci_ner.core import AnnotatorWithProps, Annotation
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron


def test_check():
    """Check that Bagel can be used as intended."""

    # Make sure all three services are available.
    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    try:
        babel_sapbert = BabelSAPBERTAnnotator()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    try:
        nameres = NameRes()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    text = (
        "In orbital cellulitis, an infection travels through the ophthalmic vein into the dural venous sinuses, "
        + "causing dural sinus thrombosis."
    )
    annotated_text = biomegatron.annotate(text)

    bagel = BagelAnnotator()
    result = bagel.annotate_with(
        annotated_text,
        [
            AnnotatorWithProps(annotator=babel_sapbert, props={"limit": 10}),
            AnnotatorWithProps(annotator=nameres, props={"limit": 10}),
        ],
        {
            # After doing the Bagel-ing, only choose the single best result from Bagel.
            "limit": 1
        }
    )
    assert result.text == annotated_text.text
    assert result.annotations == [
        Annotation(
            text="orbital cellulitis",
            id="MONDO:0006881",
            label="orbital cellulitis",
            type="biolink:Disease",
            start=3,
            end=21,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="orbital cellulitis",
                    id="I1-",
                    label="",
                    type="biolink:Disease",
                    start=3,
                    end=21,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                )
            ],
            props={
                "description": ""
            },
        ),
        Annotation(
            text="infection",
            id="UMLS:C3714514",
            label="Infection",
            type="biolink:Disease",
            start=26,
            end=35,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="infection",
                    id="I4-",
                    label="",
                    type="biolink:Disease",
                    start=26,
                    end=35,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                )
            ],
            props={
                "description": "",
            },
        ),
        Annotation(
            text="ophthalmic vein",
            id="UBERON:0011191",
            label="ophthalmic vein",
            type="biolink:GrossAnatomicalStructure",
            start=56,
            end=71,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="ophthalmic vein",
                    id="I8-",
                    label="",
                    type="biolink:GrossAnatomicalStructure",
                    start=56,
                    end=71,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                )
            ],
            props={
                "description": "",
            },
        ),
        Annotation(
            text="dural venous sinuses",
            id="UBERON:0005486",
            label="'venous dural sinus",
            type="biolink:GrossAnatomicalStructure",
            start=81,
            end=101,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="dural venous sinuses",
                    id="I12-",
                    label="",
                    type="biolink:AnatomicalEntity",
                    start=81,
                    end=101,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                )
            ],
            props={"description": ""},
        ),
        Annotation(
            text="dural sinus thrombosis",
            id="MONDO:0002907",
            label="intracranial thrombosis",
            type="biolink:Disease",
            start=111,
            end=133,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="dural sinus thrombosis",
                    id="I16-",
                    label="",
                    type="biolink:Disease",
                    start=111,
                    end=133,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                )
            ],
            props={"description": ""},
        ),
    ]
