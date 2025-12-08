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

    text = "In orbital cellulitis, an acute bacterial infection travels through the ophthalmic vein into the eye socket."
    annotated_text = biomegatron.annotate(text)

    try:
        bagel = BagelAnnotator()
    except HTTPError as err:
        pytest.skip(f"Bagel is not available: {err}")
        return

    result = bagel.annotate_with(
        annotated_text,
        [
            AnnotatorWithProps(annotator=babel_sapbert, props={"limit": 10}),
            AnnotatorWithProps(annotator=nameres, props={"limit": 10}),
        ],
        {
            # After doing the Bagel-ing, only choose the single best result from Bagel.
            "limit": 1,
            # Make this as predictable/repeatable as possible.
            "temperature": 0.1,
            "top_p": 0.1,
        },
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
                "description": "Inflammation of the eye tissues posterior to the orbital septum, and generally secondary to an infection spread from adjacent sinuses. Signs and symptoms of the affected eye include sudden loss of vision, erythema, edema, decreased eye movement, and pain. Treatment is conducted via intravenous antibiotics, observation, and surgical intervention when necessary.",
            },
        ),
        Annotation(
            text="acute bacterial infection",
            id="UMLS:C0275518",
            label="Acute infectious disease",
            type="biolink:Disease",
            start=26,
            end=51,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="acute bacterial infection",
                    id="I4-",
                    label="",
                    type="biolink:Disease",
                    start=26,
                    end=51,
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
            start=72,
            end=87,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="ophthalmic vein",
                    id="I10-",
                    label="",
                    type="biolink:GrossAnatomicalStructure",
                    start=72,
                    end=87,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                )
            ],
            props={
                "description": "Ophthalmic veins are veins which drain the eye. More specifically, they can refer to: Superior ophthalmic vein Inferior ophthalmic vein.",
            },
        ),
        Annotation(
            text="eye socket",
            id="UMLS:C3846141",
            label="Eye/Orbit",
            type="biolink:AnatomicalEntity",
            start=97,
            end=107,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="eye socket",
                    id="I14-",
                    label="",
                    type="biolink:AnatomicalEntity",
                    start=97,
                    end=107,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                ),
            ],
            props={
                "description": "",
            },
        ),
    ]
