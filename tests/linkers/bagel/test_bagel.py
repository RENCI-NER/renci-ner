import pytest
from requests import HTTPError

from renci_ner.core import Annotation, AnnotatorWithProps
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
    assert len(result.annotations) == 4

    # Annotation 0: "orbital cellulitis" -- consistently returns the same result.
    assert result.annotations[0] == Annotation(
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
    )

    # Annotations 1-3 can vary between runs because the upstream linkers
    # (BabelSAPBERT and NameRes) return different candidate sets, causing
    # Bagel's LLM re-ranker to pick different winners.

    # Annotation 1: "acute bacterial infection"
    acute_bacterial_infection_based_on = [
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
    ]
    assert result.annotations[1] in [
        Annotation(
            text="acute bacterial infection",
            id="UMLS:C0275518",
            label="Acute infectious disease",
            type="biolink:Disease",
            start=26,
            end=51,
            provenance=bagel.provenance,
            based_on=acute_bacterial_infection_based_on,
            props={"description": ""},
        ),
        Annotation(
            text="acute bacterial infection",
            id="UMLS:C4697762",
            label="acute infectious process",
            type="biolink:PhenotypicFeature",
            start=26,
            end=51,
            provenance=bagel.provenance,
            based_on=acute_bacterial_infection_based_on,
            props={"description": ""},
        ),
    ]

    # Annotation 2: "ophthalmic vein"
    ophthalmic_vein_based_on = [
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
    ]
    assert result.annotations[2] in [
        Annotation(
            text="ophthalmic vein",
            id="UBERON:0011191",
            label="ophthalmic vein",
            type="biolink:GrossAnatomicalStructure",
            start=72,
            end=87,
            provenance=bagel.provenance,
            based_on=ophthalmic_vein_based_on,
            props={
                "description": "Ophthalmic veins are veins which drain the eye. More specifically, they can refer to: Superior ophthalmic vein Inferior ophthalmic vein.",
            },
        ),
        Annotation(
            text="ophthalmic vein",
            id="UBERON:0011193",
            label="inferior ophthalmic vein",
            type="biolink:GrossAnatomicalStructure",
            start=72,
            end=87,
            provenance=bagel.provenance,
            based_on=ophthalmic_vein_based_on,
            props={
                "description": "The inferior ophthalmic vein begins in a venous net-work at the forepart of the floor and medial wall of the orbit; it receives some vorticose veins and other veins from the Rectus inferior, Obliquus inferior, lacrimal sac and eyelids, runs backward in the lower part of the orbit and divides into two branches. One of these passes through the inferior orbital fissure and joins the pterygoid venous plexus, while the other enters the cranium through the superior orbital fissure and ends in the cavernous sinus, either by a separate opening, or more frequently in common with the superior ophthalmic vein.",
            },
        ),
    ]

    # Annotation 3: "eye socket"
    eye_socket_based_on = [
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
    ]
    assert result.annotations[3] in [
        Annotation(
            text="eye socket",
            id="UMLS:C3846141",
            label="Eye/Orbit",
            type="biolink:AnatomicalEntity",
            start=97,
            end=107,
            provenance=bagel.provenance,
            based_on=eye_socket_based_on,
            props={"description": ""},
        ),
        Annotation(
            text="eye socket",
            id="UMLS:C2371860",
            label="Structure of eye socket",
            type="biolink:AnatomicalEntity",
            start=97,
            end=107,
            provenance=bagel.provenance,
            based_on=eye_socket_based_on,
            props={"description": ""},
        ),
    ]
