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

    text = "In orbital cellulitis, tissues behind the orbital septum become inflamed."
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
            "temperature": 0,
            "top_p": 0.9,
        },
    )
    assert result.text == annotated_text.text
    assert result.annotations == [
        Annotation(
            text="orbital",
            id="MONDO:0006881",
            label="orbital cellulitis",
            type="biolink:Disease",
            start=3,
            end=10,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="orbital",
                    id="I1-",
                    label="",
                    type="biolink:AnatomicalEntity",
                    start=3,
                    end=10,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                ),
            ],
            props={
                "description": "Inflammation of the eye tissues posterior to the orbital "
                "septum, and generally secondary to an infection spread from "
                "adjacent sinuses. Signs and symptoms of the affected eye "
                "include sudden loss of vision, erythema, edema, decreased "
                "eye movement, and pain. Treatment is conducted via "
                "intravenous antibiotics, observation, and surgical "
                "intervention when necessary.",
            },
        ),
        Annotation(
            text="cellulitis",
            id="MONDO:0005230",
            label="cellulitis",
            type="biolink:Disease",
            start=11,
            end=21,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="cellulitis",
                    id="I2-",
                    label="",
                    type="biolink:Disease",
                    start=11,
                    end=21,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                ),
            ],
            props={
                "description": "Inflammation of the dermis and subcutaneous tissues caused "
                "by a bacterial infection. Symptoms include erythema, edema, "
                "and pain to the affected area.",
            },
        ),
        Annotation(
            text="tissues",
            id="UBERON:0000479",
            label="tissue",
            type="biolink:GrossAnatomicalStructure",
            start=23,
            end=30,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="tissues",
                    id="I3-",
                    label="",
                    type="biolink:GrossAnatomicalStructure",
                    start=23,
                    end=30,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                ),
            ],
            props={
                "description": "Multicellular anatomical structure that consists of many "
                "cells of one or a few types, arranged in an extracellular "
                "matrix such that their long-range organisation is at least "
                "partly a repetition of their short-range organisation.",
            },
        ),
        Annotation(
            text="orbital septum",
            id="UBERON:0001822",
            label="orbital septum",
            type="biolink:GrossAnatomicalStructure",
            start=42,
            end=56,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="orbital septum",
                    id="I6-",
                    label="",
                    type="biolink:AnatomicalEntity",
                    start=42,
                    end=56,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                ),
            ],
            props={
                "description": "A membranous sheet that acts as the anterior boundary of "
                "the orbit. It extends from the orbital rims to the "
                "eyelids[WP].",
            },
        ),
        Annotation(
            text="inflamed",
            id="NCIT:C3137",
            label="Inflammation",
            type="biolink:PhenotypicFeature",
            start=64,
            end=72,
            provenance=bagel.provenance,
            based_on=[
                Annotation(
                    text="inflamed",
                    id="I9-",
                    label="",
                    type="biolink:PhenotypicFeature",
                    start=64,
                    end=72,
                    provenance=biomegatron.provenance,
                    based_on=[],
                    props={},
                ),
            ],
            props={
                "description": "A finding of a localized protective response resulting from "
                "injury or destruction of tissues. Inflammation serves to "
                "destroy, dilute, or wall off both the injurious agent and "
                "the injured tissue. In the acute phase, inflammation is "
                "characterized by the signs of pain, heat, redness, "
                "swelling, and loss of function. Histologically, "
                "inflammation involves a complex series of events, including "
                "dilatation of arterioles, capillaries, and venules, with "
                "increased permeability and blood flow; exudation of fluids, "
                "including plasma proteins; and leukocyte migration into the "
                "site of inflammation.",
            },
        ),
    ]
