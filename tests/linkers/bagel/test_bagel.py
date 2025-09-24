import pytest
from requests import HTTPError

from renci_ner.core import AnnotatorWithProps, Annotation, AnnotationProvenance
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron


def test_check():
    """Check that Bagel can be used as intended."""
    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return
    text = (
        "In orbital cellulitis, an infection travels through the ophthalmic vein into the dural venous sinuses, "
        + "causing dural sinus thrombosis."
    )
    annotated_text = biomegatron.annotate(text)

    try:
        bagel = BagelAnnotator()
    except HTTPError as err:
        pytest.skip(f"Bagel is not available: {err}")
        return
    result = bagel.annotate_with(
        annotated_text,
        [
            AnnotatorWithProps(annotator=BabelSAPBERTAnnotator(), props={"limit": 10}),
            AnnotatorWithProps(annotator=NameRes(), props={"limit": 10}),
        ],
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
                "description": "Inflammation of the eye tissues posterior to the orbital septum, and generally secondary to an infection spread from adjacent sinuses. Signs and symptoms of the affected eye include sudden loss of vision, erythema, edema, decreased eye movement, and pain. Treatment is conducted via intravenous antibiotics, observation, and surgical intervention when necessary."
            },
        ),
        Annotation(
            text="orbital cellulitis",
            id="UMLS:C4537570",
            label="cellulitis of both eyelids",
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
            props={"description": ""},
        ),
        Annotation(
            text="orbital cellulitis",
            id="MONDO:0005230",
            label="cellulitis",
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
                "description": "Inflammation of the dermis and subcutaneous tissues caused by a bacterial infection. Symptoms include erythema, edema, and pain to the affected area."
            },
        ),
        Annotation(
            text="infection",
            id="UMLS:C1400558",
            label="infection; infusion",
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
            props={"description": ""},
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
                "description": "The invasion of an organism's body tissues by disease-causing agents and their multiplication, as well as the reaction by the host to these organisms and/or toxins that the organisms produce."
            },
        ),
        Annotation(
            text="infection",
            id="UMLS:C0275521",
            label="Clinical infection",
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
            props={"description": ""},
        ),
        Annotation(
            text="infection",
            id="MONDO:0043544",
            label="Cross Infection",
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
                "description": "An infection acquired in a hospital or other healthcare setting."
            },
        ),
        Annotation(
            text="ophthalmic vein",
            id="UBERON:2005032",
            label="optic vein",
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
                "description": "Vein that connects to the hyaloid vein that drains the eye. External to eye. Isogai et al. 2001."
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
                "description": "Ophthalmic veins are veins which drain the eye. More specifically, they can refer to: Superior ophthalmic vein Inferior ophthalmic vein."
            },
        ),
        Annotation(
            text="dural venous sinuses",
            id="UBERON:0005486",
            label="Cranial Sinuses",
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
            props={
                "description": "A venous channel found between layers of dura mater in the brain. Receives blood from internal and external veins of the brain, receive cerebrospinal fluid (CSF) from the subarachnoid space, and ultimately empty into the internal jugular vein."
            },
        ),
        Annotation(
            text="dural venous sinuses",
            id="UMLS:C0598122",
            label="intracranial venous sinus",
            type="biolink:AnatomicalEntity",
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
            text="dural venous sinuses",
            id="UBERON:0005486",
            label="venous dural sinus",
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
            props={
                "description": "A venous channel found between layers of dura mater in the brain. Receives blood from internal and external veins of the brain, receive cerebrospinal fluid (CSF) from the subarachnoid space, and ultimately empty into the internal jugular vein."
            },
        ),
        Annotation(
            text="dural venous sinuses",
            id="UBERON:0017635",
            label="paired venous dural sinus",
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
            text="dural venous sinuses",
            id="UBERON:0017640",
            label="unpaired venous dural sinus",
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
            text="dural venous sinuses",
            id="UBERON:0006615",
            label="venous sinus",
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
            props={
                "description": "A large vein or channel for the circulation of venous blood."
            },
        ),
        Annotation(
            text="dural sinus thrombosis",
            id="UMLS:C4538576",
            label="Sinus vein thrombosis",
            type="biolink:PhenotypicFeature",
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
        Annotation(
            text="dural sinus thrombosis",
            id="UMLS:C0270636",
            label="CEREBRAL SINUS THROMBOSIS, EMBOLISM AND INFLAMMATION",
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
        Annotation(
            text="dural sinus thrombosis",
            id="HP:0033724",
            label="Cerebral venous sinus thrombosis",
            type="biolink:PhenotypicFeature",
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
            props={
                "description": "An intracranial thrombosis of the venous sinuses. These typically present with headache, seizures or venous stroke secondary to raised cerebral venous pressure. Cerebral venous sinus thromboses usually affect larger areas of brain parenchyma than those affected by cerebral vein thromboses."
            },
        ),
        Annotation(
            text="dural sinus thrombosis",
            id="MONDO:0002692",
            label="intracranial sinus thrombosis",
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
            props={
                "description": "Formation or presence of a blood clot (thrombus) in the cranial sinuses, large endothelium-lined venous channels situated within the skull. Intracranial sinuses, also called cranial venous sinuses, include the superior sagittal, cavernous, lateral, petrous sinuses, and many others. Cranial sinus thrombosis can lead to severe headache; seizure; and other neurological defects."
            },
        ),
    ]
