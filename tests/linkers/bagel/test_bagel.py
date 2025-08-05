from renci_ner.core import AnnotatorWithProps, Annotation, AnnotationProvenance
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron


def test_check():
    """Check that Bagel can be used as intended."""
    biomegatron = BioMegatron()
    text = (
        "In orbital cellulitis, an infection travels through the ophthalmic vein into the dural venous sinuses, "
        + "causing dural sinus thrombosis."
    )
    annotated_text = biomegatron.annotate(text)

    bagel = BagelAnnotator()
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
            id="UMLS:C0022882",
            label="Laboratory Infection",
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
            id="UMLS:C1739120",
            label="Lochial infection",
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
            id="UMLS:C1400623",
            label="infection; multiple",
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
            id="MONDO:0005504",
            label="diphtheria",
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
                "description": "A Gram-positive bacterial infection caused by Corynebacterium diphtheriae. It usually involves the oral cavity, pharynx, and nasal cavity. Patients develop pseudomembranes in the affected areas and manifest signs and symptoms of an upper respiratory infection. The diphtheria toxin may cause myocarditis, polyneuritis, and other systemic effects."
            },
        ),
        Annotation(
            text="infection",
            id="MONDO:0002026",
            label="candidiasis",
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
            props={"description": "Infection with the organism Candida."},
        ),
        Annotation(
            text="infection",
            id="MONDO:0000827",
            label="salmonellosis",
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
            props={"description": "Infections with bacteria of the genus salmonella."},
        ),
        Annotation(
            text="infection",
            id="MONDO:0000367",
            label="taeniasis",
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
                "description": "A parasitic infection caused by tapeworms of the genus Taenia. Humans are infected by eating undercooked or raw meat of infected animals. It is usually an asymptomatic infection and patients may become aware of the infection by noticing segments of the tapeworm in their feces. If symptoms are present, they include nausea, abdominal pain, indigestion, constipation, or diarrhea."
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
            text="ophthalmic vein",
            id="UBERON:0011193",
            label="inferior ophthalmic vein",
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
                "description": "The inferior ophthalmic vein begins in a venous net-work at the forepart of the floor and medial wall of the orbit; it receives some vorticose veins and other veins from the Rectus inferior, Obliquus inferior, lacrimal sac and eyelids, runs backward in the lower part of the orbit and divides into two branches. One of these passes through the inferior orbital fissure and joins the pterygoid venous plexus, while the other enters the cranium through the superior orbital fissure and ends in the cavernous sinus, either by a separate opening, or more frequently in common with the superior ophthalmic vein."
            },
        ),
        Annotation(
            text="ophthalmic vein",
            id="UMLS:C0924205",
            label="Inferior orbital vein",
            type="biolink:AnatomicalEntity",
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
            id="UMLS:C0934292",
            label="Unpaired dural venous sinus",
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
            id="UMLS:C0925220",
            label="Paired dural venous sinus",
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
        Annotation(
            text="dural sinus thrombosis",
            id="MONDO:0002695",
            label="sagittal sinus thrombosis",
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
                "description": "Formation or presence of a blood clot (thrombus) in the superior sagittal sinus or the inferior sagittal sinus. Sagittal sinus thrombosis can result from infections, hematological disorders, craniocerebral trauma; and neurosurgical procedures. Clinical features are primarily related to the increased intracranial pressure causing headache; nausea; and vomiting. Severe cases can evolve to seizures or coma."
            },
        ),
        Annotation(
            text="dural sinus thrombosis",
            id="MONDO:0002693",
            label="lateral sinus thrombosis",
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
                "description": "Formation or presence of a blood clot (thrombus) in the lateral sinuses. This condition is often associated with ear infections (otitis media or mastoiditis) without antibiotic treatment. In developed nations, lateral sinus thrombosis can result from craniocerebral trauma; brain neoplasms; neurosurgical procedures; thrombophilia; and other conditions. Clinical features include headache; vertigo; and increased intracranial pressure."
            },
        ),
        Annotation(
            text="dural sinus thrombosis",
            id="MONDO:0002694",
            label="cavernous sinus thrombosis",
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
                "description": "Formation or presence of a blood clot (thrombus) in the cavernous sinus of the brain. Infections of the paranasal sinuses and adjacent structures, craniocerebral trauma, and thrombophilia are associated conditions. Clinical manifestations include dysfunction of cranial nerves iii, iv, V, and vi, marked periorbital swelling, chemosis, fever, and visual loss. (From Adams et al., Principles of Neurology, 6th ed, p711)"
            },
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
    ]
