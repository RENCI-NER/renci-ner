from renci_ner.core import AnnotatorWithProps
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron


def test_check():
    """Check that Bagel can be used as intended."""
    biomegatron = BioMegatron()
    annotated_text = biomegatron.annotate(
        "In orbital cellulitis, an infection travels through the ophthalmic vein into the dural venous sinuses, " +
        "causing dural sinus thrombosis.")

    bagel = BagelAnnotator()
    result = bagel.annotate_with(annotated_text, [
        AnnotatorWithProps(
            annotator=BabelSAPBERTAnnotator(),
            props={"limit": 10}
        ),
        AnnotatorWithProps(
            annotator=NameRes(),
            props={"limit": 10}
        )
    ])
    assert result.text == "brain"
    annotations = result.annotations
    assert len(annotations) == 11
    top_annot = annotations[0]
    assert top_annot.label == "brain"
    assert top_annot.id == "UBERON:0000955"
    assert top_annot.type == "biolink:GrossAnatomicalStructure"

    assert top_annot.provenance.name == "NameRes"
    assert top_annot.provenance.url == "https://name-resolution-sri.renci.org"
