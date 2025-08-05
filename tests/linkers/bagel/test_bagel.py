from renci_ner.core import AnnotatorWithProps
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron


def test_check():
    """Check that Bagel can be used as intended."""
    biomegatron = BioMegatron()
    text = "In orbital cellulitis, an infection travels through the ophthalmic vein into the dural venous sinuses, " + \
        "causing dural sinus thrombosis."
    annotated_text = biomegatron.annotate(text)

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
    assert result.text == annotated_text.text
    assert result.annotations == []
