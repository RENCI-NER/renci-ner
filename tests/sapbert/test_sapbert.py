from renci_ner.core import AnnotationProvenance
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator


def test_check():
    """Check that BabelSAPBERT can be used to annotate a single word."""
    sapbert = BabelSAPBERTAnnotator()
    result = sapbert.annotate("brain", {"limit": 11})
    assert result.text == "brain"
    annotations = result.annotations
    assert len(annotations) == 11
    top_annot = annotations[0]
    assert top_annot.label == "brain"
    assert top_annot.id == "UBERON:0000955"
    assert top_annot.type == "biolink:GrossAnatomicalStructure"
    assert top_annot.provenance == AnnotationProvenance(
        name="BabelSAPBERT", version="0.1.0", url="https://sap-qdrant.apps.renci.org"
    )
