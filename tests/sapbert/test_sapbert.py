from renci_ner.services.sapbert import SAPBERTAnnotator


def test_check():
    nameres = SAPBERTAnnotator()
    result = nameres.annotate("brain", {"limit": 11})
    assert result.text == "brain"
    annotations = result.annotations
    assert len(annotations) == 11
    top_annot = annotations[0]
    assert top_annot.label == "brain"
    assert top_annot.id == "UBERON:0000955"
    assert top_annot.type == "biolink:GrossAnatomicalStructure"
