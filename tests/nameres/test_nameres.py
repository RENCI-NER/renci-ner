import re

from renci_ner.annotations import AnnotationProvenance
from renci_ner.services.nameres import NameRes


def test_check():
    nameres = NameRes()
    result = nameres.annotate("brain", {"limit": 11})
    assert result.text == "brain"
    annotations = result.annotations
    assert len(annotations) == 11
    top_annot = annotations[0]
    assert top_annot.label == "brain"
    assert top_annot.id == "UBERON:0000955"
    assert top_annot.type == "biolink:GrossAnatomicalStructure"

    assert len(top_annot.provenances) == 1
    prov = top_annot.provenances[0]
    assert prov.name == "NameRes"
    assert prov.url == "https://name-resolution-sri.renci.org"

    # NameRes version changes quite frequently, but we can confirm that we're still in a 1.x.x version.
    assert re.compile(r"^1.\d+.\d+").match(prov.version)
