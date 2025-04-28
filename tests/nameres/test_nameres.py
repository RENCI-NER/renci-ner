import re

from renci_ner.services.linkers.nameres import NameRes


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

    assert top_annot.provenance.name == "NameRes"
    assert top_annot.provenance.url == "https://name-resolution-sri.renci.org"

    # NameRes version changes quite frequently, but we can confirm that we're still in a 1.x.x version.
    assert re.compile(r"^1.\d+.\d+").match(top_annot.provenance.version)
