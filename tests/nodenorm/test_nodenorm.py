import pytest
from requests import HTTPError

from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.sapbert import SAPBERTAnnotator
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm


def test_check():
    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    nameres = NameRes()
    sapbert = SAPBERTAnnotator()
    nodenorm = NodeNorm()

    text = "Actin in the brain is part of the nervous system."
    result_nameres = (
        biomegatron.annotate(text)
        .reannotate(nameres, {"limit": 1})
        .transform(nodenorm, {"geneprotein_conflation": True})
    )
    result_sapbert = (
        biomegatron.annotate(text)
        .reannotate(sapbert, {"limit": 1})
        .transform(nodenorm, {"geneprotein_conflation": True})
    )

    assert result_nameres.text == text
    assert result_nameres.text == result_sapbert.text
    assert len(result_nameres.annotations) == 3
    assert len(result_nameres.annotations) == len(result_sapbert.annotations)

    # Let's look at the actin annotation in more detail.
    nameres_actin = result_nameres.annotations[0]
    sapbert_actin = result_sapbert.annotations[0]

    # NameRes returns
    assert nameres_actin.based_on[-1].id == "UniProtKB:P63261"
    assert nameres_actin.based_on[-1].label == "ACTG_HUMAN Actin, cytoplasmic 2 (sprot)"
    assert nameres_actin.text == "Actin"
    assert nameres_actin.id == "NCBIGene:71"
    assert nameres_actin.label == "ACTG1"
    assert nameres_actin.curie == "NCBIGene:71"
    assert nameres_actin.biolink_type == "biolink:Gene"
    assert nameres_actin.provenance == nodenorm.provenance

    # SAPBERT returns ACTIN (PANTHER.FAMILY:PTHR11937), which normalizes to itself,
    # so effectively no normalization appears to have occurred.
    assert sapbert_actin.label == "ACTIN"
    assert sapbert_actin.id == "PANTHER.FAMILY:PTHR11937"
    assert sapbert_actin.biolink_type == "biolink:GeneFamily"
    assert sapbert_actin.provenance == sapbert.provenance
