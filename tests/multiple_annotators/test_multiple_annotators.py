import pytest
from requests import HTTPError

from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.sapbert import SAPBERTAnnotator
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm


def test_multiple_annotators():
    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    nameres = NameRes()
    sapbert = SAPBERTAnnotator()
    nodenorm = NodeNorm()

    text = "The brain is part of the nervous system."
    result_nameres = (
        biomegatron.annotate(text).reannotate(nameres, {"limit": 1}).transform(nodenorm)
    )
    result_sapbert = (
        biomegatron.annotate(text).reannotate(sapbert, {"limit": 1}).transform(nodenorm)
    )

    assert result_nameres.text == text
    assert result_nameres.text == result_sapbert.text
    assert len(result_nameres.annotations) == 2
    assert len(result_nameres.annotations) == len(result_sapbert.annotations)

    # Make sure that all the annotations are identical between NodeNorm and NameRes.
    for nameres_annotation, sapbert_annotation in zip(
        result_nameres.annotations, result_sapbert.annotations
    ):
        assert nameres_annotation.text == sapbert_annotation.text
        assert nameres_annotation.id == sapbert_annotation.id
        assert nameres_annotation.label == sapbert_annotation.label
        assert nameres_annotation.type == sapbert_annotation.type

        # NameRes and BabelSAPBERT should not need to be normalized.
        assert len(nameres_annotation.provenances) == 2
        assert nameres_annotation.provenances[0] == biomegatron.provenance
        assert nameres_annotation.provenances[1] == nameres.provenance

        # Some SAPBERT annotations may need to be normalized.
        if len(sapbert_annotation.provenances) == 2:
            assert sapbert_annotation.provenances[0] == biomegatron.provenance
            assert sapbert_annotation.provenances[1] == sapbert.provenance
        else:
            assert sapbert_annotation.provenances[0] == biomegatron.provenance
            assert sapbert_annotation.provenances[1] == sapbert.provenance
            assert sapbert_annotation.provenances[2] == nodenorm.provenance
