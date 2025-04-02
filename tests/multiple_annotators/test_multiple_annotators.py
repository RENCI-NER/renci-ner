from renci_ner.services.biomegatron import BioMegatron
from renci_ner.services.nameres import NameRes
from renci_ner.services.sapbert import SAPBERTAnnotator


def test_multiple_annotators():
    biomegatron = BioMegatron()
    nameres = NameRes()
    sapbert = SAPBERTAnnotator()

    text = "The brain is located inside the nervous system."
    result_nameres = biomegatron.annotate(text).annotate_annotations_with(nameres, { "limit": 1 })
    result_sapbert = biomegatron.annotate(text).annotate_annotations_with(sapbert, { "limit": 1 })

    assert result_nameres.text == text
    assert result_nameres.text == result_sapbert.text
    assert len(result_nameres.annotations) == 2
    assert len(result_nameres.annotations) == len(result_sapbert.annotations)

    for (nameres_annotation, sapbert_annotation) in zip(result_nameres.annotations, result_sapbert.annotations):
        assert nameres_annotation.text == sapbert_annotation.text
        assert nameres_annotation.id == sapbert_annotation.id
        assert nameres_annotation.label == sapbert_annotation.label
        assert nameres_annotation.type == sapbert_annotation.type

        assert len(nameres_annotation.provenances) == 2
        assert nameres_annotation.provenances[0] == biomegatron.provenance
        assert nameres_annotation.provenances[1] == nameres.provenance

        assert len(sapbert_annotation.provenances) == 2
        assert sapbert_annotation.provenances[0] == biomegatron.provenance
        assert sapbert_annotation.provenances[1] == sapbert.provenance
