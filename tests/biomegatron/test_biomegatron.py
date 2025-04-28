import pytest
from requests import HTTPError

from renci_ner.core import AnnotationProvenance
from renci_ner.services.ner.biomegatron import BioMegatron


def test_check():
    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    query = "The brain is a significant part of the nervous system."
    result = biomegatron.annotate(query)
    assert result.text == query
    annotations = result.annotations
    assert len(annotations) == 2

    brain = annotations[0]
    assert brain.label == ""
    assert brain.type == "biolink:AnatomicalEntity"
    assert len(brain.provenances) == 1
    assert brain.provenances[0] == AnnotationProvenance(
        name="BioMegatron", version="0.1.0", url="https://med-nemo.apps.renci.org"
    )

    nervous_system = annotations[1]
    assert nervous_system.label == ""
    assert nervous_system.type == "biolink:AnatomicalEntity"
    assert len(nervous_system.provenances) == 1
    assert nervous_system.provenances[0] == AnnotationProvenance(
        name="BioMegatron", version="0.1.0", url="https://med-nemo.apps.renci.org"
    )
