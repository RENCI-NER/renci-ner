from renci_ner.services.biomegatron import BioMegatron


def test_check():
    biomegatron = BioMegatron()

    query = "The brain is a significant part of the nervous system."
    result = biomegatron.annotate(query)
    assert result.text == query
    annotations = result.annotations
    assert len(annotations) == 2

    brain = annotations[0]
    assert brain.label == ""
    assert brain.type == "biolink:AnatomicalEntity"

    nervous_system = annotations[1]
    assert brain.label == ""
    assert brain.type == "biolink:AnatomicalEntity"
