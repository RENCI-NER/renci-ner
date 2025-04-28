import pytest

from renci_ner.annotations import NormalizedAnnotation


def test_normalized_annotations():
    # NormalizedAnnotations without a `biolink:` prefix are not allowed!
    with pytest.raises(ValueError):
        normalized_annotation = NormalizedAnnotation(
            text='brain',
            id='UBERON:0000955',
            label='brain',
            type='AnatomicalEntity',
            biolink_type='AnatomicalEntity',
            start=0,
            end=4
        )
    normalized_annotation = NormalizedAnnotation(
        text='brain',
        id='UBERON:0000955',
        label='brain',
        type='AnatomicalEntity',
        biolink_type='biolink:AnatomicalEntity',
        start=0,
        end=4
    )
    with pytest.raises(ValueError):
        normalized_annotation.biolink_type = "AnatomicalEntity"
