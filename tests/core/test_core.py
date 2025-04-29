import pytest

from renci_ner.core import AnnotationProvenance, NormalizedAnnotation


def test_normalized_annotations():
    provenance = AnnotationProvenance("Test", "http://example.com", "0.1.0")

    # NormalizedAnnotations without a `biolink:` prefix are not allowed!
    with pytest.raises(ValueError):
        normalized_annotation = NormalizedAnnotation(
            provenance=provenance,
            text="brain",
            id="UBERON:0000955",
            label="brain",
            type="AnatomicalEntity",
            biolink_type="AnatomicalEntity",
            start=0,
            end=4,
        )
    normalized_annotation = NormalizedAnnotation(
        provenance=provenance,
        text="brain",
        id="UBERON:0000955",
        label="brain",
        type="AnatomicalEntity",
        biolink_type="biolink:AnatomicalEntity",
        start=0,
        end=4,
    )
    with pytest.raises(ValueError):
        normalized_annotation.biolink_type = "AnatomicalEntity"
