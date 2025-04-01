import datetime
from typing import Self

from renci_ner.annotations import AnnotatedText, AnnotationProvenance


class Annotator:
    """
    An interface for annotating text using a service.
    """

    def provenance(self) -> AnnotationProvenance:
        return AnnotationProvenance(
            name="Annotator",
            url="",
            version="0.0.1",
            at=datetime.datetime.now(datetime.UTC).isoformat()
        )

    def annotate(self, text: str, props: dict = {}) -> AnnotatedText:
        """
        Annotate a text with a set of service-specific properties.
        """
        pass

    def supported_properties(self) -> dict[str, str]:
        """
        Return a list of supported properties for this service.

        :return: A dictionary of supported properties, with the values describing each property.
        """
        return {}

    # SHARED METHODS.

    def annotate_annotations_with(self, annotator: Self) -> Self:
        """
        Construct a pipeline of annotators by sending the annotators from this annotator to the next annotator.
        Note that this does NOT mean that the original text is reannotated -- rather, each individual annotation
        will be annotated by the next annotator. This allows us to standardize some common situations:
        - If an annotation could not be annotated by the next annotator, it will be left as-is.
        - If an annotation is annotated by the next annotator with a single annotation, we will replace the previous
          annotation with the


        :param annotator:
        :return:
        """
