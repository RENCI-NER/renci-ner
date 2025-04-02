import abc
from dataclasses import dataclass, field
from typing import Self


@dataclass
class AnnotationProvenance:
    name: str
    url: str
    version: str


@dataclass
class Annotation:
    """
    A class for storing a single annotation.
    """

    text: str
    id: str
    label: str
    type: str
    start: int
    end: int
    provenances: list[AnnotationProvenance] = field(default_factory=list)
    based_on: list[Self] = field(default_factory=list)
    props: dict = field(default_factory=dict)


@dataclass
class NormalizedAnnotation(Annotation):
    curie: str = None
    biolink_type: str = None


class AnnotatedText:
    """
    A class for storing a text along with a set of annotations from a single source.
    """

    def __init__(self, text: str, annotations: list[Annotation]) -> None:
        self.text = text
        self.annotations = annotations

    def annotate_annotations_with(self, annotator: 'Annotator', props: dict = {}) -> Self:
        """
        Re-annotate the annotations in this AnnotatedText with another annotator.

        Note that this does NOT mean that the original text is reannotated -- rather, each individual annotation
        will be annotated by the next annotator. This allows us to standardize some common situations:
        - If an annotation could not be annotated by the next annotator, it will be left as-is.
        - If an annotation is annotated by the next annotator with a single annotation, we will replace the previous
          annotation with this annotation, but update provenance and based_on fields.
        - If an annotation is annotated by the next annotator with multiple annotations, we will replace this
          annotation with all of those annotations, with provenance and based_on fields updated appropriately.

        :param annotator: The other annotator to annotate these annotations with.
        :param props: A dictionary of properties to pass to the annotator.
        :return: AnnotatedText with annotations re-annotated with the other annotator.
        """

        new_annotations = []
        for annotation in self.annotations:
            annotated_text = annotator.annotate(annotation.text, props=props)
            reannotations = annotated_text.annotations
            if len(reannotations) == 0:
                # Leave the current annotation unchanged.
                new_annotations.append(annotation)
            else:
                # We have one or more annotations. So we need to update:
                # - the annotation provenances by adding this annotator to the end of that list.
                new_provenances = [*annotation.provenances, annotator.provenance]
                # - the based_on by adding annotation to the existing list.
                new_based_on = [*annotation.based_on, annotation]

                for reannotation in reannotations:
                    reannotation.provenances = new_provenances
                    reannotation.based_on = new_based_on
                    new_annotations.append(reannotation)

        return AnnotatedText(self.text, new_annotations)
