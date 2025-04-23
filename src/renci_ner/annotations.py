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

    def __setattr__(self, name, value):
        """Validate biolink_type."""
        if (
            name == "biolink_type"
            and value is not None
            and not value.lower().startswith("biolink:")
        ):
            raise ValueError(
                f"Invalid biolink_type: must start with 'biolink:' but got '{value}'."
            )

        # TODO: it'd probably be a good idea to check formatting for CURIEs as well, but that's less well defined.

        # No problems.
        super().__setattr__(name, value)

    @classmethod
    def from_annotation(
        cls, annotation: Annotation, curie=None, biolink_type=None, label=None
    ) -> Self:
        """
        Creates an instance of NormalizedAnnotation from the provided Annotation object.

        This method serves as a factory method to instantiate a NormalizedAnnotation
        object using the given Annotation instance. Additional optional properties,
        such as curie and biolink_type, may also be provided to customize the
        generated object. The method utilizes the attributes from the Annotation
        instance to populate the corresponding fields in the NormalizedAnnotation.

        :param annotation: The Annotation instance used as the basis for constructing
            the NormalizedAnnotation object.
        :type annotation: Annotation
        :param curie: Optional CURIE string for additional annotation details. If not
            provided, it defaults to None.
        :type curie: str, optional
        :param biolink_type: Optional biolink type string to specify the type of the
            annotation. If not provided, it defaults to None.
        :type biolink_type: str, optional
        :param label: Optional label string to specify the label of the annotation. Will overwrite the
            annotation label if one is provided.
        :type label: str, optional
        :return: A new instance of NormalizedAnnotation initialized with the provided
            annotation and optional parameters.
        :rtype: Self
        """

        if label is None:
            annotation_label = annotation.label
        else:
            annotation_label = label

        return NormalizedAnnotation(
            text=annotation.text,
            id=annotation.id,
            label=annotation_label,
            type=annotation.type,
            start=annotation.start,
            end=annotation.end,
            provenances=annotation.provenances,
            based_on=annotation.based_on,
            props=annotation.props,
            curie=curie,
            biolink_type=biolink_type,
        )


class AnnotatedText:
    """
    A class for storing a text along with a set of annotations from a single source.
    """

    def __init__(self, text: str, annotations: list[Annotation]) -> None:
        self.text = text
        self.annotations = annotations

    def annotate_annotations_with(
        self, annotator: "Annotator", props: dict = {}
    ) -> Self:
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
