from dataclasses import dataclass


@dataclass
class Annotation:
    """
    A class for storing a single annotation.
    """
    text: str
    start: int
    end: int
    id: str
    label: str
    type: str
    props: dict


@dataclass
class NormalizedAnnotation (Annotation):
    curie: str
    biolink_type: str


@dataclass
class AnnotatedText:
    """
    A class for storing a text along with a set of annotations from a single source.
    """
    text: str
    annotations: list[Annotation]
