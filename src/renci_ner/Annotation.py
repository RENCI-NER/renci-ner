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


@dataclass
class NormalizedAnnotation (Annotation):
    curie: str
    biolink_type: str
