from dataclasses import dataclass
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
    prov: list[AnnotationProvenance] = list
    prev: list[Self] = list
    props: dict = dict


@dataclass
class NormalizedAnnotation(Annotation):
    curie: str = None
    biolink_type: str = None


@dataclass
class AnnotatedText:
    """
    A class for storing a text along with a set of annotations from a single source.
    """

    text: str
    annotations: list[Annotation]
