from dataclasses import dataclass

from src.renci_ner.Annotation import Annotation


@dataclass
class AnnotatedText:
    """
    A class for storing a text along with a set of annotations from a single source.
    """
    text: str
    annotations: list[Annotation]