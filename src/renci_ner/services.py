from src.renci_ner.annotations import AnnotatedText


class Annotator:
    """
    An interface for annotating text using a service.
    """

    def annotate(self, text: str, props: dict) -> AnnotatedText:
        """
        Annotate a text with a set of service-specific properties.
        """
        pass
