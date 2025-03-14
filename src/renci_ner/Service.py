from src.renci_ner.AnnotationText import AnnotatedText


class Service:
    """
    An interface for communicating with an NER service.
    """

    def annotate(self, text: str, props: dict) -> AnnotatedText:
        """
        Annotate a text with a set of service-specific properties.
        """
        pass
