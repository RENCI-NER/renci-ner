from renci_ner.annotations import AnnotatedText


class Annotator:
    """
    An interface for annotating text using a service.
    """

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
