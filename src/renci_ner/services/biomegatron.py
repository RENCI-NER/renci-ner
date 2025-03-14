#
# An annotator service to identify biomedical concepts in plain text.
# Source code: https://github.com/RENCI-NER/nemo-serve
# Hosted at: https://med-nemo.apps.renci.org/docs
#
from src.renci_ner.annotations import AnnotatedText, Annotation
from src.renci_ner.services.core import Annotator

import requests

# Configuration.
RENCI_BIOMEGATRON_URL = "https://med-nemo.apps.renci.org"


class BioMegatron (Annotator):
    """
    Provides an Annotator interface to a BioMegatron service.
    """

    def __init__(self, url=RENCI_BIOMEGATRON_URL, requests_session=requests.Session()):
        """
        Set up a BioMegatron service.

        :param url: The URL of the BioMegatron service.
        :param requests_session: A Requests session object to use instead of the default one.
        """
        self.url = url
        self.annotate_url = url + "/annotate"
        self.requests_session = requests_session

    def supported_properties(self):
        return {}

    def annotate(self, text, props) -> AnnotatedText:
        # Set up query.
        session = self.requests_session

        response = session.post(self.annotate_url, json={
            "text": text,
            "model_name": "token_classification",
        })

        response.raise_for_status()

        result = response.json()

        annotations = []
        for denotation in result.get("denotations", []):
            span = denotation.get('span', {})
            start_index = span.get('begin', -1)
            end_index = span.get('end', -1)

            annotations.append(Annotation(
                text = denotation.get('text', ''),
                start = start_index,
                end = end_index,
                id = denotation.get('id', ''),
                label = '',
                type = denotation.get('obj', ''),
                props = {}
            ))

        return AnnotatedText(text, annotations)