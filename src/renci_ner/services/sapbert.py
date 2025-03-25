#
# A Named Entity Linker based on the Babel cliques using SAPBERT embeddings.
# Source code: https://github.com/RENCI-NER/sapbert
# Hosted at: https://sap-qdrant.apps.renci.org/docs
#
from renci_ner.annotations import AnnotatedText, Annotation
from renci_ner.services.core import Annotator

import requests

# Configuration.
RENCI_SAPBERT_URL = "https://sap-qdrant.apps.renci.org"
DEFAULT_LIMIT = 10


class SAPBERTAnnotator(Annotator):
    """
    Provides an Annotator interface to a SAPBERT service.
    """

    def __init__(self, url=RENCI_SAPBERT_URL, requests_session=requests.Session()):
        """
        Set up a SAPBERT service.

        :param url: The URL of a SAPBERT service.
        :param requests_session: A Requests session object to use instead of the default one.
        """
        self.url = url
        self.annotate_url = url + "/annotate/"
        self.requests_session = requests_session

    def supported_properties(self):
        return {
            "limit": "The maximum number of results to return.",
            "score": "The (minimum) score for this result returned by SAPBERT (higher is better)."
        }

    def annotate(self, text, props) -> Annotation | AnnotatedText:
        # Set up query.
        session = self.requests_session

        min_score = props.get("score", 0)
        limit = props.get("limit", DEFAULT_LIMIT)

        response = session.post(
            self.annotate_url,
            json={
                "text": text,
                "model_name": "sapbert",
                "count": limit,
            },
        )

        response.raise_for_status()

        results = response.json()

        # Find the first result that meets our criteria.
        annotations = []
        for result in results:
            if result.get("score", 0) < min_score:
                continue

            annotations.append(Annotation(
                text=text,
                id=result.get("curie", ""),
                label=result.get("name", ""),
                type=result.get("category", ""),
                props={
                    "score": result.get("score", 0),
                },
                # Since we're using the whole text, let's just use that
                # as the start/end.
                start=0,
                end=len(text),
            ))

        return AnnotatedText(text, annotations)
