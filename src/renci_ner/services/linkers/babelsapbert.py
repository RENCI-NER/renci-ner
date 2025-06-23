#
# A Named Entity Linker based on the Babel cliques using SAPBERT embeddings.
# Source code: https://github.com/RENCI-NER/sapbert
# Hosted at: https://sap-qdrant.apps.renci.org/docs
#
import requests

from renci_ner.core import (
    AnnotatedText,
    AnnotationProvenance,
    Annotator,
    NormalizedAnnotation,
)

# Configuration.
RENCI_SAPBERT_URL = "https://sap-qdrant.apps.renci.org"
DEFAULT_LIMIT = 10


class BabelSAPBERTAnnotator(Annotator):
    """
    Provides an Annotator interface to a SAPBERT service.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="BabelSAPBERT", url=RENCI_SAPBERT_URL, version=self.openapi_version
        )

    def __init__(
        self, url=RENCI_SAPBERT_URL, requests_session=requests.Session(), timeout=120
    ):
        """
        Set up a SAPBERT service.

        :param url: The URL of a SAPBERT service.
        :param requests_session: A Requests session object to use instead of the default one.
        :param timeout: The timeout to use for requests in seconds. Default: 120 seconds.
        """
        self.url = url
        self.annotate_url = url + "/annotate/"
        self.requests_session = requests_session

        response = self.requests_session.get(
            self.url + "/openapi.json", timeout=timeout
        )
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )

    def supported_properties(self):
        """Configurable properties for SAPBERT."""
        return {
            "timeout": "The timeout in seconds for requests to SAPBERT. Default: 120 seconds.",
            "limit": "The maximum number of results to return.",
            "score": "The minimum score for this result returned by SAPBERT (higher is better).",
        }

    def annotate(self, text, props=None) -> AnnotatedText:
        """
        Annotate text using BabelSAPBERT.

        :param text: The text to annotate.
        :param props: The properties to pass to SAPBERT.
        :return: An AnnotatedText object containing the annotations.
        """
        if props is None:
            props = {}

        session = self.requests_session
        timeout = props.get("timeout", 120)

        min_score = props.get("score", 0)
        limit = props.get("limit", DEFAULT_LIMIT)

        response = session.post(
            self.annotate_url,
            json={
                "text": text,
                "model_name": "sapbert",
                "count": limit,
            },
            timeout=timeout,
        )

        response.raise_for_status()
        results = response.json()

        # Find all the results that meet our criteria.
        annotations = []
        for result in results:
            if result.get("score", 0) < min_score:
                continue

            annotations.append(
                # Since SAPBERT is normalized to Babel, we can treat it as a NormalizedAnnotation.
                NormalizedAnnotation(
                    text=text,
                    id=result.get("curie", ""),
                    label=result.get("name", ""),
                    biolink_type=result.get("category", ""),
                    type=result.get("category", ""),
                    props={
                        "score": result.get("score", 0),
                    },
                    provenance=self.provenance,
                    # Since we're using the whole text, let's just use that
                    # as the start/end.
                    start=0,
                    end=len(text),
                )
            )

        return AnnotatedText(text, annotations)
