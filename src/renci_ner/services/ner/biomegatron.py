#
# An annotator service to identify biomedical concepts in plain text.
# Source code: https://github.com/RENCI-NER/nemo-serve
# Hosted at: https://med-nemo.apps.renci.org/docs
#
import logging

import requests
from cachetools import LRUCache

from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance, Annotator
from renci_ner.utils import log_http_403_errors

# Configuration.
RENCI_BIOMEGATRON_URL = "https://med-nemo.apps.renci.org"


class BioMegatron(Annotator):
    """
    Provides an Annotator interface to a BioMegatron service.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="BioMegatron", url=RENCI_BIOMEGATRON_URL, version=self.openapi_version
        )

    def __init__(
        self,
        url=RENCI_BIOMEGATRON_URL,
        requests_session=requests.Session(),
        timeout=120,
    ):
        """
        Set up a BioMegatron service.

        :param url: The URL of the BioMegatron service.
        :param requests_session: A Requests session object to use instead of the default one.
        :param timeout: The timeout to use for requests in seconds. Default: 120 seconds.
        """
        self.url = url
        self.annotate_url = url + "/annotate/"
        self.requests_session = requests_session

        result = self.requests_session.get(self.url + "/openapi.json", timeout=timeout)
        result.raise_for_status()
        openapi_data = result.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )
        self.logger = logging.getLogger(str(self))

        # Set up a cache.
        self.cache = LRUCache(maxsize=10_000)

    def supported_properties(self):
        """Some configurable parameters for BioMegatron (none at present)."""
        return {
            "timeout": "The timeout in seconds for requests to BioMegatron. Default: 120 seconds.",
            "skip_cache": "Do not use the cache (default: FALSE)",
        }

    def annotate(self, text: str, props: dict = None) -> AnnotatedText:
        """
        Annotate text using BioMegatron.

        :param text: Text to annotate.
        :param props: Properties to pass to BioMegatron.
        :return: An AnnotatedText object containing the annotations.
        """

        if props is None:
            props = {}

        flag_skip_cache = False
        if "skip_cache" in props and props["skip_cache"]:
            flag_skip_cache = True

        if not flag_skip_cache and text in self.cache:
            return self.cache[text]

        session = self.requests_session
        timeout = props.get("timeout", 120)

        data = {
            "text": text,
            "model_name": "token_classification",
        }
        response = session.post(
            self.annotate_url,
            json=data,
            timeout=timeout,
        )

        if response.status_code == 403:
            log_http_403_errors(text, self.annotate_url, data, logger=self.logger)
            return AnnotatedText(text, [])

        response.raise_for_status()
        result = response.json()

        annotations = []
        for denotation in result.get("denotations", []):
            span = denotation.get("span", {})
            start_index = span.get("begin", -1)
            end_index = span.get("end", -1)

            annotations.append(
                Annotation(
                    text=denotation.get("text", ""),
                    start=start_index,
                    end=end_index,
                    id=denotation.get("id", ""),
                    label="",
                    type=denotation.get("obj", ""),
                    props={},
                    provenance=self.provenance,
                )
            )

        final_result = AnnotatedText(text, annotations)
        if not flag_skip_cache:
            self.cache[text] = final_result

        return final_result
