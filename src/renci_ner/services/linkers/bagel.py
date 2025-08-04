#
# Bagel is an LLM-based combination linker developed at RENCI.
# Because Bagel is a re-ranker, it needs existing annotation possibilities to rerank, which can be delivered in
# two ways:
#   - annotate() finds text that has been annotated with multiple results
#       - Not implemented yet: we will need some method for combining the same annotated text with different
#         annotations into a single document of some kind. We'll need to do that anyway for doing benchmarking,
#         so let's wait a bit and do it right (see https://github.com/RENCI-NER/renci-ner/issues/6).
#   - annotate_with() re-annotates an AnnotatedText using the result from a list of annotators, after feeding them
#     into Bagel.
#
# Source code: https://github.com/RENCI-NER/bagel
# Hosted at: https://bagel.apps.renci.org/
#
import json
import logging
import os
import random

import requests
import webcolors
from requests.auth import HTTPBasicAuth

from renci_ner.core import (
    AnnotatedText,
    AnnotationProvenance,
    Annotator,
    NormalizedAnnotation, AnnotatorWithProps,
)
from renci_ner.services.normalization.nodenorm import NodeNorm

# Configuration.
RENCI_BAGEL_URL = "https://bagel.apps.renci.org"
BAGEL_PROMPT_NAME = "bagel/ask_classes_cot"
BAGEL_DEFAULT_TIMEOUT = 120

# Load BAGEL_USERNAME and BAGEL_PASSWORD from the environment.
BAGEL_USERNAME = os.environ.get("BAGEL_USERNAME")
BAGEL_PASSWORD = os.environ.get("BAGEL_PASSWORD")

class BagelAnnotator(Annotator):
    """
    Provides an Annotator interface to a BAGEL service.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="Bagel", url=RENCI_BAGEL_URL, version=self.openapi_version
        )

    def __init__(
        self, url=RENCI_BAGEL_URL, requests_session=requests.Session(), timeout=120
    ):
        """
        Set up a Bagel service.

        :param url: The URL of the Bagel service.
        :param requests_session: A Requests session object to use instead of the default one.
        :param timeout: The timeout to use for requests in seconds. Default: 120 seconds.
        """
        self.url = url
        self.rerank_url = url + "/group_synonyms_openai"
        self.requests_session = requests_session

        response = self.requests_session.get(
            self.url + "/openapi.json",
            auth=HTTPBasicAuth(BAGEL_USERNAME, BAGEL_PASSWORD),
            timeout=timeout
        )
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )

        # TODO: properly configure NodeNorm.
        self.nodenorm = NodeNorm()

    def supported_properties(self):
        """Configurable properties for Bagel."""
        return {
            "timeout": f"The timeout in seconds for requests to Bagel. Default: ${BAGEL_DEFAULT_TIMEOUT} seconds.",
            "bagel_prompt_name": "The name of the Bagel prompt to use. Default: '${BAGEL_PROMPT_NAME}'.",
        }

    def annotate_with(self, text: AnnotatedText, annotators: list[AnnotatorWithProps], props: dict = None) -> AnnotatedText:
        """
        Given an AnnotatedText, re-annotate it using the given list of AnnotatorWithProps objects.

        :param props: Properties to use with Bagel.
        :param text: An AnnotatedText containing annotations that need to be linked.
        :param annotators: A list of AnnotatorWithProps objects to use for re-annotation.
        :return: An AnnotatedText object containing the re-annotated annotations.
        """
        session = self.requests_session
        if props is None:
            props = {}
        timeout = props.get("timeout", BAGEL_DEFAULT_TIMEOUT)

        for ann in text.annotations:
            possible_matches = []
            colors_available = list(set(webcolors.names(spec=webcolors.CSS3)))

            # Run it through every annotator, and collect all the resulting matches.
            for annotator_with_props in annotators:
                annotator = annotator_with_props.annotator
                props = annotator_with_props.props

                result = annotator.annotate(ann.text, props)
                for result_ann in result.annotations:
                    identifier = result_ann.id
                    entity_type = result_ann.type
                    description = ""

                    normalized = self.nodenorm.normalize([identifier], {"description": True})
                    if identifier in normalized:
                        norm_result = normalized[identifier]
                        if "type" in norm_result:
                            entity_type = norm_result["type"][0]
                        if "id" in norm_result:
                            if "description" in norm_result["id"]:
                                description = norm_result["id"]["description"]

                    # Choose a color.
                    selected_color = random.sample(colors_available, 1)[0]
                    colors_available.remove(selected_color)

                    possible_matches.append({
                        "label": result_ann.label,
                        "identifier": result_ann.id,
                        "description": description,
                        "entity_type": entity_type,
                        "color_code": selected_color,
                        # TODO: implement taxa
                        "taxa": "",
                        "taxa_ids": [],
                    })

            # Query Bagel.
            request_json = {
                "prompt_name": props.get("bagel_prompt_name", BAGEL_PROMPT_NAME),
                "text": text.text,      # TODO: We currently give the full text as context, but in the future
                # we'll probably want to limit it to +/- 3 sentences or so.
                "entity": ann.text,
                "synonyms": possible_matches,
                "config": {
                    "llm_model_name": "google/gemma-3-12b-it",
                    "organization": "",
                    "access_key": "",
                    "url": "http://vllm-server/v1",
                    "llm_model_args": {
                        "top_p": 0.1,
                        "temperature": 0
                    }
                }
            }
            logging.warning(f"Bagel request: {json.dumps(request_json, indent=2)}")
            response = session.post(
                self.rerank_url,
                json=request_json,
                # TODO: make this more configurable.
                auth=HTTPBasicAuth(BAGEL_USERNAME, BAGEL_PASSWORD),
                timeout=timeout,
            )

            response.raise_for_status()
            result = response.json()
            raise RuntimeError(f"Bagel result: {json.dumps(result, indent=2)}")


    def annotate(self, text, props=None) -> AnnotatedText:
        """
        Annotate text using BabelSAPBERT.

        TODO: needs to be completed rewritten.

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
