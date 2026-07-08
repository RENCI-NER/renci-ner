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
import functools
import json
import logging
import os
from dataclasses import dataclass

import requests
from requests import HTTPError
from requests.auth import HTTPBasicAuth

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    AnnotationProvenance,
    Annotator,
    AnnotatorWithProps,
    NormalizedAnnotation,
)
from renci_ner.services.normalization.nodenorm import NodeNorm
from renci_ner.utils import log_http_403_errors

# Configuration.
RENCI_BAGEL_URL = "https://bagel.apps.renci.org"
BAGEL_PROMPT_NAME = "bagel/ask_classes"
DEFAULT_LIMIT = 10
BAGEL_DEFAULT_TIMEOUT = 120
DEFAULT_TOP_P = 0.5
DEFAULT_TEMPERATURE = 1.0

# Load BAGEL_USERNAME and BAGEL_PASSWORD from the environment.
BAGEL_USERNAME = os.environ.get("BAGEL_USERNAME")
BAGEL_PASSWORD = os.environ.get("BAGEL_PASSWORD")


# A case class for uniquifying Bagel results.
@dataclass(frozen=True)
class BagelResult:
    label: str = ""
    identifier: str = ""
    description: str = ""
    entity_type: str = ""
    taxa: str = ""
    taxa_ids: str = ""
    synonym_type: str = ""

    @staticmethod
    def from_dict(d):
        # Taxa_ids is a list, which isn't hashable. We turn it into a string so we can hash them.
        taxa_ids = d.get("taxa_ids", [])
        taxa_ids_str = "||".join(taxa_ids)

        return BagelResult(
            label=d.get("label", ""),
            identifier=d.get("identifier", ""),
            description=d.get("description", ""),
            entity_type=d.get("entity_type", ""),
            taxa=d.get("taxa", ""),
            taxa_ids=taxa_ids_str,
            synonym_type=d.get("synonym_type", ""),
        )

    def to_dict(self):
        return {
            "label": self.label,
            "identifier": self.identifier,
            "description": self.description,
            "entity_type": self.entity_type,
            "taxa": self.taxa,
            "taxa_ids": self.taxa_ids.split("||"),
            "synonym_type": self.synonym_type,
        }

    @staticmethod
    def get_bagel_sort_key(br) -> tuple:
        if not isinstance(br, BagelResult):
            raise TypeError(
                f"get_synonym_type_order_key({br}) called, but we can only work with BagelResult objects, got {type(br)} instead."
            )

        if not br:
            return 999, 999

        # We need to sort in two ways:
        # - Bagel results are marked as exact, broad, narrow or related. We want to sort exact matches first, followed by the others.
        # - If we have multiple matches in a category, we want to figure out some way of choosing one.
        #   - For now, let's just demote UMLS.
        synonym_type_order = 0
        identical_synonym_type_order = 0

        if br.synonym_type == "exact":
            synonym_type_order = 1
        elif br.synonym_type in {"broad", "narrow"}:
            synonym_type_order = 2
        elif br.synonym_type == "related":
            synonym_type_order = 3
        else:
            raise RuntimeError(f"Unknown synonym type found in {br}: {br.synonym_type}")

        curie_prefix = br.identifier.split(":", 2)[0].upper()
        if curie_prefix == "UMLS":
            identical_synonym_type_order = 999
        else:
            identical_synonym_type_order = 0

        return synonym_type_order, identical_synonym_type_order


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
            timeout=timeout,
        )
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )
        self.logger = logging.getLogger(str(self))

        # TODO: properly configure NodeNorm.
        self.nodenorm = NodeNorm()

    def supported_properties(self):
        """Configurable properties for Bagel."""
        return {
            "timeout": f"The timeout in seconds for requests to Bagel. Default: ${BAGEL_DEFAULT_TIMEOUT} seconds.",
            "bagel_prompt_name": "The name of the Bagel prompt to use. Default: '${BAGEL_PROMPT_NAME}'.",
            "temperature": f"The temperature to use for the LLM (default: {DEFAULT_TEMPERATURE}).",
            "top_p": f"The top_p to use for the LLM (default: {DEFAULT_TOP_P}).",
            "limit": "The maximum number of results to return.",
        }

    def annotate_with(
        self,
        text: AnnotatedText,
        annotators: list[AnnotatorWithProps],
        bagel_props: dict = None,
    ) -> AnnotatedText:
        """
        Given an AnnotatedText, re-annotate it using the given list of AnnotatorWithProps objects.

        :param bagel_props: Properties to use with Bagel.
        :param text: An AnnotatedText containing annotations that need to be linked.
        :param annotators: A list of AnnotatorWithProps objects to use for re-annotation.
        :return: An AnnotatedText object containing the re-annotated annotations.
        """
        if bagel_props is None:
            bagel_props = {}
        timeout = bagel_props.get("timeout", BAGEL_DEFAULT_TIMEOUT)
        limit = bagel_props.get("limit", DEFAULT_LIMIT)
        nodenorm_props = {"description": True, "timeout": timeout}

        # --- Pass 1: run all annotators, collect results, accumulate identifiers ---
        per_ann_results = []   # list of (ann, [(result_ann, annotator_with_props), ...])
        all_identifiers = set()

        for index, ann in enumerate(text.annotations):
            self.logger.debug(
                f"Annotating '{ann.text}' with Bagel ({index}/{len(text.annotations)})"
            )
            ann_matches = []
            for annotator_with_props in annotators:
                annotator = annotator_with_props.annotator
                annotator_props = annotator_with_props.props
                result = annotator.annotate(ann.text, annotator_props)
                for result_ann in result.annotations:
                    ann_matches.append((result_ann, annotator_with_props))
                    all_identifiers.add(result_ann.id)
            per_ann_results.append((ann, ann_matches))

        # --- Single bulk NodeNorm call for all collected identifiers ---
        normalized = self.nodenorm.normalize(list(all_identifiers), nodenorm_props)

        # --- Pass 2: build BagelResults and query Bagel ---
        output_annotations = []
        for ann, ann_matches in per_ann_results:
            if not ann_matches:
                output_annotations.append(ann)
                continue

            possible_matches = set()
            for result_ann, annotator_with_props in ann_matches:
                identifier = result_ann.id
                entity_type = result_ann.type
                description = ""

                norm_result = normalized.get(identifier)
                if norm_result is not None:
                    if "type" in norm_result:
                        entity_type = norm_result["type"][0]
                    if "id" in norm_result and "description" in norm_result:
                        description = norm_result["description"]

                possible_matches.add(
                    BagelResult(
                        label=result_ann.label,
                        identifier=identifier,
                        description=description,
                        entity_type=entity_type,
                        # TODO: implement taxa
                        #   - Should include this for genes and proteins for NameRes
                        #   - Might be worth putting in a default, but probably not needed.
                        taxa="",
                        taxa_ids="",
                    )
                )

            self.logger.debug(
                f"Found {len(possible_matches)} possible matches for '{ann.text}'."
            )

            self.logger.debug(
                f"Querying Bagel for '{ann.text}'."
            )
            unique_bagel_results = self.query_bagel(
                ann.text,
                text.text,
                tuple(possible_matches),
                json.dumps(bagel_props, sort_keys=True),
            )

            # Update annotation with Bagel results.
            result_count = 0
            for bagel_result in unique_bagel_results:
                if result_count >= limit:
                    break
                new_based_on = list(ann.based_on)
                new_based_on.append(ann)
                # This is almost certainly a NormalizedAnnotation, but we don't know for sure.
                output_annotations.append(
                    Annotation(
                        text=ann.text,
                        start=ann.start,
                        end=ann.end,
                        id=bagel_result.identifier,
                        label=bagel_result.label,
                        type=bagel_result.entity_type,
                        props={
                            "description": bagel_result.description,
                        },
                        based_on=new_based_on,
                        provenance=self.provenance,
                    )
                )
                result_count += 1

        return AnnotatedText(text.text, output_annotations)

    @functools.lru_cache(maxsize=10_000)
    def query_bagel(
        self,
        entity_text: str,
        context_text: str,
        possible_matches: tuple[BagelResult],
        props_json: str,
    ) -> list[BagelResult]:
        """
        Query Bagel.

        :param entity_text: The entity text to query for.
        :param context_text: The context text that the entity text is found in.
        :param possible_matches: The possible matches to query for (we store these as a tuple of BagelResult objects).
        :param props_json: Properties to use with Bagel. This is really a dictionary, but we turn it into a JSON string for memoization.
        :return: A list of BagelResult objects.
        """

        session = self.requests_session
        props = json.loads(props_json)
        timeout = props.get("timeout", BAGEL_DEFAULT_TIMEOUT)

        request_json = {
            "prompt_name": props.get("bagel_prompt_name", BAGEL_PROMPT_NAME),
            "context": {
                # TODO: We currently give the full text as context, but in the future
                # we'll probably want to limit it to +/- 3 sentences or so.
                "text": context_text,
                "entity": entity_text,
                "synonyms": list(map(lambda x: x.to_dict(), possible_matches)),
            },
            "config": {
                "llm_model_name": "google/gemma-3-12b-it",
                "organization": "",
                "access_key": "",
                "url": "http://vllm-server/v1",
                "llm_model_args": {
                    "top_p": props.get("top_p", DEFAULT_TOP_P),
                    "temperature": props.get("temperature", DEFAULT_TEMPERATURE),
                },
            },
        }
        self.logger.debug(f"Bagel request: {json.dumps(request_json, indent=2)}")
        response = session.post(
            self.rerank_url,
            json=request_json,
            # TODO: make this more configurable.
            auth=HTTPBasicAuth(BAGEL_USERNAME, BAGEL_PASSWORD),
            timeout=timeout,
        )

        # 403 errors probably mean that the RENCI Ingress is catching something it shouldn't.
        # Don't throw an error here, just log it and move on.
        if response.status_code == 403:
            log_http_403_errors(
                entity_text + "\n" + context_text,
                self.rerank_url,
                request_json,
                logger=self.logger,
            )
            return []

        if not response.ok:
            raise HTTPError(
                f"Bagel request failed with error {response.status_code} {response.text}: {json.dumps(request_json, indent=2)}"
            )

        result = response.json()
        self.logger.debug(
            f"Bagel result: {json.dumps(result, indent=2, sort_keys=True)}"
        )

        # The result here is a list of results, but they're not guaranteed to be sorted: only one of them should have
        # `"synonym_type": "exact"`, which should be sorted first. There are other narrow/broad matches that should
        # sort later.
        bagel_results = sorted(
            map(lambda x: BagelResult.from_dict(x), result),
            key=BagelResult.get_bagel_sort_key,
        )
        self.logger.debug(
            f"Bagel results: {json.dumps(list(map(lambda r: r.to_dict(), bagel_results)), indent=2, sort_keys=True)}"
        )

        unique_bagel_results = []
        # Generate a list of unique Bagel results, preserving the original order.
        seen = set()
        for bagel_result in bagel_results:
            if bagel_result not in seen:
                unique_bagel_results.append(bagel_result)
                seen.add(bagel_result)
        return unique_bagel_results

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

        data = {
            "text": text,
            "model_name": "sapbert",
            "count": limit,
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
