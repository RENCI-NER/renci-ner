#
# Bagel is an LLM-based re-ranker developed at RENCI. It is a Transformer: given an
# AnnotatedText whose spans carry several linked candidates (e.g. from a MultiAnnotator
# of NameRes and SAPBERT), it asks Bagel to pick the best candidate for each span.
#
#     Pipeline(BioMegatron(), MultiAnnotator(sapbert, nameres), (BagelAnnotator(), {"limit": 1}))
#
# Source code: https://github.com/RENCI-NER/bagel
# Hosted at: https://bagel.apps.renci.org/
#
import functools
import json
import logging
import os
from dataclasses import dataclass, replace

import requests
from requests.auth import HTTPBasicAuth

from renci_ner.core import (
    AnnotatedText,
    AnnotationProvenance,
    NormalizedAnnotation,
    Transformer,
)
from renci_ner.services.normalization.nodenorm import NodeNorm
from renci_ner.utils import forbidden

# Configuration.
RENCI_BAGEL_URL = "https://bagel.apps.renci.org"
BAGEL_PROMPT_NAME = "bagel/ask_classes"
DEFAULT_LIMIT = 10
BAGEL_DEFAULT_TIMEOUT = 120
DEFAULT_TOP_P = 0.5
DEFAULT_TEMPERATURE = 1.0

logger = logging.getLogger(__name__)


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
            # Anything else the LLM comes up with sorts last rather than crashing.
            logger.warning(f"Unknown synonym type in {br}: {br.synonym_type}")
            synonym_type_order = 4

        curie_prefix = br.identifier.split(":", 2)[0].upper()
        if curie_prefix == "UMLS":
            identical_synonym_type_order = 999
        else:
            identical_synonym_type_order = 0

        return synonym_type_order, identical_synonym_type_order


class BagelAnnotator(Transformer):
    """
    Re-ranks linked candidates on an AnnotatedText using a BAGEL service.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="Bagel", url=RENCI_BAGEL_URL, version=self.openapi_version
        )

    def __str__(self):
        return f"BagelAnnotator(url={self.url}, version={self.openapi_version})"

    def __init__(
        self,
        url=RENCI_BAGEL_URL,
        requests_session=None,
        timeout=120,
        nodenorm=None,
        username=None,
        password=None,
    ):
        """
        Set up a Bagel service.

        :param url: The URL of the Bagel service.
        :param requests_session: A Requests session object to use instead of the default one.
        :param timeout: The timeout to use for requests in seconds. Default: 120 seconds.
        :param nodenorm: The NodeNorm instance to use to look up descriptions and types for
            candidates. Defaults to a NodeNorm sharing this service's requests session.
        :param username: Bagel username; defaults to the BAGEL_USERNAME environment variable.
        :param password: Bagel password; defaults to the BAGEL_PASSWORD environment variable.
        """
        username = username or os.environ.get("BAGEL_USERNAME")
        password = password or os.environ.get("BAGEL_PASSWORD")
        if not username or not password:
            raise ValueError(
                "Bagel needs credentials: pass username/password or set BAGEL_USERNAME and BAGEL_PASSWORD."
            )
        self.auth = HTTPBasicAuth(username, password)

        self.url = url
        self.rerank_url = url + "/group_synonyms_openai"
        self.requests_session = requests_session or requests.Session()

        response = self.requests_session.get(
            self.url + "/openapi.json", auth=self.auth, timeout=timeout
        )
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )

        self.nodenorm = nodenorm or NodeNorm(
            requests_session=self.requests_session, timeout=timeout
        )

    def supported_properties(self):
        """Configurable properties for Bagel."""
        return {
            "timeout": f"The timeout in seconds for requests to Bagel. Default: {BAGEL_DEFAULT_TIMEOUT} seconds.",
            "bagel_prompt_name": f"The name of the Bagel prompt to use. Default: '{BAGEL_PROMPT_NAME}'.",
            "temperature": f"The temperature to use for the LLM (default: {DEFAULT_TEMPERATURE}).",
            "top_p": f"The top_p to use for the LLM (default: {DEFAULT_TOP_P}).",
            "limit": "The maximum number of results to return.",
        }

    def transform(
        self, annotated_text: AnnotatedText, props: dict = None
    ) -> AnnotatedText:
        """
        For each span in the text, ask Bagel to choose among the linked candidates at that span.

        Candidates are the NormalizedAnnotations sharing a start/end; up to `limit` of Bagel's
        choices replace them, each based_on the candidate it came from. Spans with no linked
        candidates (e.g. an NER annotation no linker matched) pass through unchanged.

        :param annotated_text: An AnnotatedText with candidate annotations to re-rank.
        :param props: Properties to use with Bagel (see supported_properties).
        :return: An AnnotatedText with Bagel's choices.
        """
        if props is None:
            props = {}
        timeout = props.get("timeout", BAGEL_DEFAULT_TIMEOUT)
        limit = props.get("limit", DEFAULT_LIMIT)

        spans: dict[tuple[int, int], list] = {}
        for ann in annotated_text.annotations:
            spans.setdefault((ann.start, ann.end), []).append(ann)

        # One NodeNorm call for every candidate in the text, for descriptions and types.
        candidate_ids = {
            ann.id
            for group in spans.values()
            for ann in group
            if isinstance(ann, NormalizedAnnotation)
        }
        normalized = self.nodenorm.normalize(
            sorted(candidate_ids), {"description": True, "timeout": timeout}
        )

        output_annotations = []
        for group in spans.values():
            candidates = [a for a in group if isinstance(a, NormalizedAnnotation)]
            if not candidates:
                output_annotations.extend(group)
                continue

            possible_matches = {
                self._bagel_result(candidate, normalized.get(candidate.id))
                for candidate in candidates
            }
            logger.debug(
                f"Querying Bagel for '{candidates[0].text}' with {len(possible_matches)} candidates."
            )
            bagel_results = self.query_bagel(
                candidates[0].text,
                annotated_text.text,
                tuple(possible_matches),
                json.dumps(props, sort_keys=True),
            )

            by_id = {candidate.id: candidate for candidate in candidates}
            for bagel_result in bagel_results[:limit]:
                winner = by_id.get(bagel_result.identifier)
                if winner is None:
                    # Bagel returned an identifier we didn't send; keep the NER chain only.
                    winner = candidates[0]
                    based_on = list(winner.based_on)
                else:
                    based_on = [*winner.based_on, winner]
                output_annotations.append(
                    NormalizedAnnotation(
                        text=winner.text,
                        start=winner.start,
                        end=winner.end,
                        id=bagel_result.identifier,
                        label=bagel_result.label,
                        type=bagel_result.entity_type,
                        biolink_type=bagel_result.entity_type,
                        props={"description": bagel_result.description},
                        based_on=based_on,
                        provenance=self.provenance,
                    )
                )

        return replace(annotated_text, annotations=output_annotations)

    @staticmethod
    def _bagel_result(candidate: NormalizedAnnotation, norm_result: dict | None):
        """Build the candidate description Bagel expects, using NodeNorm's type and description."""
        entity_type = candidate.biolink_type
        description = ""
        if norm_result:
            if norm_result.get("type"):
                entity_type = norm_result["type"][0]
            description = norm_result.get("id", {}).get("description", "")
        return BagelResult(
            label=candidate.label,
            identifier=candidate.id,
            description=description,
            entity_type=entity_type,
            # TODO: implement taxa (NameRes has them for genes and proteins).
            taxa="",
            taxa_ids="",
        )

    @functools.cache
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
        logger.debug(f"Bagel request: {json.dumps(request_json, indent=2)}")
        response = session.post(
            self.rerank_url, json=request_json, auth=self.auth, timeout=timeout
        )
        if forbidden(response, entity_text, request_json):
            return []

        result = response.json()
        logger.debug(f"Bagel result: {json.dumps(result, indent=2, sort_keys=True)}")

        # The result here is a list of results, but they're not guaranteed to be sorted: only one of them should have
        # `"synonym_type": "exact"`, which should be sorted first. There are other narrow/broad matches that should
        # sort later.
        bagel_results = sorted(
            map(lambda x: BagelResult.from_dict(x), result),
            key=BagelResult.get_bagel_sort_key,
        )
        logger.debug(
            f"Bagel results: {json.dumps(list(map(lambda r: r.to_dict(), bagel_results)), indent=2, sort_keys=True)}"
        )

        unique_bagel_results = []
        # Generate a list of unique Bagel results, preserving the original order.
        unique_bagel_results_set = {}
        for bagel_result in bagel_results:
            if bagel_result not in unique_bagel_results_set:
                unique_bagel_results.append(bagel_result)
                unique_bagel_results_set[bagel_result] = True
        return unique_bagel_results
