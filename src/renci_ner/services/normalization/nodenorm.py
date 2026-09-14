#
# The Translator Node Normalizer as a Transformer.
# Source code: https://github.com/TranslatorSRI/NodeNormalization
# Hosted at: https://nodenormalization-sri.renci.org/
#
from dataclasses import replace

import requests

from renci_ner.core import (
    AnnotatedText,
    AnnotationProvenance,
    NormalizedAnnotation,
    Transformer,
)
from renci_ner.utils import forbidden

# Configuration.
RENCI_NODENORM_URL = "https://nodenormalization-sri.renci.org"
NODENORM_DEFAULT_TIMEOUT = 120


class NodeNorm(Transformer):
    """
    The Translator Node Normalizer as a Transformer.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="NodeNorm", url=RENCI_NODENORM_URL, version=self.openapi_version
        )

    def __str__(self):
        return f"NodeNorm(url={self.url}, version={self.openapi_version})"

    def __init__(self, url=RENCI_NODENORM_URL, requests_session=None, timeout=120):
        """
        Set up a NodeNorm service.

        :param url: The URL of the NodeNorm service.
        :param requests_session: A Requests session object to use instead of the default one.
        :param timeout: The timeout to use for requests in seconds. Default: 120 seconds.
        """
        self.url = url
        self.get_normalized_nodes_url = url + "/get_normalized_nodes"
        self.requests_session = requests_session or requests.Session()

        response = self.requests_session.get(
            self.url + "/openapi.json", timeout=timeout
        )
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )

        # ponytail: (identifier, flags) -> result, emptied when full. Swap for an LRU
        # if that ever matters.
        self._cache = {}
        self.cache_size = 100_000

    def supported_properties(self):
        """Some configurable parameters."""
        return {
            "timeout": f"The timeout in seconds for requests to NodeNorm. Default: {NODENORM_DEFAULT_TIMEOUT} seconds.",
            "geneprotein_conflation": "(true/false, default: true) Whether to conflate gene and protein identifiers.",
            "drugchemical_conflation": "(true/false, default: false) Whether to conflate drug and chemical identifiers.",
            "description": "(true/false, default: false) Whether to include descriptions in the response.",
            "skip_cache": "(true/false, default: false) Bypass the in-memory cache for this call.",
        }

    def normalize(self, identifiers: list[str], props=None):
        """
        Normalize a list of identifiers using NodeNorm.

        :param identifiers: A list of identifiers to normalize.
        :param props: Properties to use when normalizing. See supported_properties.
        :return: Output from NodeNorm.
        """
        if props is None:
            props = {}
        session = self.requests_session
        timeout = props.get("timeout", NODENORM_DEFAULT_TIMEOUT)
        flags = (
            props.get("geneprotein_conflation", True),
            props.get("drugchemical_conflation", False),
            props.get("description", False),
        )
        skip_cache = props.get("skip_cache", False)

        results = {}
        if not skip_cache:
            results = {
                identifier: self._cache[(identifier, flags)]
                for identifier in identifiers
                if (identifier, flags) in self._cache
            }
        missing = sorted(set(identifiers) - results.keys())
        if not missing:
            # Also covers the empty list, which NodeNorm rejects.
            return results

        data = {
            "curies": missing,
            "conflate": flags[0],
            "drug_chemical_conflate": flags[1],
            "description": flags[2],
        }
        response = session.post(
            self.get_normalized_nodes_url, json=data, timeout=timeout
        )
        if forbidden(response, missing, data):
            return results
        fetched = response.json()
        if not skip_cache:
            if len(self._cache) + len(fetched) > self.cache_size:
                self._cache.clear()
            for identifier, result in fetched.items():
                self._cache[(identifier, flags)] = result
        return results | fetched

    def transform(self, annotated_text: AnnotatedText, props=None) -> AnnotatedText:
        """
        Transform an AnnotatedText object using NodeNorm. For every annotation, we pass the IDs to NodeNorm, and if
        it changes the identifier, we would return a NormalizedAnnotation. Otherwise, we return the original
        annotation.

        :param annotated_text: The annotated text to transform.
        :param props: Properties to pass to NodeNorm (see supported_properties).
        :return: The AnnotatedText with normalized annotations where possible.
        """
        if props is None:
            props = {}

        ids = list(set(map(lambda a: a.id, annotated_text.annotations)))
        results = self.normalize(ids, props=props)

        output_annotations = []
        for annotation in annotated_text.annotations:
            # No result?
            if annotation.id not in results or results[annotation.id] is None:
                output_annotations.append(annotation)
                continue

            # We have a result!
            result = results[annotation.id]
            if (
                "id" not in result
                or "identifier" not in result["id"]
                or not result["id"]["identifier"]
            ):
                # No identifier, skip.
                output_annotations.append(annotation)
                continue

            if (
                isinstance(annotation, NormalizedAnnotation)
                and result["id"]["identifier"] == annotation.id
            ):
                # Already normalized, skip.
                output_annotations.append(annotation)
                continue

            types = result["type"]
            if not types:
                types = ["biolink:NamedThing"]

            normalized_annotation = NormalizedAnnotation.from_annotation(
                annotation,
                provenance=self.provenance,
                curie=result["id"]["identifier"],
                biolink_type=types[0],
                label=result["id"].get("label", ""),
            )
            normalized_annotation.props["types"] = types
            normalized_annotation.props["ic"] = result.get("information_content")

            if props.get("description", False):
                normalized_annotation.props["description"] = result["id"].get(
                    "description", None
                )

            output_annotations.append(normalized_annotation)

        return replace(annotated_text, annotations=output_annotations)
