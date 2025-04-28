#
# The Translator Node Normalizer.
# Source code: https://github.com/TranslatorSRI/NodeNormalization
# Hosted at: https://nodenormalization-sri.renci.org/
#
import logging

import requests

from renci_ner.annotations import (
    AnnotatedText,
    AnnotationProvenance,
    NormalizedAnnotation,
)

# Configuration.
RENCI_NODENORM_URL = "https://nodenormalization-sri.renci.org"


class NodeNorm:
    """
    The Translator Node Normalizer.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="NodeNorm", url=RENCI_NODENORM_URL, version=self.openapi_version
        )

    def __init__(self, url=RENCI_NODENORM_URL, requests_session=requests.Session()):
        """
        Set up a BioMegatron service.

        :param url: The URL of the BioMegatron service.
        :param requests_session: A Requests session object to use instead of the default one.
        """
        self.url = url
        self.get_normalized_nodes_url = url + "/get_normalized_nodes"
        self.requests_session = requests_session

        response = requests.get(self.url + "/openapi.json")
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )

    def supported_properties(self):
        """Some configurable parameters."""
        return {
            "geneprotein_conflation": "(true/false, default: true) Whether to conflate gene and protein identifiers.",
            "drugchemical_conflation": "(true/false, default: false) Whether to conflate drug and chemical identifiers.",
            "description": "(true/false, default: false) Whether to include descriptions in the response.",
        }

    def transform(self, annotated_text: AnnotatedText, props={}) -> AnnotatedText:
        # Set up query.
        session = self.requests_session

        ids = list(map(lambda a: a.id, annotated_text.annotations))

        response = session.post(
            self.get_normalized_nodes_url,
            json={
                "curies": ids,
                "conflate": props.get("geneprotein_conflation", "true"),
                "drug_chemical_conflate": props.get("drugchemical_conflation", "false"),
                "description": props.get("description", "false"),
            },
        )
        if response.status_code != 200:
            # raise Exception(f"NodeNorm returned status code {response.status_code}")
            logging.error(
                f"NodeNorm returned status code {response.status_code} {response.text} for CURIEs {ids}, skipping."
            )
            return annotated_text

        results = response.json()

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

            types = result["type"]
            if not types:
                types = ["biolink:NamedThing"]

            normalized_annotation = NormalizedAnnotation.from_annotation(
                annotation,
                result["id"]["identifier"],
                types[0],
                result["id"].get("label", None),
            )
            normalized_annotation.based_on = normalized_annotation.based_on + [
                annotation
            ]
            normalized_annotation.provenances.append(self.provenance)
            normalized_annotation.props["types"] = types
            normalized_annotation.props["ic"] = results.get("ic", None)

            if props.get("description", False):
                normalized_annotation.props["description"] = result["id"].get(
                    "description", None
                )

            output_annotations.append(normalized_annotation)

        return AnnotatedText(annotated_text.text, output_annotations)
