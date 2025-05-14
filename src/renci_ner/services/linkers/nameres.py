#
# An Apache Solr based Named Entity Linker based on the Babel cliques.
# Source code: https://github.com/TranslatorSRI/NameResolution
# Hosted at: https://name-resolution-sri.renci.org/docs
#
import requests

from renci_ner.core import (
    AnnotatedText,
    AnnotationProvenance,
    Annotator,
    NormalizedAnnotation,
)

# Configuration.
RENCI_NAMERES_URL = "https://name-resolution-sri.renci.org"


class NameRes(Annotator):
    """
    An Apache Solr based Named Entity Linker based on the Babel cliques.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="NameRes", url=RENCI_NAMERES_URL, version=self.openapi_version
        )

    def __init__(
        self, url=RENCI_NAMERES_URL, requests_session=requests.Session(), timeout=120
    ):
        """
        Set up a NameRes service.

        :param url: The URL of the NameRes service.
        :param requests_session: A Requests session object to use instead of the default one.
        :param timeout: The timeout to use for requests in seconds. Default: 120 seconds.
        """
        self.url = url
        self.lookup_url = url + "/lookup"
        self.requests_session = requests_session

        response = self.requests_session.get(self.url + "/openapi.json", timeout=120)
        response.raise_for_status()
        openapi_data = response.json()
        self.openapi_version = openapi_data.get("info", {"version": "NA"}).get(
            "version", "NA"
        )

    def supported_properties(self):
        """Configurable properties for NameRes."""
        return {
            "timeout": "(int, default: 120) The timeout in seconds for requests to NameRes.",
            "autocomplete": "(true/false, default: false) Whether to search for incomplete words (e.g. 'bra' for brain).",
            "limit": "(int, default: 10) The number of results to return.",
            "highlighting": "(true/false, default: false) Whether to return lists of the names and synonyms matched by the query.",
            "biolink_types": "(list of biolink types, default: []) The biolink types to filter results to, combined with OR.",
            "only_prefixes": "(list of prefixes, default: []) The prefixes to filter results to, combined with OR.",
            "exclude_prefixes": "(list of prefixes, default: []) The prefixes to exclude from search results, combined with AND.",
            "only_taxa": "(list of taxa, default: []) The taxa to filter results to as NCBITaxon identifiers, combined with OR.",
        }

    def annotate(self, text, props=None) -> AnnotatedText:
        """
        Annotate a piece of text using NameRes.

        :param text: A piece of text with the label of a biomedical entity (e.g. "brain" or "ACT1").
        :param props: A dictionary of properties to configure NameRes.
        :return: An AnnotatedText object containing the annotations.
        """
        if props is None:
            props = {}

        session = self.requests_session
        timeout = props.get("timeout", 120)

        response = session.get(
            self.lookup_url,
            params={
                "string": text,
                "autocomplete": props.get("autocomplete", "false"),
                "limit": props.get("limit", 10),
                "highlighting": props.get("highlighting", "false"),
                "biolink_type": "|".join(props.get("biolink_types", [])),
                "only_prefixes": "|".join(props.get("only_prefixes", [])),
                "exclude_prefixes": "|".join(props.get("exclude_prefixes", [])),
                "only_taxa": "|".join(props.get("only_taxa", [])),
            },
            timeout=timeout,
        )

        response.raise_for_status()
        results = response.json()

        annotations = [
            NormalizedAnnotation(
                text=text,
                id=result.get("curie", ""),
                label=result.get("label", ""),
                biolink_type=result.get("types", ["biolink:NamedThing"])[0],
                type=result.get("types", ["biolink:NamedThing"])[0],
                props={
                    "score": result.get("score", 0),
                    "clique_identifier_count": result.get("clique_identifier_count", 0),
                    "synonyms": result.get("synonyms", []),
                    "highlighting": result.get("highlighting", {}),
                    "types": result.get("types", []),
                    "taxa": result.get("taxa", []),
                },
                provenance=self.provenance,
                # Since we're using the whole text, let's just use that
                # as the start/end.
                start=0,
                end=len(text),
            )
            for result in results
        ]

        return AnnotatedText(text, annotations)
