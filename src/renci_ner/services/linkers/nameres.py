#
# A Named Entity Linker based on the Babel cliques.
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
    A Named Entity Linker based on the Babel cliques.
    """

    @property
    def provenance(self) -> AnnotationProvenance:
        """Return an AnnotationProvenance describing annotations produced by this service."""
        return AnnotationProvenance(
            name="NameRes", url=RENCI_NAMERES_URL, version=self.openapi_version
        )

    def __init__(self, url=RENCI_NAMERES_URL, requests_session=requests.Session()):
        """
        Set up a BioMegatron service.

        :param url: The URL of the BioMegatron service.
        :param requests_session: A Requests session object to use instead of the default one.
        """
        self.url = url
        self.lookup_url = url + "/lookup"
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
            "autocomplete": "(true/false, default: false) Whether to search for incomplete words (e.g. 'bra' for brain).",
            "limit": "(int, default: 10) The number of results to return.",
            # TODO: add remaining.
        }

    def annotate(self, text, props={}) -> AnnotatedText:
        # Set up query.
        session = self.requests_session

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
        )

        response.raise_for_status()
        results = response.json()

        annotations = []
        for result in results:
            annotations.append(
                # Since NameRes is normalized to Babel, we can treat it as a NormalizedAnnotation.
                NormalizedAnnotation(
                    text=text,
                    curie=result.get("curie", ""),
                    id=result.get("curie", ""),
                    label=result.get("label", ""),
                    biolink_type=result.get("types", ["biolink:NamedThing"])[0],
                    type=result.get("types", ["biolink:NamedThing"])[0],
                    props={
                        "score": result.get("score", 0),
                        "clique_identifier_count": result.get(
                            "clique_identifier_count", 0
                        ),
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
            )

        return AnnotatedText(text, annotations)
