import pytest
from requests import HTTPError

from renci_ner.core import (
    AnnotatedText,
    Annotation,
    NormalizedAnnotation,
)
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm


def test_check():
    nodenorm = NodeNorm()
    results = nodenorm.normalize(
        ["UMLS:C1412149"],
        {
            "geneprotein_conflation": True,
            "description": True,
        },
    )
    assert len(results) == 1
    assert "UMLS:C1412149" in results

    umls_C1412149 = results["UMLS:C1412149"]
    assert "id" in umls_C1412149
    assert umls_C1412149["id"]["identifier"] == "NCBIGene:71"
    assert umls_C1412149["id"]["label"] == "ACTG1"
    assert "information_content" in umls_C1412149
    assert umls_C1412149["taxa"] == ["NCBITaxon:9606"]


def test_with_transform():
    """
    Test if we can use NodeNorm as a transformer. We can no longer actually test this because both GeneProtein
    and DrugChemical conflation are turned on in NameRes and (eventually) in SAPBERT. When we get a non-RENCI
    linker in here we can use it there; until then, I'll just make up an example to test this.
    """

    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    nameres = NameRes()

    annotated_text = AnnotatedText(
        "What does actin do?",
        [
            NormalizedAnnotation(
                text="actin",
                id="UniProtKB:P63261",
                label="ACTG_HUMAN Actin, cytoplasmic 2 (sprot)",
                type="biolink:Protein",
                biolink_type="biolink:Protein",
                start=10,
                end=15,
                provenance=nameres.provenance,
                based_on=[
                    Annotation(
                        text="actin",
                        id="I2-",
                        label="",
                        type="biolink:Protein",
                        start=10,
                        end=15,
                        provenance=biomegatron.provenance,
                        based_on=[],
                        props={},
                    ),
                ],
                props={
                    "highlighting": {},
                    "taxa": [
                        "NCBITaxon:9606",
                    ],
                    "types": [
                        "biolink:Gene",
                        "biolink:GeneOrGeneProduct",
                        "biolink:GenomicEntity",
                        "biolink:ChemicalEntityOrGeneOrGeneProduct",
                        "biolink:PhysicalEssence",
                        "biolink:OntologyClass",
                        "biolink:BiologicalEntity",
                        "biolink:ThingWithTaxon",
                        "biolink:NamedThing",
                        "biolink:PhysicalEssenceOrOccurrent",
                        "biolink:MacromolecularMachineMixin",
                        "biolink:Protein",
                        "biolink:GeneProductMixin",
                        "biolink:Polypeptide",
                        "biolink:ChemicalEntityOrProteinOrPolypeptide",
                    ],
                },
            )
        ],
    )

    nodenorm = NodeNorm()
    result_nodenorm = annotated_text.transform(
        nodenorm, {"geneprotein_conflation": True}
    )

    assert result_nodenorm == AnnotatedText(
        "What does actin do?",
        [
            NormalizedAnnotation(
                text="actin",
                id="NCBIGene:71",
                label="ACTG1",
                type="biolink:Gene",
                biolink_type="biolink:Gene",
                start=10,
                end=15,
                provenance=nodenorm.provenance,
                based_on=[
                    Annotation(
                        text="actin",
                        id="I2-",
                        label="",
                        type="biolink:Protein",
                        start=10,
                        end=15,
                        provenance=biomegatron.provenance,
                        based_on=[],
                        props={},
                    ),
                    NormalizedAnnotation(
                        text="actin",
                        id="UniProtKB:P63261",
                        label="ACTG_HUMAN Actin, cytoplasmic 2 (sprot)",
                        type="biolink:Protein",
                        biolink_type="biolink:Protein",
                        start=10,
                        end=15,
                        provenance=nameres.provenance,
                        based_on=[
                            Annotation(
                                text="actin",
                                id="I2-",
                                label="",
                                type="biolink:Protein",
                                start=10,
                                end=15,
                                provenance=biomegatron.provenance,
                                based_on=[],
                                props={},
                            ),
                        ],
                        props={
                            "highlighting": {},
                            "ic": None,
                            "taxa": [
                                "NCBITaxon:9606",
                            ],
                            "types": [
                                "biolink:Gene",
                                "biolink:GeneOrGeneProduct",
                                "biolink:GenomicEntity",
                                "biolink:ChemicalEntityOrGeneOrGeneProduct",
                                "biolink:PhysicalEssence",
                                "biolink:OntologyClass",
                                "biolink:BiologicalEntity",
                                "biolink:ThingWithTaxon",
                                "biolink:NamedThing",
                                "biolink:PhysicalEssenceOrOccurrent",
                                "biolink:MacromolecularMachineMixin",
                                "biolink:Protein",
                                "biolink:GeneProductMixin",
                                "biolink:Polypeptide",
                                "biolink:ChemicalEntityOrProteinOrPolypeptide",
                            ],
                        },
                    ),
                ],
                props={
                    "highlighting": {},
                    "ic": None,
                    "taxa": [
                        "NCBITaxon:9606",
                    ],
                    "types": [
                        "biolink:Gene",
                        "biolink:GeneOrGeneProduct",
                        "biolink:GenomicEntity",
                        "biolink:ChemicalEntityOrGeneOrGeneProduct",
                        "biolink:PhysicalEssence",
                        "biolink:OntologyClass",
                        "biolink:BiologicalEntity",
                        "biolink:ThingWithTaxon",
                        "biolink:NamedThing",
                        "biolink:PhysicalEssenceOrOccurrent",
                        "biolink:MacromolecularMachineMixin",
                        "biolink:Protein",
                        "biolink:GeneProductMixin",
                        "biolink:Polypeptide",
                        "biolink:ChemicalEntityOrProteinOrPolypeptide",
                    ],
                },
            )
        ],
    )
