import pytest
from requests import HTTPError

from renci_ner.core import AnnotatedText, NormalizedAnnotation, AnnotationProvenance, Annotation
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm


def test_check():
    """
    Test if we can use NodeNorm as a transformer on top of NameRes and BabelSAPBERT.
    """
    try:
        biomegatron = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    nameres = NameRes()
    sapbert = BabelSAPBERTAnnotator()
    nodenorm = NodeNorm()

    text = "What does actin do?"
    result_nameres = (
        biomegatron.annotate(text)
        .reannotate(nameres, {"limit": 1})
        .transform(nodenorm, {"geneprotein_conflation": True})
    )
    result_sapbert = (
        biomegatron.annotate(text)
        .reannotate(sapbert, {"limit": 1})
        .transform(nodenorm, {"geneprotein_conflation": True})
    )

    # Check NameRes results.
    assert result_nameres == AnnotatedText(
        text,
        [
            NormalizedAnnotation(
                         text='actin',
             id='NCBIGene:71',
             label='ACTG1',
             type='biolink:Gene',
            biolink_type='biolink:Gene',
                start=10,
                end=15,
                provenance=nodenorm.provenance,
                based_on=[
                    Annotation(
                                                  text='actin',
                              id='I2-',
                              label='',
                              type='biolink:Protein',
                              start=10,
                              end=15,
                              provenance=biomegatron.provenance,
                              based_on=[],
                              props={},
                          ),
                                  NormalizedAnnotation(
                                          text='actin',
                              id='UniProtKB:P63261',
                              label='ACTG_HUMAN Actin, cytoplasmic 2 (sprot)',
                              type='biolink:Protein',
                                      biolink_type='biolink:Protein',
                              start=10,
                              end=15,
                              provenance=AnnotationProvenance(
                                                 name='NameRes',
                                             url='https://name-resolution-sri.renci.org',
                                             version='1.4.7',
                                         ),
                              based_on=[
                                               Annotation(
                                                       text='actin',
                                               id='I2-',
                                               label='',
                                               type='biolink:Protein',
                                               start=10,
                                               end=15,
                                               provenance=AnnotationProvenance(
                                                                  name='BioMegatron',
                                                              url='https://med-nemo.apps.renci.org',
                                                              version='0.1.0',
                                                          ),
                                               based_on=[],
                                               props={},
                                           ),
                                       ],
                              props={
                                            'clique_identifier_count': 63,
                                            'highlighting': {},
                                            'ic': None,
                                            'score': 16.23857,
                                            'synonyms': [
                                                    'ACTG',
                                                    'ACTG1',
                                                    'hACTG1',
                                                    'Gamma-Actin',
                                                    'gamma-actin (human)',
                                                    'Actin, Cytoplasmic 2',
                                                    'ACTG1 protein, human',
                                                    'Cytoskeletal Gamma-Actin',
                                                    'actin, cytoplasmic 2 (human)',
                                                    'Epididymis Luminal Protein 176',
                                                    'ACTG_HUMAN Actin, cytoplasmic 2 '
                                                    '(sprot)',
                                                ],
                                            'taxa': [
                                                    'NCBITaxon:9606',
                                                ],
                                            'types': [
                                                    'biolink:Gene',
                                                    'biolink:GeneOrGeneProduct',
                                                    'biolink:GenomicEntity',
                                                    'biolink:ChemicalEntityOrGeneOrGeneProduct',
                                                    'biolink:PhysicalEssence',
                                                    'biolink:OntologyClass',
                                                    'biolink:BiologicalEntity',
                                                    'biolink:ThingWithTaxon',
                                                    'biolink:NamedThing',
                                                    'biolink:PhysicalEssenceOrOccurrent',
                                                    'biolink:MacromolecularMachineMixin',
                                                    'biolink:Protein',
                                                    'biolink:GeneProductMixin',
                                                    'biolink:Polypeptide',
                                                    'biolink:ChemicalEntityOrProteinOrPolypeptide',
                                                ],
                                        })
                                      ],
                         props={
                           'clique_identifier_count': 63,
                           'highlighting': {},
                           'ic': None,
                           'score': 16.23857,
                           'synonyms': [
                                   'ACTG',
                                   'ACTG1',
                                   'hACTG1',
                                   'Gamma-Actin',
                                   'gamma-actin (human)',
                                   'Actin, Cytoplasmic 2',
                                   'ACTG1 protein, human',
                                   'Cytoskeletal Gamma-Actin',
                                   'actin, cytoplasmic 2 (human)',
                                   'Epididymis Luminal Protein 176',
                                   'ACTG_HUMAN Actin, cytoplasmic 2 (sprot)',
                               ],
                           'taxa': [
                                   'NCBITaxon:9606',
                               ],
                           'types': [
                                   'biolink:Gene',
                                   'biolink:GeneOrGeneProduct',
                                   'biolink:GenomicEntity',
                                   'biolink:ChemicalEntityOrGeneOrGeneProduct',
                                   'biolink:PhysicalEssence',
                                   'biolink:OntologyClass',
                                   'biolink:BiologicalEntity',
                                   'biolink:ThingWithTaxon',
                                   'biolink:NamedThing',
                                   'biolink:PhysicalEssenceOrOccurrent',
                                   'biolink:MacromolecularMachineMixin',
                                   'biolink:Protein',
                                   'biolink:GeneProductMixin',
                                   'biolink:Polypeptide',
                                   'biolink:ChemicalEntityOrProteinOrPolypeptide',
                               ],
                       },
            )

    ]
    )

    # Check SAPBERT results.
    assert result_sapbert == AnnotatedText(
        text,
        [
            NormalizedAnnotation(
                text='actin',
                id='PANTHER.FAMILY:PTHR11937',
                label='ACTIN',
                type='biolink:GeneFamily',
                biolink_type='biolink:GeneFamily',
                start=10,
                end=15,
                provenance=sapbert.provenance,
                based_on=[
                    Annotation(
                        text='actin',
                        id='I2-',
                        label='',
                        start=10,
                        end=15,
                        provenance=biomegatron.provenance,
                        type='biolink:Protein',
                    )
                ],
                props={
                    'score': 0.9999999,
                }
            )
        ]
    )
