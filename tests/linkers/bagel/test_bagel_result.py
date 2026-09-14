from renci_ner.services.linkers.bagel import BagelResult


def test_sort_key_orders_exact_first_and_umls_last():
    def result(identifier, synonym_type):
        return BagelResult(identifier=identifier, synonym_type=synonym_type)

    results = [
        result("UMLS:C1", "exact"),
        result("MONDO:1", "related"),
        result("MONDO:2", "narrow"),
        result("MONDO:3", "exact"),
        result("MONDO:4", "broad"),
        result("MONDO:5", "something new"),
    ]
    ordered = sorted(results, key=BagelResult.get_bagel_sort_key)
    assert [r.identifier for r in ordered] == [
        "MONDO:3",  # exact, non-UMLS
        "UMLS:C1",  # exact, but UMLS is demoted within its group
        "MONDO:2",  # narrow and broad tie
        "MONDO:4",
        "MONDO:1",  # related
        "MONDO:5",  # unknown synonym types sort last instead of raising
    ]


def test_dict_round_trip():
    d = {
        "label": "brain",
        "identifier": "UBERON:0000955",
        "description": "",
        "entity_type": "biolink:AnatomicalEntity",
        "taxa": "",
        "taxa_ids": ["NCBITaxon:9606", "NCBITaxon:10090"],
        "synonym_type": "exact",
    }
    assert BagelResult.from_dict(d).to_dict() == d
