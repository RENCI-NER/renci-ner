"""Offline tests for NodeNorm's per-identifier cache, using the fake session from conftest."""

from conftest import FakeSession

from renci_ner.services.normalization.nodenorm import NodeNorm


def echo_identifiers(url, body):
    return {curie: {"id": {"identifier": curie}} for curie in body["curies"]}


def posted(session):
    return [body["curies"] for _, _, body in session.requests]


def test_nodenorm_caches_per_identifier_and_flags():
    session = FakeSession(payload=echo_identifiers)
    nodenorm = NodeNorm(requests_session=session)

    assert nodenorm.normalize(["A:1", "B:2"]).keys() == {"A:1", "B:2"}
    # Only the identifier not yet seen is requested; the result still covers both.
    assert nodenorm.normalize(["A:1", "C:3"]).keys() == {"A:1", "C:3"}
    assert posted(session) == [["A:1", "B:2"], ["C:3"]]

    # Different conflation flags are a different cache entry.
    nodenorm.normalize(["A:1"], {"drugchemical_conflation": True})
    assert posted(session)[-1] == ["A:1"]

    # Everything cached: no request at all, including for the empty list.
    nodenorm.normalize(["A:1", "B:2", "C:3"])
    nodenorm.normalize([])
    assert len(session.requests) == 3

    nodenorm.normalize(["A:1"], {"skip_cache": True})
    assert posted(session)[-1] == ["A:1"]


def test_nodenorm_cache_is_emptied_when_full():
    session = FakeSession(payload=echo_identifiers)
    nodenorm = NodeNorm(requests_session=session)
    nodenorm.cache_size = 3

    nodenorm.normalize(["A:1", "B:2"])
    nodenorm.normalize(["C:3", "D:4"])  # would make 4 entries: cache is cleared first
    assert len(nodenorm._cache) == 2
    nodenorm.normalize(["A:1"])
    assert posted(session) == [["A:1", "B:2"], ["C:3", "D:4"], ["A:1"]]
