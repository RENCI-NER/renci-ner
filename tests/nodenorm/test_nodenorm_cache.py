"""Offline test for NodeNorm's per-identifier cache, using a fake session."""

from renci_ner.services.normalization.nodenorm import NodeNorm


class FakeResponse:
    status_code = 200
    url = "http://fake/"

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.posted = []

    def get(self, url, **kwargs):
        return FakeResponse({"info": {"version": "0.0"}})

    def post(self, url, json, **kwargs):
        self.posted.append(json["curies"])
        return FakeResponse(
            {curie: {"id": {"identifier": curie}} for curie in json["curies"]}
        )


def test_nodenorm_caches_per_identifier_and_flags():
    session = FakeSession()
    nodenorm = NodeNorm(requests_session=session)

    assert nodenorm.normalize(["A:1", "B:2"]).keys() == {"A:1", "B:2"}
    # Only the identifier not yet seen is requested; the result still covers both.
    assert nodenorm.normalize(["A:1", "C:3"]).keys() == {"A:1", "C:3"}
    assert session.posted == [["A:1", "B:2"], ["C:3"]]

    # Different conflation flags are a different cache entry.
    nodenorm.normalize(["A:1"], {"drugchemical_conflation": True})
    assert session.posted[-1] == ["A:1"]

    # Everything cached: no request at all, including for the empty list.
    nodenorm.normalize(["A:1", "B:2", "C:3"])
    nodenorm.normalize([])
    assert len(session.posted) == 3

    nodenorm.normalize(["A:1"], {"skip_cache": True})
    assert session.posted[-1] == ["A:1"]
