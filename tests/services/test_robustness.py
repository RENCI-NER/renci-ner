"""Offline tests for 403 handling and the retrying session, using the fake session from conftest."""

import logging

import pytest
from conftest import FakeSession
from requests import HTTPError

from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm
from renci_ner.utils import make_session


@pytest.mark.parametrize("cls", [BioMegatron, NameRes, BabelSAPBERTAnnotator])
def test_annotators_log_and_skip_403(cls, caplog):
    service = cls(requests_session=FakeSession(403))
    with caplog.at_level(logging.WARNING):
        result = service.annotate("brain")
    assert result.annotations == []
    assert "HTTP 403" in caplog.text
    assert '"brain"' in caplog.text


@pytest.mark.parametrize("cls", [BioMegatron, NameRes, BabelSAPBERTAnnotator])
def test_annotators_raise_on_other_errors(cls):
    service = cls(requests_session=FakeSession(500))
    with pytest.raises(HTTPError):
        service.annotate("brain")


def test_nodenorm_403_returns_empty(caplog):
    nodenorm = NodeNorm(requests_session=FakeSession(403))
    with caplog.at_level(logging.WARNING):
        assert nodenorm.normalize(["UBERON:0000955"]) == {}
    assert "UBERON:0000955" in caplog.text


def test_bagel_requires_credentials(monkeypatch):
    monkeypatch.delenv("BAGEL_USERNAME", raising=False)
    monkeypatch.delenv("BAGEL_PASSWORD", raising=False)
    with pytest.raises(ValueError):
        BagelAnnotator(requests_session=FakeSession(200))


def test_bagel_403_returns_no_results():
    bagel = BagelAnnotator(
        requests_session=FakeSession(403), username="u", password="p"
    )
    assert bagel.query_bagel("brain", "the brain", (), "{}") == []


def test_make_session_retries():
    session = make_session(retries=3)
    retry = session.get_adapter("https://example.org/").max_retries
    assert retry.total == 3
    assert "POST" in retry.allowed_methods
    assert 503 in retry.status_forcelist
