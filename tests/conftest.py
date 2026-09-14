"""Shared test doubles: a requests session that never touches the network."""

import pytest
from requests import HTTPError


class FakeResponse:
    def __init__(self, status_code, payload, url):
        self.status_code = status_code
        self.payload = payload
        self.url = url
        self.text = str(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"{self.status_code} for {self.url}")


class FakeSession:
    """
    Answers openapi.json with a version, and everything else with `status_code` and
    `payload`. `payload` may be a callable taking (url, json body or params) so a test
    can tailor the answer to the request. Every non-openapi request is recorded in
    `requests` as (method, url, body).
    """

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload
        self.requests = []

    def _respond(self, method, url, body):
        if url.endswith("/openapi.json"):
            return FakeResponse(200, {"info": {"version": "0.0"}}, url)
        self.requests.append((method, url, body))
        payload = self.payload(url, body) if callable(self.payload) else self.payload
        return FakeResponse(self.status_code, payload, url)

    def get(self, url, params=None, **kwargs):
        return self._respond("GET", url, params)

    def post(self, url, json=None, **kwargs):
        return self._respond("POST", url, json)


@pytest.fixture
def fake_session():
    return FakeSession
