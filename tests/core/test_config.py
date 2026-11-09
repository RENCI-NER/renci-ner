from renci_ner.config import get_setting
from renci_ner.services.linkers.nameres import NameRes


def test_get_setting(tmp_path, monkeypatch):
    """Settings come from the environment first, then the nearest .env, then the default."""
    (tmp_path / ".env").write_text(
        "RENCI_NER_TEST_A=from-dotenv\nRENCI_NER_TEST_B=from-dotenv\nRENCI_NER_TEST_EMPTY=\n"
    )
    subdir = tmp_path / "sub"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    monkeypatch.setenv("RENCI_NER_TEST_A", "from-environment")
    monkeypatch.delenv("RENCI_NER_TEST_B", raising=False)

    assert get_setting("RENCI_NER_TEST_A") == "from-environment"
    assert get_setting("RENCI_NER_TEST_B") == "from-dotenv"
    assert get_setting("RENCI_NER_TEST_EMPTY", "default") == "default"
    assert get_setting("RENCI_NER_TEST_MISSING") is None
    assert get_setting("RENCI_NER_TEST_MISSING", "default") == "default"


def test_service_url_setting(tmp_path, monkeypatch):
    """A service's URL comes from its *_URL setting, and its provenance records it."""

    class FakeSession:
        def __init__(self):
            self.urls = []

        def get(self, url, **kwargs):
            self.urls.append(url)
            return FakeResponse()

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"info": {"version": "test"}}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NAMERES_URL", "http://nameres.example")

    session = FakeSession()
    nameres = NameRes(requests_session=session)
    assert session.urls == ["http://nameres.example/openapi.json"]
    assert nameres.provenance.url == "http://nameres.example"

    # An explicit URL still wins.
    assert NameRes("http://other.example", session).url == "http://other.example"
