from renci_ner.config import get_setting


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
