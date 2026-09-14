import csv
import io
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from requests import HTTPError

from renci_ner import cli
from renci_ner.core import AnnotatedText, Annotation, AnnotationProvenance, Annotator
from renci_ner.formats import ANNOTATION_COLUMNS
from renci_ner.services.ner.biomegatron import BioMegatron

PMID_TXT = Path(__file__).parent.parent / "data" / "pmid" / "pmid-27351941.txt"


class StubPipeline(Annotator):
    """Annotates every 'brain' in the text."""

    provenance = AnnotationProvenance("Stub", "http://stub.example/", "1")

    def _annotate(self, text, props):
        anns = []
        start = text.find("brain")
        while start >= 0:
            anns.append(
                Annotation(
                    "brain",
                    "UBERON:0000955",
                    "brain",
                    "biolink:AnatomicalEntity",
                    start,
                    start + 5,
                    self.provenance,
                )
            )
            start = text.find("brain", start + 1)
        return AnnotatedText(text, anns)


@pytest.fixture
def stub_pipeline(monkeypatch):
    built = []

    def build(method, session=None, limit=1):
        built.append((method, limit))
        return StubPipeline()

    monkeypatch.setattr(cli, "build_pipeline", build)
    return built


def test_csv_in_csv_out(tmp_path, stub_pipeline):
    path = tmp_path / "in.csv"
    path.write_text("id,text\n1,The brain.\n2,No match.\n")
    result = CliRunner().invoke(
        cli.main, ["--exclude-column", "id", "--no-progress", "--limit", "3", str(path)]
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader(io.StringIO(result.output)))
    assert [(r["text"], r["annotation_id"]) for r in rows] == [
        ("The brain.", "UBERON:0000955"),
        ("No match.", ""),
    ]
    assert rows[0]["annotation_prov"] == "Stub 1"
    assert stub_pipeline == [("biomegatron-sapbert", 3)]


def test_txt_in_jsonl_file_out(tmp_path, stub_pipeline):
    path = tmp_path / "in.txt"
    path.write_text("brain brain\n")
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(
        cli.main,
        ["-f", "jsonl", "-o", str(out), "--method", "biomegatron-bagel", str(path)],
    )
    assert result.exit_code == 0, result.output
    (line,) = out.read_text().splitlines()
    data = json.loads(line)
    assert data["location"] == [str(path), "row=1", "text"]
    assert [a["start"] for a in data["annotations"]] == [0, 6]


def test_column_options_need_delimited_input(tmp_path, stub_pipeline):
    path = tmp_path / "in.txt"
    path.write_text("brain\n")
    result = CliRunner().invoke(cli.main, ["-c", "text", str(path)])
    assert result.exit_code != 0
    assert "only apply to CSV/TSV" in result.output


def test_live_pmid():
    """Run the default pipeline on a real abstract; needs BioMegatron."""
    try:
        BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
    result = CliRunner().invoke(
        cli.main, ["--no-progress", "-f", "jsonl", str(PMID_TXT)]
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines()]
    assert len(lines) == 3
    assert sum(len(line["annotations"]) for line in lines) > 10


@pytest.mark.parametrize(
    "method, step_types, last_props",
    [
        ("biomegatron-sapbert", ["BabelSAPBERTAnnotator", "NodeNorm"], {"limit": 4}),
        ("biomegatron-nameres", ["NameRes"], {"limit": 4}),
        ("biomegatron-bagel", ["MultiAnnotator", "BagelAnnotator"], {"limit": 4}),
    ],
)
def test_build_pipeline_wiring(method, step_types, last_props, monkeypatch):
    """Each method builds the right steps, offline, all on the one session."""
    from conftest import FakeSession

    monkeypatch.setenv("BAGEL_USERNAME", "u")
    monkeypatch.setenv("BAGEL_PASSWORD", "p")
    session = FakeSession()
    pipeline = cli.build_pipeline(method, session, limit=4)

    assert type(pipeline.first[0]).__name__ == "BioMegatron"
    assert [type(service).__name__ for service, _ in pipeline.steps] == step_types
    linker, props = pipeline.steps[0]
    assert props == last_props or pipeline.steps[-1][1] == last_props
    for service, _ in [pipeline.first, *pipeline.steps]:
        if hasattr(service, "requests_session"):
            assert service.requests_session is session


def test_bagel_without_credentials_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.delenv("BAGEL_USERNAME", raising=False)
    monkeypatch.delenv("BAGEL_PASSWORD", raising=False)
    monkeypatch.setattr(
        cli, "make_session", lambda retries: __import__("conftest").FakeSession()
    )
    path = tmp_path / "in.txt"
    path.write_text("brain\n")
    result = CliRunner().invoke(cli.main, ["--method", "biomegatron-bagel", str(path)])
    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "BAGEL_USERNAME" in result.output


def test_several_input_files_and_txt_output(tmp_path, stub_pipeline):
    a, b = tmp_path / "a.txt", tmp_path / "b.jsonl"
    a.write_text("brain one\n")
    b.write_text(json.dumps({"text": "brain two"}) + "\n")
    result = CliRunner().invoke(
        cli.main, ["--no-progress", "-f", "txt", str(a), str(b)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["brain one", "brain two"]


def test_empty_input_file(tmp_path, stub_pipeline):
    path = tmp_path / "empty.csv"
    path.write_text("text\n")
    result = CliRunner().invoke(cli.main, ["--no-progress", str(path)])
    assert result.exit_code == 0, result.output
    # Header only; input columns are learned from the texts, so none appear here.
    assert result.output.splitlines() == [",".join(ANNOTATION_COLUMNS)]
