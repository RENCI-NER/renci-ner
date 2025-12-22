# Test PMID articles by comparing them to known outputs.
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from itertools import product

import pytest

from renci_ner.cli import renci_ner_executor
from renci_ner.formats.txt import TextFile

# Get a list of all the files in `../data/pmid`.
test_pmid_dir = Path(__file__).parent.parent / "data" / "pmid"
test_pmid_files = [f for f in test_pmid_dir.glob("pmid-*.txt") if f.is_file()]

@pytest.mark.parametrize("pmid_filename", test_pmid_files)
def test_reading_pmid_files(pmid_filename):
    texts = list(TextFile(str(pmid_filename)).read_file())
    assert len(texts) > 0
