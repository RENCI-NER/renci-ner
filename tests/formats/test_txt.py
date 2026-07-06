# Test PMID articles by comparing them to known outputs.
from pathlib import Path

import pytest

from renci_ner.formats.txt import TextFile

# Get a list of all the files in `../data/pmid`.
test_pmid_dir = Path(__file__).parent.parent / "data" / "pmid"
test_pmid_files = [f for f in test_pmid_dir.glob("pmid-*.txt") if f.is_file()]


@pytest.mark.parametrize("pmid_filename", test_pmid_files)
def test_reading_pmid_files(pmid_filename):
    texts = list(TextFile(str(pmid_filename)).read_file())

    # We should have multiple texts for each PMID file.
    assert len(texts) > 0

    # Check the locations within the AnnotatedTexts.
    for text in texts:
        assert text.location[0] == str(pmid_filename)
        assert text.location[1] == "TextFile"
        assert text.location[2].startswith("row=")
