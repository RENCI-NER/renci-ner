# Tests for CSV format support
import json
import os
from glob import glob
from pathlib import Path

import pytest

from renci_ner.formats.csv import DelimitedFile

# Get a list of all the CSV and TSV files in `../data/delimited`.
csv_files = glob("../data/delimited/**/*.csv", root_dir=os.path.dirname(__file__))
tsv_files = glob("../data/delimited/**/*.tsv", root_dir=os.path.dirname(__file__))
test_files = csv_files + tsv_files

@pytest.mark.parametrize("test_file", test_files)
def test_reading_delimited_file(test_file: str):
    test_file_path = Path(os.path.join(os.path.dirname(__file__), test_file))
    reader = DelimitedFile(str(test_file_path))

    # We replace the filename with the test directory to simplify comparisons.
    root_location = test_file_path.parent.name
    annotated_texts = list(reader.read_file(root_location=root_location))
    assert len(annotated_texts) > 0

    # TODO: write this out as a JSON file to confirm that we can read it.
    comparison_file_json = test_file_path.with_suffix(".json")
    annotated_texts_json = [text.to_dict() for text in annotated_texts]
    if comparison_file_json.exists():
        annotated_texts_expected_json = [json.loads(s) for s in comparison_file_json.read_text().splitlines()]
        assert annotated_texts_json == annotated_texts_expected_json, f"JSON comparison failed for {test_file_path} vs {comparison_file_json}"
    else:
        annotated_texts_json_str = "\n".join(map(lambda s: json.dumps(s), annotated_texts_json))
        print(f"No JSON comparison file found for {test_file_path}, should contain:\n---\n{annotated_texts_json_str}\n---\n")
        assert False, "No JSON comparison file found for this test, use `-v` to see the JSON output."

    # Check the locations within the AnnotatedTexts.
    for text in annotated_texts:
        assert text.location[0] == root_location
        assert text.location[1] == "DelimitedFile"
        assert text.location[2].startswith("row=")
