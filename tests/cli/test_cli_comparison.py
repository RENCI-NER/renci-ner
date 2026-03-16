#!/usr/bin/env python3
#
# Test PMID articles by comparing them to known outputs.
import logging
import os
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from requests import HTTPError

from renci_ner.cli import AnnotationJob, build_annotator, make_session
from renci_ner.services.ner.biomegatron import BioMegatron

# Config
WRITE_EXPECTED_OUTPUT = "WRITE_EXPECTED_OUTPUT" in os.environ

# Get a list of all the files in `../data/pmid`.
test_pmid_dir = Path(__file__).parent.parent / "data" / "pmid"
test_pmid_files = [f for f in test_pmid_dir.glob("pmid-*.txt") if f.is_file()]

# Supported output formats.
# TODO: add to renci_ner.cli so that we can get this programmatically.
OUTPUT_FORMATS = ["csv", "tsv", "jsonl"]

all_tests = list(product(test_pmid_files, OUTPUT_FORMATS))


@pytest.mark.parametrize("pmid_filename, output_format", all_tests)
def test_pmid_comparison(pmid_filename: str, output_format: str):
    """
    If there is a comparison file (`pmid-[X].csv`, `pmid-[X].json`, etc.), convert it using the CLI
    and compare the output to the expected output.

    :param pmid_filename: The PMID file to test. This should be a plain text file.
    """
    logger = logging.getLogger(__name__)

    # SAPBERT is publicly accessible but BioMegatron is not, so we should check to
    # see if we can access it before using it.
    try:
        _ = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    # Use delete=False so we control cleanup; close immediately so write_output
    # can open the path in text mode without conflict.
    tmpfile = NamedTemporaryFile(delete=False, suffix=f".{output_format}")
    tmpfile_path = Path(tmpfile.name)
    tmpfile.close()

    try:
        session = make_session(retries=10)
        annotate_fn = build_annotator("biomegatron-sapbert", session, ner_limit=10)

        job = AnnotationJob(annotate_fn=annotate_fn)
        job.run(
            input_filenames=[pmid_filename.as_posix()],
            output_format=output_format,
            output_filename=str(tmpfile_path),
        )
        # Read with read_text() so newline handling matches the expected file.
        output_content = tmpfile_path.read_text()
    finally:
        tmpfile_path.unlink(missing_ok=True)

    if (output_filename := pmid_filename.with_suffix(f".{output_format}")).exists():
        expected_output_text = output_filename.read_text()
        # TODO: it would be better to do a line-by-line comparison.
        # TODO: for JSONL file, it would be better to load the JSON object and then do the comparison.
        assert expected_output_text.strip() == output_content.strip()
    else:
        if WRITE_EXPECTED_OUTPUT:
            with open(output_filename, "w") as f:
                f.write(output_content)
            logger.info(
                f"Converted {pmid_filename} into output format {output_format} and wrote to {output_filename}."
            )
            pytest.skip(
                f"No expected output file {output_filename}, but created in this run."
            )
        else:
            logger.info(
                f"Converted {pmid_filename} into output format {output_format} but expected output file {output_filename} not found. Use verbose mode to see output."
            )
            print(f"--- start {output_filename} expected output ---")
            print(output_content)
            print(f"--- end {output_filename} expected output ---")
            assert False, (
                f"No expected output file for output format {output_format}: {output_filename}"
            )
