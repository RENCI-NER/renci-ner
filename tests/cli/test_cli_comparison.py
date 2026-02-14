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

from renci_ner.cli import renci_ner_executor
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

    input_path = Path(test_pmid_dir) / pmid_filename

    # input_text = (Path(test_pmid_dir) / pmid_filename).read_text().strip()
    #
    # if not input_text:
    #     raise ValueError(f"Empty file: {pmid_filename}, cannot test.")

    tmpfile = NamedTemporaryFile()

    # SAPBERT is publicly accessible but BioMegatron is not, so we should check to
    # see if we can access it before using it.
    try:
        _ = BioMegatron()
    except HTTPError as err:
        pytest.skip(f"BioMegatron is not available: {err}")
        return

    # TODO: This is unnecessarily slowed by the fact that we have to reannotate the text multiple times.
    # We should rewrite this so that instead of renci_ner_executor() we call the internal method that returns
    # a list of AnnotatedTexts, and then we can export them out using the individual tools.
    renci_ner_executor(
        input_filenames=[input_path.as_posix()],
        output_format=output_format,
        output_filename=tmpfile.name,
        # TODO: make this configurable
        method="biomegatron-sapbert",
        ner_limit=10,
        retries=10,
        verbose=True,
    )
    output_content = ""
    for lines in tmpfile:
        output_content += lines.decode("utf-8")
    tmpfile.close()

    if (output_filename := pmid_filename.with_suffix(f".{output_format}")).exists():
        expected_output_text = output_filename.read_text().strip()
        # TODO: it would be better to do a line-by-line comparison.
        # TODO: for JSONL file, it would be better to load the JSON object and then do the comparison.
        assert expected_output_text == output_content
    else:
        if WRITE_EXPECTED_OUTPUT:
            with open(output_filename, "w") as f:
                f.write(output_content)
            logger.info(
                f"Converted {pmid_filename} into output format {output_format} and wrote to {output_filename}."
            )
            pytest.skip(f"No expected output file {output_filename}, but created in this run.")
        else:
            logger.info(
                f"Converted {pmid_filename} into output format {output_format} but expected output file {output_filename} not found. Use verbose mode to see output."
            )
            print(f"--- start {output_filename} expected output ---")
            print(output_content)
            print(f"--- end {output_filename} expected output ---")
            assert False, f"No expected output file for output format {output_format}: {output_filename}"
