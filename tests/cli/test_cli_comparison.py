#!/usr/bin/env python3
#
# Test PMID articles by comparing them to known outputs.
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from itertools import product

import pytest

from renci_ner.cli import renci_ner_executor

# Get a list of all the files in `../data/pmid`.
test_pmid_dir = Path(__file__).parent.parent / "data" / "pmid"
test_pmid_files = [f for f in test_pmid_dir.glob("pmid-*.txt") if f.is_file()]

# Supported output formats.
# TODO: add to renci_ner.cli so that we can get this programmatically.
OUTPUT_FORMATS = ['csv', 'tsv', 'jsonl']

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
    renci_ner_executor(
        input_filenames=[input_path.as_posix()],
        output_format=output_format,
        output_filename=tmpfile.name,
        # TODO: make this configurable
        method='biomegatron-sapbert',
        ner_limit=10,
        duplicate_data=True,
        allow_duplicate_ids=False,
        retries=10,
        verbose=True,
    )
    output_content = ""
    for lines in tmpfile:
        output_content += lines.decode("utf-8")
    tmpfile.close()

    if (output_filename := pmid_filename.with_suffix(f".{output_format}")).exists():
        expected_output_text = output_filename.read_text().strip()
        assert expected_output_text == output_content
    else:
        logger.info(f"Converted {pmid_filename} into output format {output_format} produced the following output:")
        print(output_content)
        logger.info("---")
        pytest.skip(f"No expected output file for output format {output_format}: {output_filename}")
