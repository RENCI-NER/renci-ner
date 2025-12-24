import csv
import json
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm
from urllib3 import Retry

from renci_ner.core import AnnotatorWithProps, AnnotatedText
from renci_ner.formats.csv import DelimitedFile
from renci_ner.formats.txt import TextFile
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.normalization.nodenorm import NodeNorm

import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command
@click.argument(
    "input_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    nargs=-1,
    required=True,
)
@click.option(
    "--include-column",
    "-c",
    type=str,
    multiple=True,
    help="Column name(s) to include for processing",
)
@click.option(
    "--exclude-column",
    type=str,
    multiple=True,
    help="Column name(s) to exclude for processing",
)
@click.option(
    "--method",
    type=click.Choice(
        ["biomegatron-sapbert", "biomegatron-nameres", "biomegatron-bagel"]
    ),
    default="biomegatron-sapbert",
    help="NER method",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="-",
    help="Output file",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["csv", "tsv", "jsonl"]),
    default="csv",
    help="Output format",
)
@click.option(
    "--ner-limit",
    type=int,
    default=1,
    help="Limit the number of results per annotation.",
)
@click.option(
    "--retries",
    type=int,
    default=10,
    help="Number of retries for failed requests",
)
@click.option(
    "--gzipped",
    type=bool,
    default=False,
    help="Whether the input files are gzipped",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
)
@click.option(
    "--progress-every",
    type=int,
    default=10,
    help="Print progress every N texts",
)
def renci_ner(
    input_files,
    include_column,
    exclude_column,
    method,
    output,
    ner_limit,
    output_format,
    retries,
    verbose,
    gzipped,
    progress_every,
):
    """
    A CLI for the RENCI NER.

    :param input_files: The input files or directories to read. We guess the file type using the extension.
    :param include_column: The column(s) to include for processing. If none is specified, every column will be used.
    :param exclude_column: The column(s) to exclude from processing.
    :param method: The NER method to use. Limited for now, will be quite expansive later.
    :param output: The output file to write to. Defaults to STDOUT.
    :param output_format: The output format to write to.
    :param ner_limit: The maximum number of results per annotation.
    :param retries: Number of retries for failed requests.
    :param verbose: Whether to enable verbose logging.
    :param gzipped: Whether the input files are gzipped.
    """
    input_filenames = list(map(click.format_filename, input_files))
    output_filename = click.format_filename(output)
    return renci_ner_executor(
        input_filenames,
        include_column,
        exclude_column,
        method,
        output_filename,
        ner_limit,
        output_format,
        retries,
        verbose,
        gzipped,
        progress_every,
    )


def renci_ner_executor(
    input_filenames,
    include_column=None,
    exclude_column=None,
    method="biomegatron-nameres",
    output_filename=None,
    ner_limit=10,
    output_format="csv",
    retries=10,
    verbose=True,
    gzipped=False,
    progress_every=10,
):
    # Set the logging level.
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Set up a Requests session we can use.
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
    )
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))

    all_texts = []
    for input_filename in input_filenames:
        logging.debug(f"Reading input file: {input_filename}")

        # Step 1. Read the input file.
        input_filepath = Path(input_filename)
        suffixes = input_filepath.suffixes

        # Is this file compressed?
        file_gzipped = False
        if suffixes[-1].lower() == ".gz":
            file_gzipped = True
            suffixes.pop()
        if not gzipped:
            file_gzipped = gzipped

        # What kind of file is this?
        texts = []
        last_suffix = suffixes[-1].lower() if len(suffixes) > 0 else ""
        if last_suffix.endswith(".csv"):
            texts = DelimitedFile(
                input_filename,
                columns_include=include_column,
                columns_exclude=exclude_column,
                gzipped=file_gzipped,
                dialect="excel",
            ).read_file()
        elif last_suffix.endswith(".tsv"):
            texts = DelimitedFile(
                input_filename,
                columns_include=include_column,
                columns_exclude=exclude_column,
                gzipped=file_gzipped,
                dialect="excel-tab",
            ).read_file()
        elif last_suffix.endswith(".txt"):
            texts = TextFile(input_filename, gzipped=file_gzipped).read_file()
        else:
            logger.error(
                f"Could not determine a file type for {input_filename} based on the suffixes {input_filepath.suffixes}."
            )

        # logger.info(f"Read {len(texts)} texts from {input_filename}.")
        all_texts.extend(texts)

    # logger.info(f"Read a total of {len(all_texts)} texts across all input files.")

    # Step 2. Annotate the input files.
    logging.info(f"Annotating texts with {method}.")
    match method:
        case "biomegatron-sapbert":
            annotators = [
                AnnotatorWithProps(BioMegatron(requests_session=session), {}),
                AnnotatorWithProps(
                    BabelSAPBERTAnnotator(requests_session=session),
                    {"limit": ner_limit},
                ),
            ]

            def annotator(annotated_text):
                return annotated_text.annotate_with(annotators).transform(NodeNorm())
        case "biomegatron-nameres":
            annotators = [
                AnnotatorWithProps(BioMegatron(requests_session=session), {}),
                AnnotatorWithProps(
                    NameRes(requests_session=session), {"limit": ner_limit}
                ),
            ]

            def annotator(annotated_text):
                return annotated_text.annotate_with(annotators)
        case "biomegatron-bagel":
            bagel = BagelAnnotator()

            def annotator(annotated_text):
                return bagel.annotate_with(
                    BioMegatron(requests_session=session).annotate(annotated_text.text),
                    [
                        AnnotatorWithProps(
                            annotator=BabelSAPBERTAnnotator(requests_session=session),
                            props={"limit": ner_limit},
                        ),
                        AnnotatorWithProps(
                            annotator=NameRes(requests_session=session),
                        ),
                    ],
                )
        case _:
            raise ValueError(f"Unsupported method: {method}")

    annotated_texts = []
    for text in tqdm(all_texts):
        logging.debug(f"Annotating text: {text}")

        annotated_texts.append(annotator(text))

    # Step 3. Write out the output files.
    if output_filename is None or output_filename == "-":
        outputf = sys.stdout
    else:
        outputf = open(output_filename, "w")
    with outputf:
        match output_format:
            case "jsonl":
                for annotated_text in annotated_texts:
                    outputf.write(json.dumps(annotated_text.to_dict()) + "\n")
            case "csv":
                writer = csv.writer(outputf)
                for annotated_text in annotated_texts:
                    writer.writerow(annotated_text.to_csv())
            case "tsv":
                writer = csv.writer(outputf, delimiter="\t")
                for annotated_text in annotated_texts:
                    writer.writerow(annotated_text.to_csv())
            case _:
                raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    renci_ner()
