import csv
import json

import requests
from urllib3 import Retry

from renci_ner.core import AnnotatorWithProps, AnnotatedText
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.normalization.nodenorm import NodeNorm

import click
import logging

logging.basicConfig(level=logging.INFO)

# HELPER FUNCTIONS


def get_novel_column_name(new_column: str, old_columns: list):
    """
    Given a list of old columns, find a version of $new_column that
    doesn't already exist, and return that.

    :param new_column: The new column to add.
    :param old_columns: The existing columns in this file.
    :return: The new column name, which will not exist in the existing columns.
    """
    old_columns_set = set(old_columns)
    new_column_name = new_column
    index = 0
    while new_column_name in old_columns_set:
        index += 1
        new_column_name = f"{new_column}_{index}"
    return new_column_name


@click.command
@click.argument(
    "input_files",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    nargs=-1,
    required=True,
)
@click.option(
    "--column", "-c", type=str, multiple=True, help="Column name(s) to use for NER"
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
    "--duplicate-data", is_flag=True, default=False, help="Duplicate data in output."
)
@click.option(
    "--allow-duplicate-ids",
    is_flag=True,
    default=False,
    help="Allow duplicate IDs in output.",
)
@click.option(
    "--retries",
    type=int,
    default=10,
    help="Number of retries for failed requests",
)
@click.option(
    "--continue-jsonl",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="A JSONL output file to continue from",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
)
def renci_ner(
    input_files,
    column,
    method,
    output,
    ner_limit,
    output_format,
    duplicate_data,
    allow_duplicate_ids,
    retries,
    verbose,
    continue_jsonl,
):
    """
    A CLI for the RENCI NER.

    :param input_files: The input files or directories to read. We guess the file type using the extension.
    :param column: The column to use for the NER. If none is specified, every column will be used.
    :param method: The NER method to use. Limited for now, will be quite expansive later.
    :param output: The output file to write to. Defaults to STDOUT.
    :param output_format: The output format to write to.
    :param ner_limit: The maximum number of results per annotation.
    :param duplicate_data: Whether to duplicate the data in the output.
    :param allow_duplicate_ids: Whether to allow duplicate IDs in the output.
    :param retries: Number of retries for failed requests.
    :param verbose: Whether to enable verbose logging.
    """
    input_filenames = list(map(click.format_filename, input_files))
    output_filename = click.format_filename(output)
    continue_jsonl_filename = (
        click.format_filename(continue_jsonl) if continue_jsonl else None
    )
    return renci_ner_executor(input_filenames, column, method, output_filename, ner_limit, output_format, duplicate_data, allow_duplicate_ids, retries, verbose, continue_jsonl_filename)

def renci_ner_executor(input_filenames,
                       column="",
                       method="biomegatron-nameres",
                       output_filename="STDOUT",
                       ner_limit=10,
                       output_format="csv",
                       duplicate_data=False,
                       allow_duplicate_ids=False,
                       retries=10,
                       verbose=True,
                       continue_jsonl_filename=None,
                       ):
    columns = column

    # TODO: if output_format is not set, we should guess it from the extension on output_filename.

    # Set the logging level.
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    # Load up the continue data if specified.
    text_already_processed = dict()
    if continue_jsonl_filename:
        with open(continue_jsonl_filename, "r") as continuef:
            for line in continuef:
                data = json.loads(line)
                if "text" in data:
                    text_already_processed[data["text"]] = data
        logging.info(
            f"Loaded {len(text_already_processed)} text already processed entries from continue JSONL file {continue_jsonl_filename}."
        )

    # Set up the pipeline.
    if method == "biomegatron-sapbert":

        def ner_method(text):
            sapbert_annotations = (
                BioMegatron(requests_session=session)
                .annotate(text)
                .reannotate(
                    BabelSAPBERTAnnotator(requests_session=session),
                    {"limit": ner_limit},
                )
            )
            return NodeNorm(requests_session=session).transform(sapbert_annotations)
    elif method == "biomegatron-nameres":

        def ner_method(text):
            return (
                BioMegatron(requests_session=session)
                .annotate(text)
                .reannotate(NameRes(requests_session=session), {"limit": ner_limit})
            )

    elif method == "biomegatron-bagel":

        def ner_method(text):
            annotated_text = BioMegatron(requests_session=session).annotate(text)
            return BagelAnnotator(requests_session=session).annotate_with(
                annotated_text,
                [
                    AnnotatorWithProps(
                        annotator=BabelSAPBERTAnnotator(requests_session=session),
                        props={"limit": ner_limit},
                    ),
                    AnnotatorWithProps(
                        annotator=NameRes(requests_session=session),
                        props={"limit": ner_limit},
                    ),
                ],
            )

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Read the input files.
    for input_filename in input_filenames:
        # TODO: add support for directories.
        with open(input_filename, "r") as inputf:
            if input_filename.lower().endswith(".csv"):
                reader = csv.DictReader(inputf, dialect="excel")
            elif input_filename.lower().endswith(".tsv"):
                reader = csv.DictReader(inputf, dialect="excel_tab")
            else:
                raise ValueError(f"Unsupported file type: {input_filename}")

            # If no `--column` arguments were given on the command line,
            # fall back to use every column in the file.
            if len(columns) == 0:
                columns = reader.fieldnames
                if len(columns) == 0:
                    raise ValueError(f"No columns found in file: {input_filename}")
                column_list = " - " + "\n - ".join(columns)
                logging.warning(
                    f"No columns specified, using all columns:\n{column_list}"
                )

            # Prepare to write the output.
            with open(output_filename, "w") as outputf:
                old_columns = list(reader.fieldnames)

                if output_format in ["csv", "tsv"]:
                    # TODO: need to add support for continue.

                    # Make sure our new columns don't overlap with existing columns.
                    ner_text_column = get_novel_column_name("ner_text", old_columns)
                    ner_label_column = get_novel_column_name("ner_label", old_columns)
                    ner_curie_column = get_novel_column_name("ner_curie", old_columns)
                    ner_biolink_type_column = get_novel_column_name(
                        "ner_biolink_type", old_columns
                    )

                    output_fields = old_columns + [
                        ner_text_column,
                        ner_label_column,
                        ner_curie_column,
                        ner_biolink_type_column,
                    ]
                    writer = (
                        csv.DictWriter(
                            outputf, dialect="excel", fieldnames=output_fields
                        )
                        if output_format == "csv"
                        else csv.DictWriter(
                            outputf, dialect="excel_tab", fieldnames=output_fields
                        )
                    )
                    writer.writeheader()

                    for row in reader:
                        logging.info(f"Processing row: {row}")

                        ner_text = "\n".join(
                            [
                                row[column]
                                for column in columns
                                if row[column].strip() != ""
                            ]
                        )

                        if ner_text.strip() == "":
                            writer.writerow(row)
                            continue

                        annotation_ids = set()

                        annotated_text = ner_method(ner_text)

                        if len(annotated_text.annotations) == 0:
                            writer.writerow(row)
                            continue

                        first_row = True
                        for annotation in annotated_text.annotations:
                            if first_row:
                                output_row = row.copy()
                                first_row = False
                            elif not duplicate_data:
                                output_row = dict(map(lambda x: (x, ""), row.keys()))

                            if (not allow_duplicate_ids) and (
                                annotation.id in annotation_ids
                            ):
                                continue
                            annotation_ids.add(annotation.id)

                            output_row[ner_text_column] = annotation.text
                            output_row[ner_label_column] = annotation.label
                            output_row[ner_curie_column] = annotation.id
                            output_row[ner_biolink_type_column] = annotation.type
                            writer.writerow(output_row)

                            logging.info(
                                f" - Annotation: '{annotation.text}' annotated as {annotation.id} '{annotation.label}' (type {annotation.type})"
                            )

                        logging.info("")
                elif output_format == "jsonl":
                    count_outputs = 0

                    # The easiest output format: we basically serialize the AnnotatedText object.
                    for row in reader:
                        logging.info(f"Processing row: {row}")

                        ner_text = "\n".join(
                            [
                                row[column]
                                for column in columns
                                if row[column].strip() != ""
                            ]
                        )

                        if ner_text in text_already_processed:
                            logging.info(
                                f" - Text already processed, returning previous entry: '{ner_text}'"
                            )
                            outputf.write(
                                json.dumps(text_already_processed[ner_text]) + "\n"
                            )
                            count_outputs += 1
                            continue

                        if ner_text.strip() == "":
                            annotated_text = AnnotatedText("", [])
                        else:
                            annotated_text = ner_method(ner_text)

                        logging.info(f" - Annotated text: {annotated_text}")

                        outputf.write(json.dumps(annotated_text.to_dict()) + "\n")
                        count_outputs += 1

                    logging.info(
                        f"Wrote {count_outputs} JSON lines to {output_filename}."
                    )
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == "__main__":
    renci_ner()
