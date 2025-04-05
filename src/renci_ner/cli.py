import csv

import requests

from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.linkers.sapbert import SAPBERTAnnotator
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
    "--column", "-c", type=str, multiple=True, help="Column name(s) to use for NER"
)
@click.option(
    "--method",
    type=click.Choice(["biomegatron-sapbert", "biomegatron-nameres"]),
    default="biomegatron-sapbert",
    help="NER method",
)
@click.option(
    "--output",
    "-O",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    default="-",
    help="Output file",
)
@click.option(
    "--output-format",
    type=click.Choice(["csv", "tsv"]),
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
def renci_ner(
    input_files,
    column,
    method,
    output,
    ner_limit,
    output_format,
    duplicate_data,
    allow_duplicate_ids,
):
    """
    A CLI for the RENCI NER.

    :param input_file: The input file or directory to read. We guess the file type using the extension.
    :param column: The column to use for the NER. If none is specified, every column will be used.
    :param method: The NER method to use. Limited for now, will be quite expansive later.
    :param output_file: The output file to write to. Defaults to STDOUT.
    :param output_format: The output format to write to.
    :param ner_limit: The maximum number of results per annotation.
    :param duplicate_data: Whether to duplicate the data in the output.
    :param allow_duplicate_ids: Whether to allow duplicate IDs in the output.
    """
    input_filenames = list(map(click.format_filename, input_files))
    output_filename = click.format_filename(output)
    columns = column

    # Set up the pipeline.
    if method == "biomegatron-sapbert":

        def ner_method(text):
            sapbert_annotations = (
                BioMegatron()
                .annotate(text)
                .annotate_annotations_with(SAPBERTAnnotator(), {"limit": ner_limit})
            )
            return NodeNorm().transform(sapbert_annotations)
    elif method == "biomegatron-nameres":

        def ner_method(text):
            return (
                BioMegatron()
                .annotate(text)
                .annotate_annotations_with(NameRes(), {"limit": ner_limit})
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

            if len(columns) == 0:
                columns = reader.fieldnames
                if len(columns) == 0:
                    raise ValueError(f"No columns found in file: {input_filename}")
                column_list = " - " + "\n - ".join(columns)
                logging.warning(
                    f"No columns specified, using all columns:\n{column_list}"
                )

            # Prepare the write the output.
            with open(output_filename, "w") as outputf:
                output_fields = list(reader.fieldnames) + [
                    "ner_text",
                    "ner_label",
                    "ner_curie",
                    "ner_biolink_type",
                ]
                writer = (
                    csv.DictWriter(outputf, dialect="excel", fieldnames=output_fields)
                    if output_format == "csv"
                    else csv.writer(
                        outputf, dialect="excel_tab", fieldnames=output_fields
                    )
                )
                writer.writeheader()

                for row in reader:
                    logging.info(f"Processing row: {row}")
                    ner_text = "\n".join(
                        [row[column] for column in columns if row[column].strip() != ""]
                    )

                    annotation_ids = set()

                    annotated_text = ner_method(ner_text)
                    first_row = True
                    for annotation in annotated_text.annotations:
                        if first_row:
                            output_row = row.copy()
                            first_row = False
                        elif not duplicate_data:
                            output_row = dict(map(lambda x: (x, ""), row.keys()))

                        if (not allow_duplicate_ids) and (
                            annotation.curie in annotation_ids
                        ):
                            continue
                        annotation_ids.add(annotation.curie)

                        output_row["ner_text"] = annotation.text
                        output_row["ner_label"] = annotation.label
                        output_row["ner_curie"] = annotation.curie
                        output_row["ner_biolink_type"] = annotation.biolink_type
                        writer.writerow(output_row)

                        logging.info(
                            f" - Annotation: '{annotation.text}' annotated as {annotation.curie} '{annotation.label}' (type {annotation.biolink_type})"
                        )

                    logging.info("")


if __name__ == "__main__":
    renci_ner()
