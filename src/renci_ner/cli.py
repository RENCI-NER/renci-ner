import logging
import sys
from collections.abc import Callable

import click
import requests
from tqdm import tqdm
from urllib3 import Retry

from renci_ner.core import AnnotatedText, AnnotatorWithProps
from renci_ner.formats import reader_for_file, writer_for_format
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm


def make_session(retries: int = 10) -> requests.Session:
    """Create a requests Session with retry configuration."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods={"GET", "POST"},
    )
    session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry))
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
    return session


def build_annotator(
    method: str, session: requests.Session, ner_limit: int = 10
) -> Callable[[AnnotatedText], AnnotatedText]:
    """Build an annotation callable for the given method.

    :param method: The NER method name.
    :param session: The requests session to use.
    :param ner_limit: Maximum number of results per annotation.
    :return: A callable that annotates an AnnotatedText.
    """
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
            biomegatron = BioMegatron(requests_session=session)
            sapbert = BabelSAPBERTAnnotator(requests_session=session)
            nameres = NameRes(requests_session=session)

            def annotator(annotated_text):
                return bagel.annotate_with(
                    biomegatron.annotate(annotated_text.text),
                    [
                        AnnotatorWithProps(
                            annotator=sapbert,
                            props={"limit": ner_limit},
                        ),
                        AnnotatorWithProps(
                            annotator=nameres,
                        ),
                    ],
                )

        case _:
            raise ValueError(f"Unsupported method: {method}")

    return annotator


def run_annotation_job(
    annotate_fn: Callable[[AnnotatedText], AnnotatedText],
    input_filenames: list[str],
    columns_include=None,
    columns_exclude=None,
    gzipped: bool = False,
    output_filename: str = None,
    output_format: str = "csv",
) -> list[AnnotatedText]:
    """Read inputs, annotate each text (with a progress bar), and write the results."""
    logger = logging.getLogger(__name__)

    texts = []
    for input_filename in input_filenames:
        logger.debug(f"Reading input file: {input_filename}")
        reader = reader_for_file(
            input_filename,
            columns_include=columns_include,
            columns_exclude=columns_exclude,
            gzipped=gzipped,
        )
        texts.extend(reader.read_file())

    logger.info(f"Annotating {len(texts)} texts.")
    annotated_texts = [annotate_fn(text) for text in tqdm(texts)]

    writer = writer_for_format(output_format, output_filename)
    if output_filename is None or output_filename == "-":
        writer.write_file(annotated_texts, sys.stdout)
    else:
        with open(output_filename, "w") as outputf:
            writer.write_file(annotated_texts, outputf)

    return annotated_texts


@click.command()
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
    is_flag=True,
    default=False,
    help="Whether the input files are gzipped",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging"
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
):
    """A CLI for the RENCI NER."""
    logging.basicConfig(level=logging.INFO)
    if verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    session = make_session(retries)
    annotate_fn = build_annotator(method, session, ner_limit)

    run_annotation_job(
        annotate_fn,
        input_filenames=list(map(click.format_filename, input_files)),
        columns_include=include_column,
        columns_exclude=exclude_column,
        gzipped=gzipped,
        output_filename=click.format_filename(output),
        output_format=output_format,
    )


if __name__ == "__main__":
    renci_ner()
