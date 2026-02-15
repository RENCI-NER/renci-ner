import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

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

logging.basicConfig(level=logging.INFO)


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


@dataclass
class AnnotationJob:
    """A job that reads inputs, annotates them, and writes the results."""

    annotate_fn: Callable[[AnnotatedText], AnnotatedText]
    session: requests.Session = field(default_factory=requests.Session)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def read_inputs(
        self,
        input_filenames: list[str],
        columns_include=None,
        columns_exclude=None,
        gzipped: bool = False,
    ) -> list[AnnotatedText]:
        """Read all input files, returning a flat list of AnnotatedText objects."""
        all_texts = []
        for input_filename in input_filenames:
            self.logger.debug(f"Reading input file: {input_filename}")
            reader = reader_for_file(
                input_filename,
                columns_include=columns_include,
                columns_exclude=columns_exclude,
                gzipped=gzipped,
            )
            all_texts.extend(reader.read_file())
        return all_texts

    def annotate_texts(self, texts: list[AnnotatedText]) -> list[AnnotatedText]:
        """Run the annotation function on all texts with progress bar."""
        logging.info(f"Annotating {len(texts)} texts.")
        annotated_texts = []
        for text in tqdm(texts):
            self.logger.debug(f"Annotating text: {text}")
            annotated_texts.append(self.annotate_fn(text))
        return annotated_texts

    def write_output(
        self,
        annotated_texts: list[AnnotatedText],
        output_filename: str = None,
        output_format: str = "csv",
    ) -> None:
        """Write annotated texts to the output file in the requested format."""
        if output_filename is None or output_filename == "-":
            outputf = sys.stdout
        else:
            outputf = open(output_filename, "w")
        with outputf:
            writer = writer_for_format(output_format, output_filename)
            writer.write_file(annotated_texts, outputf, duplicate_values=False)

    def run(
        self,
        input_filenames: list[str],
        columns_include=None,
        columns_exclude=None,
        gzipped: bool = False,
        output_filename: str = None,
        output_format: str = "csv",
    ) -> list[AnnotatedText]:
        """Full job: read -> annotate -> write. Returns annotated texts."""
        texts = self.read_inputs(
            input_filenames, columns_include, columns_exclude, gzipped
        )
        annotated_texts = self.annotate_texts(texts)
        self.write_output(annotated_texts, output_filename, output_format)
        return annotated_texts


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
    if verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    session = make_session(retries)
    annotate_fn = build_annotator(method, session, ner_limit)

    job = AnnotationJob(annotate_fn=annotate_fn, session=session)
    job.run(
        input_filenames=list(map(click.format_filename, input_files)),
        columns_include=include_column,
        columns_exclude=exclude_column,
        gzipped=gzipped,
        output_filename=click.format_filename(output),
        output_format=output_format,
    )


if __name__ == "__main__":
    renci_ner()
