"""
A command line interface: annotate the texts in TXT, JSONL, CSV or TSV files with one of
a few standard pipelines and write the results out.

    renci-ner --method biomegatron-sapbert -o out.csv input.csv
"""

import logging
import sys
from collections.abc import Iterable
from dataclasses import replace

import click
from tqdm import tqdm

from renci_ner.core import AnnotatedText, Annotator, MultiAnnotator, Pipeline
from renci_ner.formats import (
    FORMATS,
    DelimitedFile,
    open_text,
    reader_for_file,
    writer_for_format,
)
from renci_ner.services.linkers.babelsapbert import BabelSAPBERTAnnotator
from renci_ner.services.linkers.bagel import BagelAnnotator
from renci_ner.services.linkers.nameres import NameRes
from renci_ner.services.ner.biomegatron import BioMegatron
from renci_ner.services.normalization.nodenorm import NodeNorm
from renci_ner.utils import make_session

logger = logging.getLogger(__name__)

METHODS = ["biomegatron-sapbert", "biomegatron-nameres", "biomegatron-bagel"]
# How many candidates each linker hands to Bagel.
BAGEL_CANDIDATES = 10


def build_pipeline(method: str, session=None, limit: int = 1) -> Annotator:
    """
    The Pipeline for one of METHODS.

    :param method: One of METHODS.
    :param session: A requests session shared by every service (see utils.make_session).
    :param limit: How many linked results to keep per recognized entity.
    """
    biomegatron = BioMegatron(requests_session=session)
    match method:
        case "biomegatron-sapbert":
            return Pipeline(
                biomegatron,
                (BabelSAPBERTAnnotator(requests_session=session), {"limit": limit}),
                NodeNorm(requests_session=session),
            )
        case "biomegatron-nameres":
            return Pipeline(
                biomegatron, (NameRes(requests_session=session), {"limit": limit})
            )
        case "biomegatron-bagel":
            nodenorm = NodeNorm(requests_session=session)
            return Pipeline(
                biomegatron,
                MultiAnnotator(
                    (
                        BabelSAPBERTAnnotator(requests_session=session),
                        {"limit": BAGEL_CANDIDATES},
                    ),
                    (NameRes(requests_session=session), {"limit": BAGEL_CANDIDATES}),
                ),
                (
                    BagelAnnotator(requests_session=session, nodenorm=nodenorm),
                    {"limit": limit},
                ),
            )
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")


def read_texts(
    input_files: Iterable[str], columns_include=(), columns_exclude=()
) -> list[AnnotatedText]:
    """Read every text from the input files; column options apply to CSV/TSV only."""
    texts = []
    for filename in input_files:
        reader = reader_for_file(filename)
        if isinstance(reader, DelimitedFile):
            reader.columns_include = list(columns_include)
            reader.columns_exclude = set(columns_exclude)
        elif columns_include or columns_exclude:
            raise click.UsageError(
                f"--include-column/--exclude-column only apply to CSV/TSV input, not {filename}"
            )
        texts.extend(reader.read())
    return texts


def annotate_texts(
    pipeline: Annotator, texts: list[AnnotatedText], progress: bool = True
) -> list[AnnotatedText]:
    """Annotate each text, keeping its location; shows a progress bar on stderr."""
    return [
        replace(pipeline.annotate(text.text), location=text.location)
        for text in tqdm(texts, disable=not progress)
    ]


@click.command()
@click.argument(
    "input_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--method",
    type=click.Choice(METHODS),
    default=METHODS[0],
    show_default=True,
    help="Which pipeline to run.",
)
@click.option(
    "-c",
    "--include-column",
    multiple=True,
    help="CSV/TSV column to annotate (repeatable; default: all columns).",
)
@click.option(
    "--exclude-column",
    multiple=True,
    help="CSV/TSV column to skip (repeatable).",
)
@click.option(
    "-o",
    "--output",
    default="-",
    show_default=True,
    help="Output file (gzipped if it ends in .gz), or - for standard output.",
)
@click.option(
    "-f",
    "--output-format",
    type=click.Choice(sorted(s.lstrip(".") for s in FORMATS)),
    default="csv",
    show_default=True,
)
@click.option(
    "--limit",
    default=1,
    show_default=True,
    help="Linked results to keep per recognized entity.",
)
@click.option(
    "--retries", default=10, show_default=True, help="Retries for failed requests."
)
@click.option("--no-progress", is_flag=True, help="Don't show a progress bar.")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def main(
    input_files,
    method,
    include_column,
    exclude_column,
    output,
    output_format,
    limit,
    retries,
    no_progress,
    verbose,
):
    """Annotate the texts in INPUT_FILES (txt, jsonl, csv or tsv; .gz is fine)."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    texts = read_texts(input_files, include_column, exclude_column)
    logger.info(f"Read {len(texts)} texts from {len(input_files)} file(s).")

    pipeline = build_pipeline(method, make_session(retries), limit)
    annotated = annotate_texts(pipeline, texts, progress=not no_progress)

    writer = writer_for_format(output_format, output)
    if output == "-":
        writer.write(annotated, sys.stdout)
    else:
        with open_text(output, "wt", newline="") as f:
            writer.write(annotated, f)
        logger.info(f"Wrote {len(annotated)} annotated texts to {output}.")


if __name__ == "__main__":
    main()
