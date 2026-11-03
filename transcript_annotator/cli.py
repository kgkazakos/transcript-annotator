"""Click CLI. The `paradigm` command is mandatory on every CPR Phase II tool."""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

import click

from . import __version__
from .engine import analyze, build_report
from .io import (
    apply_reviews,
    load_input,
    load_json,
    load_list,
    load_reviews,
    render_agreement,
    render_check,
    render_text,
    report_from_export,
    to_export,
)
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, TOOL_NAME

_FILE = click.Path(exists=True, dir_okay=False)


def _study(path: str, annotations: Optional[str]) -> dict[str, Any]:
    doc = load_input(path)
    if annotations:
        doc = dict(doc)
        doc["annotations"] = list(doc.get("annotations") or []) + load_list(annotations, "annotations")
    return doc


def _emit(payload: dict[str, Any], output: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        click.echo(f"Wrote {output}", err=True)
    else:
        sys.stdout.write(text + "\n")


def _run_options(f):
    f = click.option("--batch-size", type=click.IntRange(min=1), default=None,
                     help="Segments per model call (default 20).")(f)
    f = click.option("--reviews", type=_FILE, default=None, help="Researcher review decisions to check.")(f)
    f = click.option("--annotations", type=_FILE, default=None,
                     help="Researcher annotations to verify alongside the model's.")(f)
    f = click.option("--model", default=None, help="litellm model id (or set CPR_LLM_MODEL).")(f)
    f = click.option("--no-llm", is_flag=True, help="Deterministic core only; never calls a model.")(f)
    return f


def _report(path, no_llm, model, annotations, reviews, batch_size):
    return build_report(
        _study(path, annotations),
        no_llm=no_llm,
        model=model,
        batch_size=batch_size,
        reviews=load_reviews(reviews) if reviews else None,
    )


@click.group()
@click.version_option(__version__, prog_name=TOOL_NAME)
def main() -> None:
    """Slow transcript annotation: codes with their reasoning, verified against the codebook."""


@main.command()
@click.argument("path", type=_FILE)
@_run_options
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def run(path, no_llm, model, annotations, reviews, batch_size, fmt) -> None:
    """Annotate a study and verify every annotation."""
    report = _report(path, no_llm, model, annotations, reviews, batch_size)
    if fmt == "json":
        _emit(to_export(report), None)
    else:
        click.echo(render_text(report))


@main.command()
@click.argument("path", type=_FILE)
@_run_options
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None)
def export(path, no_llm, model, annotations, reviews, batch_size, output) -> None:
    """Write the versioned JSON export (the run file that `review` and `agreement` read)."""
    _emit(to_export(_report(path, no_llm, model, annotations, reviews, batch_size)), output)


@main.command()
@click.argument("path", type=_FILE)
def check(path: str) -> None:
    """Transcript readiness and codebook checks only. Never calls a model."""
    click.echo(render_check(analyze(load_input(path))))


@main.command()
@click.argument("run_file", type=_FILE)
@click.argument("reviews", type=_FILE)
@click.option("-o", "--output", type=click.Path(dir_okay=False), default=None,
              help="Write the run export with the review check added.")
def review(run_file: str, reviews: str, output: Optional[str]) -> None:
    """Check a researcher's review decisions against a saved run. Never calls a model."""
    report = apply_reviews(report_from_export(load_json(run_file)), load_reviews(reviews))
    if output:
        _emit(to_export(report), output)
    if not report.review:
        click.echo("Review record: no findings.")
    for f in report.review:
        click.echo(f"[{f.severity.upper():6}] {f.rule_id}{' @ ' + f.location if f.location else ''}: {f.message}")


@main.command()
@click.argument("run_file", type=_FILE)
def agreement(run_file: str) -> None:
    """Print the researcher vs model agreement table from a saved run. Descriptive only."""
    click.echo(render_agreement(report_from_export(load_json(run_file))))


@main.command()
def paradigm() -> None:
    """Print the AI paradigm and why it fits this research task."""
    click.echo(f"AI paradigm: {AI_PARADIGM}\n")
    click.echo(AI_PARADIGM_RATIONALE)


if __name__ == "__main__":
    main()
