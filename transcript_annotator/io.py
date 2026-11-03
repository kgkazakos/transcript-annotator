"""Loaders, renderers and the versioned export."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from .engine import check_reviews, parse_reviews
from .models import (
    AI_PARADIGM,
    EXPORT_SCHEMA,
    Annotation,
    ExclusionNote,
    Finding,
    Report,
    Review,
    VerifiedAnnotation,
)

PathLike = Union[str, Path]


def load_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_input(path: PathLike) -> dict[str, Any]:
    """A study: {question, codebook, transcript, annotations?}."""
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object with 'codebook' and 'transcript'.")
    if is_export(data):
        raise ValueError("This is a run export, not a study. Use `tan review` on exports.")
    return data


def load_list(path: PathLike, key: str) -> list[dict[str, Any]]:
    """A list file, either a bare JSON list or an object holding it under `key`."""
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get(key, [])
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or an object with a '{key}' list.")
    return data


def load_reviews(path: PathLike) -> list[Review]:
    return parse_reviews(load_list(path, "reviews"))


def is_export(data: Any) -> bool:
    return isinstance(data, dict) and data.get("schema") == EXPORT_SCHEMA and "report" in data


# --------------------------------------------------------------------------- export


def to_export(report: Report) -> dict[str, Any]:
    return {
        "schema": EXPORT_SCHEMA,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "paradigm": AI_PARADIGM,
        "model": (report.enrichment or {}).get("model"),
        # No _cpr_synthetic block: annotations are analytic labels about the researcher's data,
        # not synthetic data about people.
        "report": report.to_dict(),
    }


def _finding(d: dict[str, Any]) -> Finding:
    return Finding(rule_id=d["rule_id"], severity=d["severity"], message=d["message"], location=d.get("location"))


def _annotation(d: dict[str, Any]) -> Annotation:
    return Annotation(
        id=d["id"],
        segment_id=d["segment_id"],
        code_id=d["code_id"],
        span=d["span"],
        source=d.get("source", "researcher"),
        criteria_met=tuple(d.get("criteria_met") or ()),
        exclusions_considered=tuple(ExclusionNote(**x) for x in d.get("exclusions_considered") or ()),
        reasoning=d.get("reasoning", ""),
        counter_reading=d.get("counter_reading", ""),
        alternatives=tuple(d.get("alternatives") or ()),
    )


def report_from_export(data: dict[str, Any]) -> Report:
    if not is_export(data):
        raise ValueError(f"Not a {EXPORT_SCHEMA} export.")
    r = data["report"]
    return Report(
        tool=r["tool"],
        version=r["version"],
        question=r.get("question", ""),
        code_names=r.get("code_names", {}),
        readiness=[_finding(f) for f in r.get("readiness", [])],
        codebook=[_finding(f) for f in r.get("codebook", [])],
        annotations=[
            VerifiedAnnotation(annotation=_annotation(v["annotation"]), status=v["status"],
                               findings=[_finding(f) for f in v.get("findings", [])])
            for v in r.get("annotations", [])
        ],
        coverage=r.get("coverage", {}),
        agreement=r.get("agreement"),
        review=[_finding(f) for f in r.get("review", [])],
        prompts=r.get("prompts", []),
        synthetic_share=r.get("synthetic_share", 0.0),
        enrichment=r.get("enrichment"),
        model_note=r.get("model_note"),
    )


def apply_reviews(report: Report, reviews: list[Review]) -> Report:
    """Run the review check on a saved run. The decisions stay the researcher's."""
    report.review = check_reviews(report.annotations, reviews)
    return report


# --------------------------------------------------------------------------- text


def _wrap(label: str, text: str, indent: str = "      ") -> list[str]:
    if not text:
        return []
    return textwrap.wrap(f"{label}{text}", width=96, initial_indent=indent, subsequent_indent=indent + "  ")


def _findings(fs: list[Finding], indent: str = "  ") -> list[str]:
    return [f"{indent}[{f.severity.upper():6}] {f.rule_id}{' @ ' + f.location if f.location else ''}: {f.message}"
            for f in fs]


def render_annotation(v: VerifiedAnnotation, code_names: dict[str, str]) -> list[str]:
    a = v.annotation
    name = code_names.get(a.code_id, "not in codebook")
    out = [f"  {a.id:<4} {a.segment_id:<6} {a.code_id} ({name})  [{v.status}]  {a.source}"]
    out += _wrap("span: ", f'"{a.span}"')
    out += _wrap("criteria met: ", ", ".join(a.criteria_met) or "none cited")
    out += _wrap("exclusions considered: ",
                 "; ".join(f"{x.id}: {x.reason}" if x.reason else x.id for x in a.exclusions_considered) or "none")
    out += _wrap("reasoning: ", a.reasoning or "(none)")
    out += _wrap("counter-reading: ", a.counter_reading or "(none)")
    if a.alternatives:
        out += _wrap("alternatives rejected: ", ", ".join(a.alternatives))
    out += _findings(v.findings, indent="      ")
    return out


def render_agreement(report: Report) -> str:
    if not report.agreement:
        return "No agreement table: it needs both researcher and model annotations."
    t = report.agreement["totals"]
    out = [f"Agreement (descriptive): both {t['both']}, model only {t['model_only']}, "
           f"researcher only {t['researcher_only']}", ""]
    out.append(f"  {'segment':<8} {'both':<16} {'model only':<16} researcher only")
    for r in report.agreement["rows"]:
        out.append(f"  {r['segment_id']:<8} {', '.join(r['both']) or '-':<16} "
                   f"{', '.join(r['model_only']) or '-':<16} {', '.join(r['researcher_only']) or '-'}")
    return "\n".join(out)


def render_check(report: Report) -> str:
    out = ["Transcript readiness"] + (_findings(report.readiness) or ["  no findings"])
    out += ["", "Codebook"] + (_findings(report.codebook) or ["  no findings"])
    return "\n".join(out)


def render_text(report: Report) -> str:
    c = report.counts()
    model = [v for v in report.annotations if v.annotation.source == "model"]
    human = [v for v in report.annotations if v.annotation.source == "researcher"]
    out = [f"{report.tool} {report.version}"]
    if report.question:
        out.append(f"Question: {report.question}")
    out.append(f"{len(report.findings)} findings (high {c['high']}, medium {c['medium']}, low {c['low']}); "
               f"synthetic share {report.synthetic_share:.0%}")
    if report.model_note:
        out += ["", f"Note: {report.model_note}"]

    if model:
        anchored = sum(1 for v in model if v.status == "anchored")
        out += ["", f"Slow annotations by {report.enrichment.get('model') if report.enrichment else 'model'} "
                    f"(AI output, additive; the engine verified each one: {anchored} anchored, "
                    f"{len(model) - anchored} unanchored)"]
        for v in model:
            out += render_annotation(v, report.code_names)
        for n in (report.enrichment or {}).get("no_code", []):
            out += _wrap(f"{n['segment_id']}: no code fits. ", n.get("reason", ""), indent="  ")
    if human:
        out += ["", f"Researcher annotations ({len(human)})"]
        for v in human:
            out += render_annotation(v, report.code_names)

    out += ["", render_check(report)]

    for source, cov in report.coverage.items():
        counts = ", ".join(f"{k} {n}" for k, n in cov["code_counts"].items())
        out += ["", f"Coverage ({source}): {counts}"]
        if cov["uncoded_participant_segments"]:
            out.append(f"  uncoded participant segments: {', '.join(cov['uncoded_participant_segments'])}")
        if cov["unused_codes"]:
            out.append(f"  codes never applied: {', '.join(cov['unused_codes'])}")
        if cov["single_participant_codes"]:
            out.append(f"  codes from one participant only: {', '.join(cov['single_participant_codes'])}")
    if report.agreement:
        out += ["", render_agreement(report)]
    if report.review:
        out += ["", "Review record"] + _findings(report.review)
    if report.prompts and not model:
        out += ["", "Deliberation prompts (templated questions, not answers)"]
        for p in report.prompts:
            out += _wrap("", p["prompt"], indent="  ")
    return "\n".join(out)
