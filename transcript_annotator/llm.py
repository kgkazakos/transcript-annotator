"""Model layer for transcript-annotator: slow annotation with the reasoning shown per code.

One model reads the researcher's codebook and the transcript in consecutive batches and
proposes annotations, each with its span, the criteria it meets, the exclusions it considered,
its reasoning, a counter-reading and the alternatives it rejected. The engine verifies every
one. Any confidence or score the model returns is discarded. On any failure or unparseable
reply, the caller falls back to the --no-llm report with a visible note.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .engine import parse_annotations, parse_study
from .models import Annotation

DEFAULT_MODEL = "anthropic/claude-opus-5-5"
DEFAULT_BATCH_SIZE = 20

SYSTEM_PROMPT = """You are helping a human researcher code an interview transcript slowly, \
against the researcher's own codebook. You propose annotations; the researcher reviews your \
reasoning and decides. You never decide.

For each participant segment, read it in context and decide which codebook codes, if any, \
apply. For every code you apply, show your work:
- span: an exact, verbatim quote from that segment's text. Copy it character for character. \
Never paraphrase, merge segments or invent words.
- criteria_met: the ids of the code's include criteria the span meets. Use only ids from the codebook.
- exclusions_considered: each of the code's exclude criteria, by id, with one line on why it \
does or does not rule the code out.
- reasoning: two to four sentences linking the span to the criteria.
- counter_reading: the strongest reason this might not be the code.
- alternatives: ids of other codebook codes you considered and rejected.

Rules:
- Use only codes in the codebook. Do not invent codes. If no code fits a participant segment, \
add it to no_code with a short reason.
- Do not code interviewer turns.
- Do not invent quotes, data, citations or facts about participants.
- Do not judge participants, and do not state whether anything they say is true.
- Do not give confidence scores, ratings or rankings.

Reply with JSON only, in this shape:
{"annotations": [{"segment_id": "...", "code_id": "...", "span": "...", "criteria_met": ["..."], \
"exclusions_considered": [{"id": "...", "reason": "..."}], "reasoning": "...", \
"counter_reading": "...", "alternatives": ["..."]}],
 "no_code": [{"segment_id": "...", "reason": "..."}]}"""


def resolve_model(model: Optional[str] = None) -> str:
    """--model flag, then CPR_LLM_MODEL, then the default."""
    return model or os.environ.get("CPR_LLM_MODEL") or DEFAULT_MODEL


@dataclass
class ModelResult:
    model: str
    annotations: list[Annotation] = field(default_factory=list)
    no_code: list[dict[str, str]] = field(default_factory=list)
    batches: int = 0


class ModelReplyError(ValueError):
    """The model replied, but not with the JSON shape asked for."""


@contextlib.contextmanager
def _silenced():
    """litellm prints banners to stdout/stderr that corrupt JSON output. Swallow them."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _call_model(system: str, user: str, model: str) -> str:
    import litellm  # lazy: --no-llm paths never import it

    with _silenced():
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
    return resp["choices"][0]["message"]["content"]


def parse_reply(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply (tolerates code fences and leading prose)."""
    if not isinstance(text, str):
        raise ModelReplyError("Reply is not text.")
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ModelReplyError("Reply holds no JSON object.")
    try:
        data = json.loads(t[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ModelReplyError(f"Reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("annotations", []), list):
        raise ModelReplyError("Reply has no 'annotations' list.")
    return data


def _codebook_block(doc: dict[str, Any]) -> list[dict[str, Any]]:
    study = parse_study(doc)
    return [
        {"id": c.id, "name": c.name, "definition": c.definition,
         "include": [{"id": x.id, "text": x.text} for x in c.include],
         "exclude": [{"id": x.id, "text": x.text} for x in c.exclude],
         "examples": list(c.examples)}
        for c in study.codebook
    ]


def build_prompt(doc: dict[str, Any], batch: list[dict[str, Any]]) -> str:
    study = parse_study(doc)
    return json.dumps({
        "research_question": study.question,
        "codebook": _codebook_block(doc),
        "segments": batch,
    }, indent=1, ensure_ascii=False)


def annotate(doc: dict[str, Any], model: Optional[str] = None, batch_size: int = DEFAULT_BATCH_SIZE) -> ModelResult:
    """Ask one model for slow annotations, batch by batch. Raises on any failure.

    The caller (see `build_report`) catches and degrades to the --no-llm report.
    """
    model = resolve_model(model)
    study = parse_study(doc)
    segments = [{"id": s.id, "speaker": s.speaker, "role": s.role, "text": s.text} for s in study.transcript]
    size = max(1, int(batch_size))
    result = ModelResult(model=model)
    raw: list[dict[str, Any]] = []
    for i in range(0, len(segments), size):
        batch = segments[i:i + size]
        if not any(s["role"] == "participant" for s in batch):
            continue
        reply = parse_reply(_call_model(SYSTEM_PROMPT, build_prompt(doc, batch), model))
        raw += [a for a in reply.get("annotations", []) if isinstance(a, dict)]
        result.no_code += [
            {"segment_id": str(n.get("segment_id", "")), "reason": str(n.get("reason", ""))}
            for n in reply.get("no_code", []) or [] if isinstance(n, dict)
        ]
        result.batches += 1
    # Keep only the fields the engine verifies; any confidence or score is discarded here.
    result.annotations = parse_annotations(raw, "m", source="model")
    return result
