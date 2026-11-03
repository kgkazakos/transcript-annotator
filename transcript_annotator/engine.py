"""Deterministic core. Runs with no model (--no-llm).

Same input, same output, same order. The engine checks the transcript, the codebook and
every annotation (researcher or model) with the same rules. It never accepts, rejects or
ranks an annotation; that is the researcher's act.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional

from .models import (
    GROUNDING_FACTORS,
    ROLES,
    SYNTHETIC_KEY,
    TOOL_NAME,
    Annotation,
    Code,
    Criterion,
    ExclusionNote,
    Finding,
    Report,
    Review,
    Segment,
    Study,
    Thresholds,
    VerifiedAnnotation,
)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

_STOPWORDS = frozenset(
    "the and for are but not you all any can had her was one our out has have this that with "
    "they them their there from were been will would could should into than then when what "
    "which while who whom why how its it's also more most some such only own same too very "
    "just because about after before over under again further once these those does did doing "
    "being both each few other nor off per via where something someone".split()
)

# Spans that cannot be anchored: the annotation cannot be checked against the transcript.
_UNANCHORING_RULES = frozenset({"ANN_UNKNOWN_SEGMENT", "ANN_UNKNOWN_CODE", "SPAN_EMPTY", "SPAN_NOT_FOUND"})


# --------------------------------------------------------------------------- helpers


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, int)):
        value = [value]
    return tuple(_text(v) for v in value if _text(v))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _words(text) if len(w) >= 3 and w.lower() not in _STOPWORDS}


def normalise(text: str) -> str:
    """Normalise for verbatim matching: unicode form, quotes and dashes, case, whitespace."""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                   "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", t).strip().lower()


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _sorted(findings: Iterable[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.rule_id, f.location or ""))


def syntheticity(entry: Any, _path: frozenset[int] = frozenset()) -> float:
    """Share of an artifact that is synthetic, read recursively from `_cpr_synthetic`.

    Human source (no block) -> 0.0. Synthetic with no ancestry -> 1.0. Otherwise
    g * mean(parent syntheticity) + (1 - g), with g the human_grounding factor.
    A cycle in the ancestry counts as fully synthetic.
    """
    block = entry.get(SYNTHETIC_KEY) if isinstance(entry, dict) and SYNTHETIC_KEY in entry else entry
    if not isinstance(block, dict) or not any(k in block for k in ("tool", "ancestry", "human_grounding", "model")):
        return 0.0
    if id(entry) in _path:
        return 1.0
    ancestry = block.get("ancestry") or []
    if not isinstance(ancestry, list):
        ancestry = [ancestry]
    if not ancestry:
        return 1.0
    g = GROUNDING_FACTORS.get(str(block.get("human_grounding", "none")).strip().lower(), 0.0)
    path = _path | {id(entry)}
    parent_mean = sum(syntheticity(p, path) for p in ancestry) / len(ancestry)
    return round(max(0.0, min(1.0, g * parent_mean + (1.0 - g))), 6)


# --------------------------------------------------------------------------- parsing


def _criteria(value: Any, prefix: str) -> tuple[Criterion, ...]:
    out = []
    for i, c in enumerate(value or [], start=1):
        if isinstance(c, dict):
            out.append(Criterion(id=_text(c.get("id")) or f"{prefix}{i}", text=_text(c.get("text"))))
        elif _text(c):
            out.append(Criterion(id=f"{prefix}{i}", text=_text(c)))
    return tuple(out)


def parse_code(c: dict[str, Any]) -> Code:
    cid = _text(c.get("id"))
    return Code(
        id=cid,
        name=_text(c.get("name")) or cid,
        definition=_text(c.get("definition")),
        include=_criteria(c.get("include"), f"{cid}.i"),
        exclude=_criteria(c.get("exclude"), f"{cid}.x"),
        examples=_ids(c.get("examples")),
    )


def parse_segment(s: dict[str, Any], transcript_share: float = 0.0) -> Segment:
    own = syntheticity(s)
    return Segment(
        id=_text(s.get("id")),
        speaker=_text(s.get("speaker")),
        role=_text(s.get("role")).lower(),
        text=_text(s.get("text")),
        syntheticity=own if SYNTHETIC_KEY in s else transcript_share,
    )


def parse_annotation(a: dict[str, Any], default_id: str, source: Optional[str] = None) -> Annotation:
    notes = []
    for x in a.get("exclusions_considered") or []:
        if isinstance(x, dict):
            notes.append(ExclusionNote(id=_text(x.get("id")), reason=_text(x.get("reason"))))
        elif _text(x):
            notes.append(ExclusionNote(id=_text(x), reason=""))
    return Annotation(
        id=_text(a.get("id")) or default_id,
        segment_id=_text(a.get("segment_id")),
        code_id=_text(a.get("code_id")),
        span=_text(a.get("span")),
        source=source or (_text(a.get("source")).lower() or "researcher"),
        criteria_met=_ids(a.get("criteria_met")),
        exclusions_considered=tuple(notes),
        reasoning=_text(a.get("reasoning")),
        counter_reading=_text(a.get("counter_reading")),
        alternatives=_ids(a.get("alternatives")),
    )


def parse_annotations(items: Iterable[dict[str, Any]], prefix: str, source: Optional[str] = None) -> list[Annotation]:
    return [parse_annotation(a, f"{prefix}{i}", source) for i, a in enumerate(items or [], start=1) if isinstance(a, dict)]


def parse_reviews(items: Iterable[dict[str, Any]]) -> list[Review]:
    return [
        Review(annotation_id=_text(r.get("annotation_id")), decision=_text(r.get("decision")).lower(),
               note=_text(r.get("note")))
        for r in items or []
        if isinstance(r, dict)
    ]


def parse_study(doc: dict[str, Any]) -> Study:
    transcript = doc.get("transcript") or []
    transcript_share = syntheticity(doc) if SYNTHETIC_KEY in doc else 0.0
    if isinstance(transcript, dict):  # {"_cpr_synthetic": ..., "segments": [...]}
        if SYNTHETIC_KEY in transcript:
            transcript_share = syntheticity(transcript)
        transcript = transcript.get("segments") or []
    return Study(
        question=_text(doc.get("question")),
        codebook=[parse_code(c) for c in doc.get("codebook") or [] if isinstance(c, dict)],
        transcript=[parse_segment(s, transcript_share) for s in transcript if isinstance(s, dict)],
        annotations=parse_annotations(doc.get("annotations") or [], "r"),
    )


# --------------------------------------------------------------------------- 1. readiness


def check_transcript(segments: list[Segment], th: Thresholds = Thresholds()) -> list[Finding]:
    out: list[Finding] = []
    if not segments:
        return [Finding("TRANSCRIPT_EMPTY", "high", "The transcript has no segments.")]
    seen: Counter[str] = Counter(s.id for s in segments if s.id)
    for pos, s in enumerate(segments, start=1):
        loc = f"seg:{s.id}" if s.id else f"seg#{pos}"
        if not s.id:
            out.append(Finding("SEG_ID_MISSING", "high", f"Segment {pos} has no id, so no annotation can cite it.", loc))
        if not s.text:
            out.append(Finding("SEG_EMPTY", "medium", "Segment has no text.", loc))
        if not s.speaker:
            out.append(Finding("SPEAKER_MISSING", "low", "Segment has no speaker, so coverage by participant is incomplete.", loc))
        if s.role not in ROLES:
            out.append(Finding("ROLE_MISSING", "medium",
                               f"Role is {s.role or 'missing'!r}; expected participant or interviewer.", loc))
        n = len(_words(s.text))
        if n > th.long_segment_words:
            out.append(Finding("SEG_LONG", "low",
                               f"Segment has {n} words (over {th.long_segment_words}); a code applied here is coarse. "
                               "Consider splitting at turn or topic changes.", loc))
    for sid, k in sorted(seen.items()):
        if k > 1:
            out.append(Finding("SEG_ID_DUPLICATE", "high", f"Segment id {sid!r} appears {k} times.", f"seg:{sid}"))
    total = sum(len(_words(s.text)) for s in segments)
    asked = sum(len(_words(s.text)) for s in segments if s.role == "interviewer")
    if total and asked / total > th.interviewer_share:
        out.append(Finding("INTERVIEWER_HEAVY", "medium",
                           f"The interviewer speaks {asked / total:.0%} of the words (threshold {th.interviewer_share:.0%}). "
                           "Participant meaning may be led or thin."))
    share = synthetic_share(segments)
    if share > th.synthetic_share:
        out.append(Finding("SYNTHETIC_SHARE", "high",
                           f"{share:.0%} of the transcript is synthetic (threshold {th.synthetic_share:.0%}). "
                           "Codes would describe generated text, not people."))
    return _sorted(out)


def synthetic_share(segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    return round(sum(s.syntheticity for s in segments) / len(segments), 6)


# --------------------------------------------------------------------------- 2. codebook


def check_codebook(codes: list[Code], th: Thresholds = Thresholds()) -> list[Finding]:
    if not codes:
        return [Finding("CODEBOOK_EMPTY", "high", "The codebook has no codes. Slow annotation needs a researcher-authored codebook.")]
    out: list[Finding] = []
    seen = Counter(c.id for c in codes)
    for cid, k in sorted(seen.items()):
        if not cid:
            out.append(Finding("CODE_ID_MISSING", "high", f"{k} code(s) have no id.", "code:?"))
        elif k > 1:
            out.append(Finding("CODE_ID_DUPLICATE", "high", f"Code id {cid!r} appears {k} times.", f"code:{cid}"))
    for c in codes:
        loc = f"code:{c.id}"
        if not c.definition:
            out.append(Finding("CODE_NO_DEFINITION", "high", f"Code {c.name!r} has no definition.", loc))
        if not c.include:
            out.append(Finding("CODE_NO_INCLUDE", "medium",
                               f"Code {c.name!r} has no include criteria; annotations cannot say which criterion they meet.", loc))
        if not c.exclude:
            out.append(Finding("CODE_NO_EXCLUDE", "medium",
                               f"Code {c.name!r} has no exclude criteria; its boundary with other codes is undefined.", loc))
        crit = Counter(x.id for x in c.include + c.exclude)
        for xid, k in sorted(crit.items()):
            if k > 1:
                out.append(Finding("CRITERION_ID_DUPLICATE", "low", f"Criterion id {xid!r} appears {k} times in {c.id}.", loc))
    uniq = [c for c in codes if c.id and c.definition]
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            if a.id == b.id:
                continue
            j = jaccard(a.definition, b.definition)
            if j >= th.definition_overlap:
                out.append(Finding("CODE_DEFINITION_OVERLAP", "medium",
                                   f"Definitions of {a.id} and {b.id} overlap {j:.0%} (threshold {th.definition_overlap:.0%}). "
                                   "Coders may apply them interchangeably.", f"code:{a.id}+{b.id}"))
    return _sorted(out)


# --------------------------------------------------------------------------- 3. annotation verifier


def verify_annotation(
    a: Annotation,
    segments: dict[str, Segment],
    codes: dict[str, Code],
    th: Thresholds = Thresholds(),
    duplicate: bool = False,
) -> VerifiedAnnotation:
    out: list[Finding] = []
    loc = f"ann:{a.id}"
    seg = segments.get(a.segment_id)
    code = codes.get(a.code_id)
    if seg is None:
        out.append(Finding("ANN_UNKNOWN_SEGMENT", "high", f"Segment {a.segment_id or '(none)'!r} is not in the transcript.", loc))
    if code is None:
        out.append(Finding("ANN_UNKNOWN_CODE", "high",
                           f"Code {a.code_id or '(none)'!r} is not in the codebook. New codes are the researcher's call.", loc))
    if not a.span:
        out.append(Finding("SPAN_EMPTY", "high", "Annotation quotes no span.", loc))
    elif seg is not None and normalise(a.span) not in normalise(seg.text):
        out.append(Finding("SPAN_NOT_FOUND", "high", "The quoted span does not appear verbatim in the segment.", loc))
    if seg is not None and seg.role == "interviewer":
        out.append(Finding("SPAN_INTERVIEWER", "medium",
                           "Code applied to an interviewer turn; it describes the question, not the participant.", loc))
    if code is not None:
        include_ids = {x.id for x in code.include}
        exclude_ids = {x.id for x in code.exclude}
        if not a.criteria_met:
            out.append(Finding("NO_CRITERION", "high", f"No include criterion of {code.id} is cited.", loc))
        foreign = [c for c in a.criteria_met if c not in include_ids]
        if foreign:
            out.append(Finding("CRITERION_WRONG_CODE", "high",
                               f"Cited criteria {', '.join(foreign)} are not include criteria of {code.id}.", loc))
        considered = {x.id for x in a.exclusions_considered}
        if exclude_ids and not (considered & exclude_ids):
            out.append(Finding("EXCLUSIONS_NOT_CONSIDERED", "medium",
                               f"{code.id} has exclude criteria ({', '.join(sorted(exclude_ids))}) and none was considered.", loc))
        stray = sorted(considered - exclude_ids)
        if stray:
            out.append(Finding("EXCLUSION_WRONG_CODE", "low",
                               f"Considered exclusions {', '.join(stray)} are not exclude criteria of {code.id}.", loc))
        unknown_alt = [x for x in a.alternatives if x not in codes]
        if unknown_alt:
            out.append(Finding("ALTERNATIVE_UNKNOWN", "low",
                               f"Alternatives {', '.join(unknown_alt)} are not in the codebook.", loc))
    n = len(_words(a.reasoning))
    if n < th.thin_reasoning_words:
        out.append(Finding("REASONING_THIN", "medium",
                           f"Reasoning has {n} words (under {th.thin_reasoning_words}); the code arrives without its reasons.", loc))
    if not a.counter_reading:
        out.append(Finding("NO_COUNTER_READING", "medium", "No counter-reading: why might this not be the code?", loc))
    if duplicate:
        out.append(Finding("DUPLICATE_APPLICATION", "low", f"{a.code_id} is applied more than once to the same span.", loc))
    findings = _sorted(out)
    status = "unanchored" if any(f.rule_id in _UNANCHORING_RULES for f in findings) else "anchored"
    return VerifiedAnnotation(annotation=a, status=status, findings=findings)


def verify_annotations(study: Study, annotations: list[Annotation], th: Thresholds = Thresholds()) -> list[VerifiedAnnotation]:
    segments = {s.id: s for s in study.transcript if s.id}
    codes = {c.id: c for c in study.codebook if c.id}
    order = {s.id: i for i, s in enumerate(study.transcript)}
    seen: Counter[tuple[str, str, str, str]] = Counter()
    out = []
    for a in annotations:
        key = (a.source, a.segment_id, a.code_id, normalise(a.span))
        seen[key] += 1
        out.append(verify_annotation(a, segments, codes, th, duplicate=seen[key] > 1))
    out.sort(key=lambda v: (order.get(v.annotation.segment_id, len(order)), v.annotation.source != "researcher",
                            _natural(v.annotation.id)))
    return out


def _natural(s: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s))


# --------------------------------------------------------------------------- 4. coverage


def coverage(study: Study, annotations: list[Annotation]) -> dict[str, Any]:
    """Descriptive coverage per source. Counts only; no score."""
    by_id = {s.id: s for s in study.transcript if s.id}
    out: dict[str, Any] = {}
    for source in ("researcher", "model"):
        anns = [a for a in annotations if a.source == source]
        if not anns:
            continue
        counts = {c.id: 0 for c in study.codebook if c.id}
        speakers: dict[str, set[str]] = defaultdict(set)
        coded: set[str] = set()
        for a in anns:
            if a.code_id in counts:
                counts[a.code_id] += 1
                seg = by_id.get(a.segment_id)
                if seg is not None:
                    speakers[a.code_id].add(seg.speaker or "?")
            coded.add(a.segment_id)
        out[source] = {
            "code_counts": counts,
            "uncoded_participant_segments": [s.id for s in study.transcript if s.role == "participant" and s.id not in coded],
            "unused_codes": [cid for cid, n in counts.items() if n == 0],
            "single_participant_codes": sorted(cid for cid, sp in speakers.items() if len(sp) == 1 and counts[cid] > 1),
        }
    return out


# --------------------------------------------------------------------------- 5. agreement


def agreement(study: Study, annotations: list[Annotation]) -> Optional[dict[str, Any]]:
    """Per segment: codes applied by both, model only, researcher only. Descriptive, no winner."""
    model = [a for a in annotations if a.source == "model"]
    human = [a for a in annotations if a.source == "researcher"]
    if not model or not human:
        return None
    m: dict[str, set[str]] = defaultdict(set)
    h: dict[str, set[str]] = defaultdict(set)
    for a in model:
        m[a.segment_id].add(a.code_id)
    for a in human:
        h[a.segment_id].add(a.code_id)
    rows = []
    for s in study.transcript:
        if s.id not in m and s.id not in h:
            continue
        rows.append({
            "segment_id": s.id,
            "both": sorted(m[s.id] & h[s.id]),
            "model_only": sorted(m[s.id] - h[s.id]),
            "researcher_only": sorted(h[s.id] - m[s.id]),
        })
    totals = {k: sum(len(r[k]) for r in rows) for k in ("both", "model_only", "researcher_only")}
    return {"rows": rows, "totals": totals}


# --------------------------------------------------------------------------- 6. review


def check_reviews(verified: list[VerifiedAnnotation], reviews: list[Review]) -> list[Finding]:
    """Check the researcher's review record. Never accepts or rejects anything itself."""
    out: list[Finding] = []
    by_id = {v.annotation.id: v for v in verified}
    counts = Counter(r.annotation_id for r in reviews)
    for aid, k in sorted(counts.items()):
        if k > 1:
            out.append(Finding("REVIEW_DUPLICATE", "low", f"Annotation {aid} is reviewed {k} times; the last decision is used.", f"ann:{aid}"))
    latest = {r.annotation_id: r for r in reviews}
    for aid, r in sorted(latest.items()):
        v = by_id.get(aid)
        if v is None:
            out.append(Finding("REVIEW_UNKNOWN_ANNOTATION", "medium", f"Review names annotation {aid!r}, which is not in the run.", f"ann:{aid}"))
            continue
        high = [f.rule_id for f in v.findings if f.severity == "high"]
        if r.decision == "accept" and high and not r.note:
            out.append(Finding("REVIEW_ACCEPTED_WITH_HIGH", "high",
                               f"Accepted with open high findings ({', '.join(high)}) and no review note.", f"ann:{aid}"))
        if r.decision in ("revise", "reject") and not r.note:
            out.append(Finding("REVIEW_NO_NOTE", "low", f"Decision '{r.decision}' has no note; the reason is lost.", f"ann:{aid}"))
    unreviewed = [v.annotation.id for v in verified if v.annotation.id not in latest]
    if unreviewed:
        out.append(Finding("UNREVIEWED", "low",
                           f"{len(unreviewed)} annotation(s) not reviewed: {', '.join(unreviewed)}."))
    return _sorted(out)


# --------------------------------------------------------------------------- prompts


def deliberation_prompts(study: Study, annotations: list[Annotation]) -> list[dict[str, Any]]:
    """Templated prompts for participant segments no one has coded. Questions, not answers."""
    coded = {a.segment_id for a in annotations}
    ids = ", ".join(c.id for c in study.codebook if c.id) or "(no codes)"
    return [
        {
            "segment_id": s.id,
            "prompt": (f"Segment {s.id} ({s.speaker or 'unknown speaker'}) has no code. For each code that might "
                       f"apply ({ids}), which include criterion does a specific span meet, and which exclude "
                       "criterion would rule it out? If no code fits, write down why."),
        }
        for s in study.transcript
        if s.role == "participant" and s.id and s.text and s.id not in coded
    ]


# --------------------------------------------------------------------------- entry point


def analyze(
    doc: dict[str, Any],
    model_annotations: Optional[list[Annotation]] = None,
    reviews: Optional[list[Review]] = None,
    thresholds: Thresholds = Thresholds(),
) -> Report:
    """Run the deterministic core over a study, plus any model annotations to verify."""
    from . import __version__

    study = parse_study(doc)
    annotations = list(study.annotations) + list(model_annotations or [])
    verified = verify_annotations(study, annotations, thresholds)
    return Report(
        tool=TOOL_NAME,
        version=__version__,
        question=study.question,
        code_names={c.id: c.name for c in study.codebook if c.id},
        readiness=check_transcript(study.transcript, thresholds),
        codebook=check_codebook(study.codebook, thresholds),
        annotations=verified,
        coverage=coverage(study, annotations),
        agreement=agreement(study, annotations),
        review=check_reviews(verified, reviews) if reviews is not None else [],
        prompts=deliberation_prompts(study, annotations),
        synthetic_share=synthetic_share(study.transcript),
    )
