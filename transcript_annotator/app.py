"""Streamlit UI. Run: streamlit run transcript_annotator/app.py"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, so relative imports break. Use the repo root + absolute import.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from transcript_annotator import __version__  # noqa: E402
from transcript_annotator.engine import (  # noqa: E402
    MODEL_UNAVAILABLE,
    build_report,
    check_reviews,
    parse_study,
)
from transcript_annotator.io import to_export  # noqa: E402
from transcript_annotator.llm import DEFAULT_MODEL, resolve_model  # noqa: E402  (litellm loads lazily)
from transcript_annotator.models import (  # noqa: E402
    AI_PARADIGM,
    AI_PARADIGM_RATIONALE,
    EXPORT_SCHEMA,
    TOOL_NAME,
    Review,
    Thresholds,
)

ACCENT = "#4f3d8f"  # Frame A violet, keep in sync with .streamlit/config.toml
CHIP = {"high": "#9b2c2c", "medium": "#b7791f", "low": "#4a5568"}
STATUS_CHIP = {"anchored": "#2f6b4f", "unanchored": "#7a4b2a"}
SOURCE_CHIP = {"researcher": "#3d3a35", "model": ACCENT}
EXAMPLES = sorted((_ROOT / "examples").glob("*.json"))
# litellm ids, checked against litellm's model list. Each needs its provider key in the environment.
MODEL_PRESETS = {
    "Anthropic · Claude Opus 5.5 (default)": DEFAULT_MODEL,
    "Anthropic · Claude Sonnet 5.5": "anthropic/claude-sonnet-5-5",
    "OpenAI · GPT-5.5": "openai/gpt-5.5",
    "Google · Gemini 3.5 Flash": "gemini/gemini-3.5-flash",
    "Local · Ollama llama3.1": "ollama/llama3.1",
    "Other (type a litellm id)": None,
}
_D = Thresholds()

st.set_page_config(page_title=TOOL_NAME, layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
code, .mono {{ font-family: 'DM Mono', monospace; }}
.stApp {{ background: #fbfaf8; }}
section[data-testid="stSidebar"] {{ background: #16161a; }}
section[data-testid="stSidebar"] * {{ color: #ece9e2 !important; }}
div[data-testid="column"]:first-child > div {{ background: #f2ede1; border-radius: 10px; padding: 8px 12px; }}
.card {{ background: #fff; border: 1px solid #e8e2d6; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }}
.seg {{ border-left: 4px solid #d9d2c3; }}
.seg.participant {{ border-left-color: {ACCENT}; }}
.ann {{ margin-left: 24px; border-top: 3px solid {ACCENT}; }}
.ann.researcher {{ border-top-color: #3d3a35; }}
.chip {{ display: inline-block; padding: 1px 8px; border-radius: 999px; color: #fff; font-size: 12px; margin: 0 6px 4px 0; }}
.muted {{ color: #7a746a; font-size: 13px; }}
.field {{ margin-top: 6px; font-size: 14px; }}
.accent {{ color: {ACCENT}; }}
mark {{ background: #e6e0f3; padding: 0 2px; border-radius: 3px; }}
table.ag {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
table.ag th, table.ag td {{ border-bottom: 1px solid #ece6da; padding: 6px 10px; text-align: left; }}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {TOOL_NAME}")
    st.caption(f"v{__version__} · {AI_PARADIGM}")
    with st.expander("Why this paradigm"):
        st.write(AI_PARADIGM_RATIONALE)
    st.markdown("#### Model")
    use_ai = st.toggle("Slow annotation (model)", value=True,
                       help="Off is the --no-llm report: checks, your own annotations and deliberation prompts.")
    env_model = resolve_model(None)
    labels = list(MODEL_PRESETS)
    start = next((i for i, k in enumerate(labels) if MODEL_PRESETS[k] == env_model), len(labels) - 1)
    preset = st.selectbox("Preset", labels, index=start, disabled=not use_ai)
    custom = st.text_input("litellm model id", value="" if MODEL_PRESETS[preset] else env_model,
                           placeholder="provider/model", disabled=not use_ai or MODEL_PRESETS[preset] is not None)
    model = MODEL_PRESETS[preset] or custom.strip() or None
    batch_size = st.number_input("Segments per call", 1, 200, 20, disabled=not use_ai)
    st.caption(f"Using `{resolve_model(model)}`. Precedence: picker, then CPR_LLM_MODEL, then {DEFAULT_MODEL}.")
    st.caption("The model proposes codes with their reasons. You review the reasoning and decide.")

controls, main = st.columns([1, 2.6])
with controls:
    st.markdown("#### Input")
    names = [p.name for p in EXAMPLES]
    choice = st.selectbox("Example", names) if names else None
    default = (_ROOT / "examples" / choice).read_text(encoding="utf-8") if choice else '{"codebook": [], "transcript": []}'
    raw = st.text_area("Question, codebook, transcript, your annotations (JSON)", value=default, height=340,
                       key=f"raw-{choice}")
    st.markdown("#### Thresholds")
    interviewer = st.slider("Interviewer share of words above", 0.0, 1.0, _D.interviewer_share, 0.05)
    long_words = st.number_input("Long segment above (words)", 20, 2000, _D.long_segment_words)
    overlap = st.slider("Definition overlap above", 0.0, 1.0, _D.definition_overlap, 0.05)
    thin = st.number_input("Thin reasoning under (words)", 0, 200, _D.thin_reasoning_words)
    thresholds = Thresholds(interviewer, int(long_words), overlap, int(thin))


@st.cache_data(show_spinner=False)
def annotated(raw_json: str, t: Thresholds, model_id, size: int, ai: bool):
    """Cached so a rerun with unchanged inputs does not call the model again."""
    return build_report(json.loads(raw_json), no_llm=not ai, model=model_id, batch_size=size, thresholds=t)


def chip(label: str, color: str) -> str:
    return f"<span class='chip' style='background:{color}'>{html.escape(label)}</span>"


def esc(text: str) -> str:
    return html.escape(text or "")


def highlight(text: str, spans: list[str]) -> str:
    """Mark each quoted span in the segment text (case and whitespace tolerant)."""
    marks: list[tuple[int, int]] = []
    for s in spans:
        words = [re.escape(w) for w in s.split()]
        m = re.search(r"\s+".join(words), text, flags=re.IGNORECASE) if words else None
        if m:
            marks.append(m.span())
    out, pos = [], 0
    for a, b in sorted(marks):
        if a < pos:
            continue
        out += [esc(text[pos:a]), f"<mark>{esc(text[a:b])}</mark>"]
        pos = b
    return "".join(out) + esc(text[pos:])


def finding_card(f) -> str:
    return (f"<div class='card'>{chip(f.severity, CHIP[f.severity])}<span class='mono'>{f.rule_id}</span>"
            f"{' <span class=muted>' + esc(f.location) + '</span>' if f.location else ''} {esc(f.message)}</div>")


with main:
    try:
        doc = json.loads(raw)
        if not isinstance(doc, dict):
            raise ValueError("Expected a JSON object with 'codebook' and 'transcript'.")
        study = parse_study(doc)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner(f"Annotating slowly with {resolve_model(model)}..." if use_ai else "Checking..."):
        report = annotated(raw, thresholds, model, int(batch_size), use_ai)

    st.markdown("<h3 class='accent'>Annotations, with their reasoning</h3>", unsafe_allow_html=True)
    if report.model_note and report.model_note.startswith(MODEL_UNAVAILABLE):
        st.warning(f"{report.model_note[0].upper()}{report.model_note[1:]} Check the model id and its provider key. "
                   "The checks and your own annotations below are unchanged.")
    model_id = (report.enrichment or {}).get("model")
    st.caption(f"{esc(report.question) or 'No question given'} · {len(study.transcript)} segments, "
               f"{len(study.codebook)} codes · synthetic {report.synthetic_share:.0%}"
               + (f" · model annotations by {model_id} (AI output, additive, verified by the engine)"
                  if model_id else ""))

    by_segment: dict[str, list] = {}
    for v in report.annotations:
        by_segment.setdefault(v.annotation.segment_id, []).append(v)
    no_code = {n["segment_id"]: n["reason"] for n in (report.enrichment or {}).get("no_code", [])}
    prompts = {p["segment_id"]: p["prompt"] for p in report.prompts}
    reviews: list[Review] = []

    def review_control(annotation_id: str) -> None:
        c1, c2 = st.columns([1, 2.5])
        decision = c1.radio("Your review", ["not yet", "accept", "revise", "reject"], horizontal=True,
                            key=f"d-{choice}-{annotation_id}", label_visibility="collapsed")
        note = c2.text_input("Note", key=f"n-{choice}-{annotation_id}", placeholder="Review note (why)",
                             label_visibility="collapsed")
        if decision != "not yet":
            reviews.append(Review(annotation_id, decision, note.strip()))

    known = {s.id for s in study.transcript}
    for s in study.transcript:
        anns = by_segment.get(s.id, [])
        st.markdown(
            f"<div class='card seg {esc(s.role)}'><span class='mono'>{esc(s.id)}</span> "
            f"<b>{esc(s.speaker) or '?'}</b> <span class='muted'>{esc(s.role)}</span>"
            f"<div class='field'>{highlight(s.text, [v.annotation.span for v in anns])}</div></div>",
            unsafe_allow_html=True,
        )
        if s.id in no_code:
            st.markdown(f"<div class='card ann'><span class='muted'>Model: no code fits.</span> "
                        f"{esc(no_code[s.id])}</div>", unsafe_allow_html=True)
        elif s.id in prompts and not anns:
            st.markdown(f"<div class='card ann researcher'><span class='muted'>Deliberation prompt</span> "
                        f"{esc(prompts[s.id])}</div>", unsafe_allow_html=True)
        for v in anns:
            a = v.annotation
            name = report.code_names.get(a.code_id, "not in codebook")
            excl = "; ".join(f"<span class='mono'>{esc(x.id)}</span> {esc(x.reason)}"
                             for x in a.exclusions_considered) or "none considered"
            flags = "".join(chip(f.rule_id, CHIP[f.severity]) for f in v.findings)
            st.markdown(
                f"<div class='card ann {a.source}'><span class='mono'>{esc(a.id)}</span> "
                f"{chip(a.source, SOURCE_CHIP[a.source])}{chip(v.status, STATUS_CHIP[v.status])}"
                f"<b><span class='mono'>{esc(a.code_id)}</span> {esc(name)}</b>"
                f"<div class='field'><span class='muted'>Span</span> &ldquo;{esc(a.span)}&rdquo;</div>"
                f"<div class='field'><span class='muted'>Criteria met</span> "
                f"<span class='mono'>{esc(', '.join(a.criteria_met)) or 'none cited'}</span></div>"
                f"<div class='field'><span class='muted'>Exclusions considered</span> {excl}</div>"
                f"<div class='field'><span class='muted'>Reasoning</span> {esc(a.reasoning) or '(none)'}</div>"
                f"<div class='field'><span class='muted'>Counter-reading</span> {esc(a.counter_reading) or '(none)'}</div>"
                + (f"<div class='field'><span class='muted'>Alternatives rejected</span> "
                   f"<span class='mono'>{esc(', '.join(a.alternatives))}</span></div>" if a.alternatives else "")
                + (f"<div class='field'>{flags}</div>" if flags else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            review_control(a.id)
    stray = [v for v in report.annotations if v.annotation.segment_id not in known]
    if stray:
        st.markdown("**Annotations on segments not in the transcript**")
    for v in stray:
        a = v.annotation
        st.markdown(f"<div class='card ann {a.source}'><span class='mono'>{esc(a.id)} {esc(a.segment_id)} "
                    f"{esc(a.code_id)}</span> &ldquo;{esc(a.span)}&rdquo;<div class='field'>"
                    + "".join(chip(f.rule_id, CHIP[f.severity]) for f in v.findings) + "</div></div>",
                    unsafe_allow_html=True)
        review_control(a.id)

    st.markdown("#### Your review")
    report.review = check_reviews(report.annotations, reviews) if report.annotations else []
    st.caption("Decisions are yours. The engine only checks the record: accepted annotations with open "
               "high findings and no note, decisions without reasons, annotations not yet reviewed.")
    for f in report.review:
        st.markdown(finding_card(f), unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download reviews.json",
                       json.dumps({"reviews": [r.__dict__ for r in reviews]}, indent=2),
                       file_name="reviews.json", mime="application/json")
    d2.download_button(f"Download run export ({EXPORT_SCHEMA})", json.dumps(to_export(report), indent=2),
                       file_name="run.json", mime="application/json")

    tabs = st.tabs([f"Readiness ({len(report.readiness)})", f"Codebook ({len(report.codebook)})",
                    "Coverage", "Agreement"])
    with tabs[0]:
        for f in report.readiness:
            st.markdown(finding_card(f), unsafe_allow_html=True)
        if not report.readiness:
            st.write("No readiness findings.")
    with tabs[1]:
        for f in report.codebook:
            st.markdown(finding_card(f), unsafe_allow_html=True)
        if not report.codebook:
            st.write("No codebook findings.")
    with tabs[2]:
        if not report.coverage:
            st.write("No annotations yet.")
        for source, cov in report.coverage.items():
            st.markdown(f"**{source.capitalize()}**")
            st.markdown(" · ".join(f"<span class='mono'>{esc(k)}</span> {n}" for k, n in cov["code_counts"].items()),
                        unsafe_allow_html=True)
            for label, key in (("Uncoded participant segments", "uncoded_participant_segments"),
                               ("Codes never applied", "unused_codes"),
                               ("Codes from one participant only", "single_participant_codes")):
                if cov[key]:
                    st.markdown(f"<span class='muted'>{label}</span> <span class='mono'>{esc(', '.join(cov[key]))}"
                                f"</span>", unsafe_allow_html=True)
    with tabs[3]:
        if not report.agreement:
            st.write("Needs both your annotations and the model's.")
        else:
            t = report.agreement["totals"]
            st.caption(f"Descriptive: both {t['both']}, model only {t['model_only']}, researcher only "
                       f"{t['researcher_only']}. Neither side is treated as correct.")
            rows = "".join(
                f"<tr><td class='mono'>{esc(r['segment_id'])}</td><td class='mono'>{esc(', '.join(r['both']))}</td>"
                f"<td class='mono'>{esc(', '.join(r['model_only']))}</td>"
                f"<td class='mono'>{esc(', '.join(r['researcher_only']))}</td></tr>"
                for r in report.agreement["rows"]
            )
            st.markdown(f"<table class='ag'><tr><th>segment</th><th>both</th><th>model only</th>"
                        f"<th>researcher only</th></tr>{rows}</table>", unsafe_allow_html=True)
