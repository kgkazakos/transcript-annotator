# transcript-annotator

Slow annotation: every code on a transcript arrives with its reasoning, checked against your codebook.

CPR Phase II, Frame A (Test-Time Deliberation), tool 4. It reads an interview transcript and a codebook you wrote, and surfaces proposed codes with the reasoning behind each one: the exact span, the criteria it meets, the exclusions it weighed, a counter-reading and the codes it rejected. You review the reasoning, not just the codes. It never accepts, rejects or ranks an annotation, and it never says whether anything a participant said is true. The human is the epistemic ground.

## What it does

transcript-annotator reads a research question, a codebook (codes with definitions and include and exclude criteria), a transcript (segments with speaker and role) and any annotations you have already made, and produces:

1. Slow annotations from a reasoning model. For each participant segment the model applies the codes that fit, and for each one quotes the span verbatim, cites the include criteria it meets, works through the code's exclude criteria, gives its reasoning, the strongest counter-reading and the alternative codes it rejected. Segments it leaves uncoded come with a reason. These lead the output.
2. A verifier that checks every annotation, yours and the model's, with the same rules: does the span appear in the segment, does the code exist, are the cited criteria that code's own, were its exclusions considered, is the reasoning more than a label, is there a counter-reading, is it coding an interviewer turn.
3. Readiness checks on the transcript (missing and duplicate ids, empty or unattributed segments, an interviewer who does most of the talking, segments too long to code finely, synthetic share) and on the codebook (missing definitions, no include or no exclude criteria, duplicate ids, definitions that overlap).
4. Coverage per source: how often each code is applied, which participant segments no one has coded, codes never used, codes that rest on one participant.
5. An agreement table when both you and the model have coded: per segment, the codes you both applied, the model only, and you only. Descriptive. Neither side is treated as correct.
6. A review check. You record accept, revise or reject with a note for each annotation; the tool flags annotations you accepted while they still carry a high finding and no note, decisions with no reason, and annotations you have not reviewed.

Items 2 to 6 run with no model. With `--no-llm` the tool checks the transcript and codebook, verifies your own annotations and gives you a deliberation prompt for every participant segment no one has coded.

Read the counter-readings first. A code whose counter-reading is as strong as its reasoning is where your codebook boundary needs work, or where the segment deserves a second look.

## Reasoning per code

Qualitative coding goes wrong quietly. A list of labels looks the same whether the coder applied the codebook's criteria, read the segment in context, or matched on a keyword. Slow annotation borrows from codebook thematic analysis and analytic memoing: the reasons are written next to the code, in a fixed shape, so they can be checked and argued with.

- **Your codebook, always.** The tool ships no codes. The model may only apply codes in your codebook; a code it invents is kept, flagged `ANN_UNKNOWN_CODE` and never added. New codes are your call.
- **Quoted, not paraphrased.** Every span must appear in its segment (case, whitespace and curly quotes are normalised). A span that does not is `unanchored`, and so is an annotation on a segment or code that does not exist.
- **Criteria by id.** Annotations cite include and exclude criteria by id, so the verifier can tell whether the cited criterion belongs to the code at all.
- **Source is always visible.** Every annotation is marked `researcher` or `model`, on its card and in the export. Model annotations get their own ids (`m1`, `m2`, ...); the model never edits yours.
- **Verified, not graded.** Any confidence, score or ranking the model returns is dropped on parse. The engine labels each annotation `anchored` or `unanchored` and lists its findings.
- **Reviewing is yours.** `review` checks the record of your decisions against a saved run. It never accepts or rejects anything and never calls a model.

## AI paradigm

LLM (reasoning).

Qualitative coding fails quietly when codes arrive without their reasons: a list of labels hides whether the coder applied the codebook's criteria, read the segment in context, or matched on a keyword. Slow annotation asks a reasoning model to spend deliberate compute on each segment and to show its work per code: the exact span, the codebook criterion it meets, the exclusion criteria it considered, a counter-reading and the alternative codes it rejected. That is interpretive reasoning against a researcher-authored codebook, not classification, so a reasoning model fits. The deterministic layer checks every annotation against the transcript and the codebook, and the researcher reviews the reasoning, not just the codes.

Run `tan paradigm` to print this at the tool level.

## No synthetic data, no Provenance Hash

transcript-annotator produces no synthetic data. It reads a transcript and emits annotations, findings and counts. There is no `_cpr_synthetic` block in its export, by design: model annotations are analytic labels and reasoning about your data, marked with source `model` and the model id, and are not data about people.

It does consume transcripts that may be synthetic. A `_cpr_synthetic` block on a segment, or on the whole transcript, is read recursively, using the same grounding factors as `provenance-inspector`. Above 30% of the transcript, `SYNTHETIC_SHARE` fires: codes on generated text describe the generator, not people.

## Install

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,ui]" --no-build-isolation
```

The reasoning layer uses `litellm`, a core dependency, so any provider works. If the model is unavailable, a call fails or a reply cannot be parsed, the tool shows a "model unavailable" note and the report is exactly the `--no-llm` report.

### Choosing a model

The model is picked in this order: the `--model` flag (or the Streamlit model picker), then the `CPR_LLM_MODEL` environment variable, then the default `anthropic/claude-opus-5-5`. Reasoning is the product here, so the default is a strong reasoning model; `anthropic/claude-sonnet-5-5` is cheaper and faster. Each provider needs its own key:

```bash
export ANTHROPIC_API_KEY=...   # tan run study.json --model anthropic/claude-opus-5-5
export OPENAI_API_KEY=...      # tan run study.json --model openai/gpt-5.5
export GEMINI_API_KEY=...      # tan run study.json --model gemini/gemini-3.5-flash
ollama pull llama3.1           # tan run study.json --model ollama/llama3.1   (local, no key)
```

A local model keeps the transcript on your machine, which matters for interview data.

## CLI

```bash
tan run examples/onboarding-interview.json                     # model annotates slowly, engine verifies
tan run examples/researcher-coded.json                         # your codes and the model's, side by side
tan run examples/researcher-coded.json --no-llm                # verifier, coverage and prompts only
tan run examples/onboarding-interview.json --batch-size 10     # segments per model call
tan check examples/thin-transcript.json                        # transcript and codebook checks only
tan export examples/researcher-coded.json -o run.json          # versioned JSON (cpr.tan.export/1)
tan review run.json reviews.json -o reviewed.json              # check your review decisions on a run
tan agreement run.json                                         # researcher vs model table only
tan paradigm                                                   # paradigm + rationale
```

`run` and `export` call a model and accept `--model`, `--no-llm`, `--batch-size`, `--annotations` (a separate file of your annotations) and `--reviews`. `check`, `review`, `agreement` and `paradigm` never call a model. Review and agreement read a saved run, so model annotation ids stay stable while you review.

Thresholds, on `run`, `export` and `check`:

| Option | Default | Rule |
|--------|---------|------|
| `--interviewer-share` | 0.4 | `INTERVIEWER_HEAVY` fires when the interviewer's share of words is above this |
| `--long-segment` | 250 | `SEG_LONG` fires above this many words |
| `--definition-overlap` | 0.6 | `CODE_DEFINITION_OVERLAP` fires when two definitions' content words overlap (Jaccard) at or above this |
| `--thin-reasoning` | 15 | `REASONING_THIN` fires below this many words |

## Streamlit

```bash
streamlit run transcript_annotator/app.py
```

Frame A visual language: deep violet accent `#4f3d8f`, DM Sans and DM Mono, near-black sidebar, warm-beige controls, cream main, card rows with severity chips. The sidebar holds a model picker with Anthropic, OpenAI, Google and local Ollama presets, plus a free-text field for any litellm id; the controls column holds the input and the four thresholds. The transcript renders segment by segment with coded spans highlighted, each followed by its reasoning cards (source, anchoring and finding chips, model output labelled as additive with the model id) and an accept, revise or reject control with a note. Your review is checked as you go and downloads as `reviews.json` or a full run export. Readiness, codebook, coverage and agreement sit in tabs below.

## The verifier

The engine is a set of structural checks over ids, spans, criteria and word counts, honest about being one. It is fast, reproducible and cheap, and it is not an oracle: it cannot tell whether a code is right, only whether each annotation is anchored to the text, tied to the codebook and candid about the reading it rejects. Output depends only on the input and the thresholds.

| Rule | Stage | Fires when | Severity |
|------|-------|-----------|----------|
| `TRANSCRIPT_EMPTY` | readiness | The transcript has no segments | high |
| `SEG_ID_MISSING` | readiness | A segment has no id (shown as `seg#n`) | high |
| `SEG_ID_DUPLICATE` | readiness | Two segments share an id | high |
| `SEG_EMPTY` | readiness | A segment has no text | medium |
| `SPEAKER_MISSING` | readiness | A segment has no speaker | low |
| `ROLE_MISSING` | readiness | Role is not `participant` or `interviewer` | medium |
| `SEG_LONG` | readiness | A segment is longer than `--long-segment` words | low |
| `INTERVIEWER_HEAVY` | readiness | Interviewer share of words above `--interviewer-share` | medium |
| `SYNTHETIC_SHARE` | readiness | Synthetic share of the transcript above 30% | high |
| `CODEBOOK_EMPTY` | codebook | The codebook has no codes | high |
| `CODE_ID_MISSING` | codebook | A code has no id | high |
| `CODE_ID_DUPLICATE` | codebook | Two codes share an id | high |
| `CODE_NO_DEFINITION` | codebook | A code has no definition | high |
| `CODE_NO_INCLUDE` | codebook | A code has no include criteria | medium |
| `CODE_NO_EXCLUDE` | codebook | A code has no exclude criteria, so its boundary is undefined | medium |
| `CRITERION_ID_DUPLICATE` | codebook | Two criteria in one code share an id | low |
| `CODE_DEFINITION_OVERLAP` | codebook | Two definitions overlap at or above `--definition-overlap` | medium |
| `ANN_UNKNOWN_SEGMENT` | verifier | The annotation cites a segment not in the transcript (unanchored) | high |
| `ANN_UNKNOWN_CODE` | verifier | The annotation applies a code not in the codebook (unanchored) | high |
| `SPAN_EMPTY` | verifier | No span is quoted (unanchored) | high |
| `SPAN_NOT_FOUND` | verifier | The span does not appear verbatim in the segment (unanchored) | high |
| `SPAN_INTERVIEWER` | verifier | The code is applied to an interviewer turn | medium |
| `NO_CRITERION` | verifier | No include criterion is cited | high |
| `CRITERION_WRONG_CODE` | verifier | A cited criterion is not an include criterion of this code | high |
| `EXCLUSIONS_NOT_CONSIDERED` | verifier | The code has exclude criteria and none was considered | medium |
| `EXCLUSION_WRONG_CODE` | verifier | A considered exclusion belongs to another code | low |
| `REASONING_THIN` | verifier | Reasoning is shorter than `--thin-reasoning` words | medium |
| `NO_COUNTER_READING` | verifier | No counter-reading is given | medium |
| `ALTERNATIVE_UNKNOWN` | verifier | A rejected alternative is not in the codebook | low |
| `DUPLICATE_APPLICATION` | verifier | The same source applies the same code to the same span twice | low |
| `REVIEW_ACCEPTED_WITH_HIGH` | review | Accepted while a high finding is open, with no note | high |
| `REVIEW_UNKNOWN_ANNOTATION` | review | A review names an annotation not in the run | medium |
| `REVIEW_NO_NOTE` | review | A revise or reject decision has no note | low |
| `REVIEW_DUPLICATE` | review | One annotation is reviewed more than once (the last decision is used) | low |
| `UNREVIEWED` | review | Annotations in the run have no review decision | low |

Accepting an annotation with an open high finding is allowed. The rule asks only that you say why.

## Input format

A JSON object with `question`, `codebook`, `transcript` and, optionally, `annotations`. Codes take `id`, `name`, `definition`, `include` and `exclude` (lists of `{id, text}`; plain strings get ids like `NEXT.i1`) and optional `examples`. Segments take `id`, `speaker`, `role` and `text`. Annotations in an input file are the researcher's unless marked otherwise; missing ids default to `r1`, `r2` and so on.

```json
{
  "question": "How do first-time workspace admins make sense of initial setup?",
  "codebook": [
    {"id": "NEXT", "name": "Unclear next step",
     "definition": "The participant describes not knowing which action the product expected of them next.",
     "include": [{"id": "NEXT.i1", "text": "Says they did not know what to do next or where to go"}],
     "exclude": [{"id": "NEXT.x1", "text": "Knew the next step but could not take it for lack of access"}]}
  ],
  "transcript": [
    {"id": "s1", "speaker": "INT", "role": "interviewer",
     "text": "Walk me through the first hour."},
    {"id": "s2", "speaker": "P1", "role": "participant",
     "text": "Once it closed I was on an empty dashboard and I just did not know where to go."}
  ],
  "annotations": [
    {"segment_id": "s2", "code_id": "NEXT", "span": "I just did not know where to go",
     "criteria_met": ["NEXT.i1"],
     "exclusions_considered": [{"id": "NEXT.x1", "reason": "No access problem is mentioned."}],
     "reasoning": "P1 names the moment the welcome screen closed and says in plain words they did not know where to go.",
     "counter_reading": "This could be a general complaint about the empty dashboard rather than a moment of not knowing.",
     "alternatives": []}
  ]
}
```

A segment, or the transcript as a whole (`{"_cpr_synthetic": {...}, "segments": [...]}`), may carry a `_cpr_synthetic` block from an upstream CPR tool; its ancestry feeds `SYNTHETIC_SHARE`. A review file is a list, or `{"reviews": [...]}`, of `{annotation_id, decision, note}` with decision `accept`, `revise` or `reject`.

See `examples/` for:

- `onboarding-interview.json`: a paired interview with two first-time admins, ten segments and a four-code codebook, with no annotations. The model codes it; under `--no-llm` it prints a deliberation prompt per participant segment. Fires nothing.
- `researcher-coded.json`: the same interview with four researcher annotations, all anchored. With a model, it produces the agreement table.
- `flawed-annotations.json`: the same interview with annotations that break every verifier rule, one rule each.
- `reviews/flawed-annotations.reviews.json`: review decisions for a run of `flawed-annotations.json` that fire every review rule.
- `thin-transcript.json`: a short, partly synthetic transcript led by a long interviewer preamble, with a missing, a duplicated and an empty segment (every readiness rule).
- `weak-codebook.json`: a codebook with duplicate and missing ids, no definition, missing criteria and two near-identical definitions (every codebook rule).

## Export schema

`cpr.tan.export/1`: `schema`, `exported_at`, `paradigm`, `model` (the id that wrote any model annotations, or null) and `report`. The report holds `question`, `code_names`, `readiness`, `codebook`, `annotations` (each with the annotation, its `source`, its `status` and its `findings`), `coverage` (per source), `agreement`, `review`, `prompts`, `synthetic_share`, `enrichment` (model id, batch count, annotation count and the segments the model left uncoded, with reasons) and `model_note`. There is no confidence, score or ranking field.

## Build

- Model-first: a reasoning model's slow annotations lead, a rule-based verifier checks every annotation with the same rules, and the tool falls back to the `--no-llm` report when the model is unavailable.
- `paradigm` command, and `--no-llm` for the deterministic checks alone.
- pytest suite, CI on Python 3.10, 3.11, 3.12.
- Versioned export `cpr.tan.export/1`.

## IP

A single-model annotation tool: no synthetic data produced, no agents, no orchestration, no automatic linkage to external systems. One model proposes codes from your codebook with its reasons; the engine checks them against your transcript and codebook, and you decide.

## License

MIT.
