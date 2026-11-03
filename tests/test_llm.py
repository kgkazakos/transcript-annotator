import json
import sys

import pytest

from transcript_annotator import llm
from transcript_annotator.engine import MODEL_UNAVAILABLE, build_report

from test_engine import GOOD, ann, study


def reply(*annotations, no_code=()):
    return json.dumps({"annotations": list(annotations), "no_code": list(no_code)})


def without_note(report):
    d = report.to_dict()
    d["model_note"] = None
    return d


def test_model_precedence(monkeypatch):
    monkeypatch.delenv("CPR_LLM_MODEL", raising=False)
    assert llm.resolve_model() == llm.DEFAULT_MODEL == "anthropic/claude-opus-5-5"
    monkeypatch.setenv("CPR_LLM_MODEL", "openai/gpt-5.5")
    assert llm.resolve_model() == "openai/gpt-5.5"
    assert llm.resolve_model("ollama/llama3.1") == "ollama/llama3.1"


@pytest.mark.parametrize("failure", ["raise", "garbage", "wrong-shape"])
def test_failure_degrades_to_no_llm_report(monkeypatch, failure):
    def fake(system, user, model):
        if failure == "raise":
            raise RuntimeError("backend down")
        return "not json at all" if failure == "garbage" else json.dumps({"annotations": "nope"})

    monkeypatch.setattr(llm, "_call_model", fake)
    doc = study(annotations=[GOOD])
    degraded = build_report(doc, model="any/model")
    baseline = build_report(doc, no_llm=True)
    assert degraded.model_note.startswith(MODEL_UNAVAILABLE)
    assert degraded.enrichment is None
    assert without_note(degraded) == baseline.to_dict()


def test_model_annotations_are_verified_by_engine(monkeypatch):
    invented = ann(span="I loved every minute", confidence=0.99, score=5)
    monkeypatch.setattr(llm, "_call_model", lambda s, u, m: reply(GOOD, invented))
    r = build_report(study(), model="x/y")
    model = [v for v in r.annotations if v.annotation.source == "model"]
    assert [v.annotation.id for v in model] == ["m1", "m2"]
    assert [v.status for v in model] == ["anchored", "unanchored"]
    assert r.enrichment["model"] == "x/y"
    dumped = json.dumps(r.to_dict())
    assert "confidence" not in dumped and "0.99" not in dumped  # model self-grades discarded


def test_invented_code_is_flagged_not_added(monkeypatch):
    monkeypatch.setattr(llm, "_call_model", lambda s, u, m: reply(ann(code_id="NEW")))
    r = build_report(study(), model="x/y")
    assert r.annotations[0].status == "unanchored"
    assert "ANN_UNKNOWN_CODE" in {f.rule_id for f in r.annotations[0].findings}
    assert "NEW" not in r.code_names


def test_batching_and_no_code(monkeypatch):
    calls = []

    def fake(system, user, model):
        calls.append(json.loads(user))
        return reply(no_code=[{"segment_id": "s4", "reason": "Too general to code."}]) if len(calls) == 2 else reply(GOOD)

    monkeypatch.setattr(llm, "_call_model", fake)
    r = build_report(study(), model="x/y", batch_size=2)
    assert [[s["id"] for s in c["segments"]] for c in calls] == [["s1", "s2"], ["s3", "s4"]]
    assert all(c["codebook"][0]["include"][0]["id"] == "C1.i1" for c in calls)
    assert r.enrichment["batches"] == 2
    assert r.enrichment["no_code"] == [{"segment_id": "s4", "reason": "Too general to code."}]


def test_interviewer_only_batch_is_skipped(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_call_model", lambda s, u, m: calls.append(u) or reply())
    build_report(study(), model="x/y", batch_size=1)
    assert len(calls) == 3  # s1 is the interviewer's turn


def test_parse_reply_tolerates_fences():
    assert llm.parse_reply("```json\n" + reply(GOOD) + "\n```")["annotations"][0]["code_id"] == "C1"
    with pytest.raises(llm.ModelReplyError):
        llm.parse_reply("{broken")


def test_prompt_forbids_invention_and_verdicts():
    p = llm.SYSTEM_PROMPT
    assert "Do not invent codes" in p and "Do not invent quotes" in p
    assert "whether anything they say is true" in p and "confidence" in p


def test_no_llm_does_not_import_litellm(monkeypatch):
    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    build_report(study(), no_llm=True)
    assert "litellm" not in sys.modules
