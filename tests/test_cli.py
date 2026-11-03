import json

import pytest
from click.testing import CliRunner

from transcript_annotator import llm
from transcript_annotator.cli import main
from transcript_annotator.models import AI_PARADIGM, EXPORT_SCHEMA

from test_engine import GOOD, ann, study


@pytest.fixture
def files(tmp_path):
    s = tmp_path / "study.json"
    s.write_text(json.dumps(study()))
    a = tmp_path / "annotations.json"
    a.write_text(json.dumps({"annotations": [GOOD, ann(span="not there")]}))
    rv = tmp_path / "reviews.json"
    rv.write_text(json.dumps([{"annotation_id": "r2", "decision": "accept"}]))
    return {"study": str(s), "annotations": str(a), "reviews": str(rv), "dir": tmp_path}


def invoke(*args):
    r = CliRunner().invoke(main, list(args))
    assert r.exit_code == 0, r.output
    return r.output


def test_paradigm_and_version():
    assert AI_PARADIGM in invoke("paradigm")
    assert "transcript-annotator" in invoke("--version")


def test_run_no_llm_json_is_valid_and_versioned(files):
    payload = json.loads(invoke("run", files["study"], "--no-llm", "--format", "json"))
    assert payload["schema"] == EXPORT_SCHEMA
    assert payload["model"] is None
    assert payload["report"]["enrichment"] is None
    assert "_cpr_synthetic" not in payload


def test_run_no_llm_never_calls_model(files, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model called under --no-llm")

    monkeypatch.setattr(llm, "_call_model", boom)
    assert "Deliberation prompts" in invoke("run", files["study"], "--no-llm")


def test_run_with_researcher_annotations_and_reviews(files):
    out = invoke("run", files["study"], "--no-llm", "--annotations", files["annotations"],
                 "--reviews", files["reviews"])
    assert "Researcher annotations (2)" in out
    assert "SPAN_NOT_FOUND" in out
    assert "REVIEW_ACCEPTED_WITH_HIGH" in out


def test_export_then_review_and_agreement(files, monkeypatch):
    reply = json.dumps({"annotations": [GOOD], "no_code": []})
    monkeypatch.setattr(llm, "_call_model", lambda s, u, m: reply)
    run = files["dir"] / "run.json"
    invoke("export", files["study"], "--annotations", files["annotations"], "--model", "x/y", "-o", str(run))
    data = json.loads(run.read_text())
    assert data["model"] == "x/y"
    assert [v["annotation"]["id"] for v in data["report"]["annotations"]] == ["r1", "r2", "m1"]

    out = invoke("review", str(run), files["reviews"], "-o", str(files["dir"] / "reviewed.json"))
    assert "REVIEW_ACCEPTED_WITH_HIGH" in out and "UNREVIEWED" in out
    reviewed = json.loads((files["dir"] / "reviewed.json").read_text())
    assert reviewed["report"]["review"]

    table = invoke("agreement", str(run))
    assert "both 1" in table


def test_check_never_calls_model(files):
    out = invoke("check", files["study"])
    assert "Transcript readiness" in out and "Codebook" in out


def test_run_rejects_export_as_study(files):
    path = files["dir"] / "run.json"
    path.write_text(invoke("export", files["study"], "--no-llm"))
    r = CliRunner().invoke(main, ["run", str(path), "--no-llm"])
    assert r.exit_code != 0


def test_threshold_options():
    from pathlib import Path
    ex = str(Path(__file__).resolve().parent.parent / "examples" / "thin-transcript.json")
    assert "INTERVIEWER_HEAVY" in invoke("check", ex)
    assert "INTERVIEWER_HEAVY" not in invoke("check", ex, "--interviewer-share", "0.99")
    assert "SEG_LONG" not in invoke("run", ex, "--no-llm", "--long-segment", "1000")
