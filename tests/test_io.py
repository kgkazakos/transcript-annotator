import json

import pytest

from transcript_annotator.engine import analyze, parse_annotations
from transcript_annotator.io import (
    apply_reviews,
    load_input,
    load_list,
    load_reviews,
    render_agreement,
    render_text,
    report_from_export,
    to_export,
)
from transcript_annotator.models import EXPORT_SCHEMA

from test_engine import GOOD, ann, study


def test_export_is_versioned_and_has_no_synthetic_block():
    e = to_export(analyze(study(annotations=[GOOD])))
    assert e["schema"] == EXPORT_SCHEMA
    assert e["model"] is None
    assert "_cpr_synthetic" not in e
    json.dumps(e)  # serialisable


def test_export_round_trip():
    model = parse_annotations([ann(span="nope")], "m", source="model")
    r = analyze(study(annotations=[GOOD]), model_annotations=model)
    r.enrichment = {"model": "x/y", "no_code": []}
    e = json.loads(json.dumps(to_export(r)))
    back = report_from_export(e)
    assert back.to_dict() == r.to_dict()
    assert e["model"] == "x/y"


def test_apply_reviews_on_saved_run(tmp_path):
    e = json.loads(json.dumps(to_export(analyze(study(annotations=[ann(span="nope")])))))
    rv = tmp_path / "reviews.json"
    rv.write_text(json.dumps({"reviews": [{"annotation_id": "r1", "decision": "accept"}]}))
    r = apply_reviews(report_from_export(e), load_reviews(rv))
    assert [f.rule_id for f in r.review] == ["REVIEW_ACCEPTED_WITH_HIGH"]


def test_report_from_export_rejects_other_json():
    with pytest.raises(ValueError):
        report_from_export({"schema": "other/1", "report": {}})


def test_load_input_rejects_lists_and_exports(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        load_input(p)
    p.write_text(json.dumps(to_export(analyze(study()))))
    with pytest.raises(ValueError):
        load_input(p)


def test_load_list_bare_or_keyed(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([GOOD]))
    assert load_list(p, "annotations") == [GOOD]
    p.write_text(json.dumps({"annotations": [GOOD]}))
    assert load_list(p, "annotations") == [GOOD]
    p.write_text(json.dumps("x"))
    with pytest.raises(ValueError):
        load_list(p, "annotations")


def test_text_leads_with_model_annotations():
    model = parse_annotations([GOOD], "m", source="model")
    r = analyze(study(annotations=[GOOD]), model_annotations=model)
    r.enrichment = {"model": "x/y", "no_code": [{"segment_id": "s4", "reason": "Too vague to code."}]}
    t = render_text(r)
    assert t.index("Slow annotations by x/y") < t.index("Researcher annotations") < t.index("Transcript readiness")
    assert "m1" in t and "Unclear next step" in t and "no code fits" in t
    assert "Agreement (descriptive)" in t
    assert "Deliberation prompts" not in t


def test_text_no_llm_shows_prompts_and_note():
    r = analyze(study())
    r.model_note = "model unavailable"
    t = render_text(r)
    assert "Deliberation prompts" in t and "Note: model unavailable" in t


def test_render_agreement_without_both_sources():
    assert "needs both" in render_agreement(analyze(study()))
