import copy

import pytest

from transcript_annotator.engine import (
    analyze,
    jaccard,
    normalise,
    parse_annotations,
    parse_reviews,
    syntheticity,
)
from transcript_annotator.models import Annotation, Finding, Review


def study(**overrides):
    doc = {
        "question": "Why do first-time admins stall during setup?",
        "codebook": [
            {
                "id": "C1",
                "name": "Unclear next step",
                "definition": "Participant cannot tell what action the product expects next.",
                "include": [{"id": "C1.i1", "text": "Says they did not know what to do next"}],
                "exclude": [{"id": "C1.x1", "text": "Knew the step but lacked permission"}],
            },
            {
                "id": "C2",
                "name": "Permission blocker",
                "definition": "Setup halts because the participant lacks an access right.",
                "include": [{"id": "C2.i1", "text": "Mentions needing someone else to grant access"}],
                "exclude": [{"id": "C2.x1", "text": "Access exists but the path to it is unclear"}],
            },
        ],
        "transcript": [
            {"id": "s1", "speaker": "INT", "role": "interviewer", "text": "Tell me about the first day."},
            {"id": "s2", "speaker": "P1", "role": "participant",
             "text": "I clicked through the wizard and then I honestly did not know what to do next."},
            {"id": "s3", "speaker": "P1", "role": "participant",
             "text": "Later I had to wait for IT to give me admin rights before anything worked."},
            {"id": "s4", "speaker": "P2", "role": "participant", "text": "It was fine for me, mostly."},
        ],
        "annotations": [],
    }
    doc.update(overrides)
    return doc


GOOD = {
    "segment_id": "s2",
    "code_id": "C1",
    "span": "I honestly did not know what to do next",
    "criteria_met": ["C1.i1"],
    "exclusions_considered": [{"id": "C1.x1", "reason": "No mention of access or permission."}],
    "reasoning": "The participant states directly that after the wizard they did not know the next action, "
                 "which is the include criterion in plain words.",
    "counter_reading": "They may have known but been tired; the segment does not say why they stopped.",
    "alternatives": ["C2"],
}


def ann(**kw):
    a = dict(GOOD)
    a.update(kw)
    return a


def rules(findings):
    return sorted({f.rule_id for f in findings})


def verified(doc):
    return analyze(doc).annotations


# --------------------------------------------------------------------------- helpers


def test_normalise_handles_quotes_case_and_whitespace():
    assert normalise("I  didn’t\nKnow") == "i didn't know"


def test_jaccard_ignores_stopwords():
    assert jaccard("the cat sat", "the dog ran") == 0.0
    assert jaccard("setup halts access", "setup halts access") == 1.0


def test_syntheticity_human_and_synthetic():
    assert syntheticity({"text": "x"}) == 0.0
    assert syntheticity({"_cpr_synthetic": {"tool": "t", "ancestry": []}}) == 1.0
    partial = {"_cpr_synthetic": {"tool": "t", "human_grounding": "partial", "ancestry": [{"text": "human"}]}}
    assert syntheticity(partial) == 0.5


# --------------------------------------------------------------------------- readiness


def test_clean_study_has_no_readiness_or_codebook_findings():
    r = analyze(study())
    assert r.readiness == []
    assert r.codebook == []


def test_transcript_empty():
    assert rules(analyze(study(transcript=[])).readiness) == ["TRANSCRIPT_EMPTY"]


def test_segment_id_missing_and_duplicate():
    doc = study()
    doc["transcript"].append({"speaker": "P3", "role": "participant", "text": "x"})
    doc["transcript"].append(dict(doc["transcript"][1]))
    assert {"SEG_ID_MISSING", "SEG_ID_DUPLICATE"} <= set(rules(analyze(doc).readiness))


def test_empty_segment_speaker_and_role():
    doc = study()
    doc["transcript"].append({"id": "s9", "text": ""})
    assert {"SEG_EMPTY", "SPEAKER_MISSING", "ROLE_MISSING"} <= set(rules(analyze(doc).readiness))


def test_long_segment():
    doc = study()
    doc["transcript"][3]["text"] = "word " * 260
    assert "SEG_LONG" in rules(analyze(doc).readiness)
    assert "SEG_LONG" not in rules(analyze(study()).readiness)


def test_interviewer_heavy():
    doc = study()
    doc["transcript"][0]["text"] = "So would you say " * 30
    assert "INTERVIEWER_HEAVY" in rules(analyze(doc).readiness)


def test_synthetic_share_segment_level():
    doc = study()
    for s in doc["transcript"][1:3]:
        s["_cpr_synthetic"] = {"tool": "persona-gen", "ancestry": []}
    r = analyze(doc)
    assert r.synthetic_share == 0.5
    assert "SYNTHETIC_SHARE" in rules(r.readiness)


def test_synthetic_share_transcript_level_and_under_threshold():
    doc = study()
    doc["transcript"] = {"_cpr_synthetic": {"tool": "t", "ancestry": []}, "segments": doc["transcript"]}
    assert analyze(doc).synthetic_share == 1.0
    one = study()
    one["transcript"][3]["_cpr_synthetic"] = {"tool": "t", "ancestry": []}
    assert "SYNTHETIC_SHARE" not in rules(analyze(one).readiness)  # 25% < 30%


# --------------------------------------------------------------------------- codebook


def test_codebook_empty():
    assert rules(analyze(study(codebook=[])).codebook) == ["CODEBOOK_EMPTY"]


def test_codebook_rules_fire():
    cb = [
        {"id": "A", "name": "A", "definition": "Setup halts because participant lacks access rights."},
        {"id": "A", "name": "A again", "definition": ""},
        {"id": "B", "name": "B", "definition": "Setup halts because the participant lacks access rights.",
         "include": [{"id": "B.1", "text": "x"}], "exclude": [{"id": "B.1", "text": "y"}]},
    ]
    got = set(rules(analyze(study(codebook=cb)).codebook))
    assert {"CODE_ID_DUPLICATE", "CODE_NO_DEFINITION", "CODE_NO_INCLUDE", "CODE_NO_EXCLUDE",
            "CODE_DEFINITION_OVERLAP", "CRITERION_ID_DUPLICATE"} <= got


def test_code_id_missing():
    assert "CODE_ID_MISSING" in rules(analyze(study(codebook=[{"definition": "x"}])).codebook)


def test_string_criteria_get_ids():
    cb = [{"id": "Z", "definition": "d", "include": ["a"], "exclude": ["b"]}]
    doc = study(codebook=cb, annotations=[ann(code_id="Z", criteria_met=["Z.i1"],
                                              exclusions_considered=["Z.x1"])])
    v = verified(doc)[0]
    assert "CRITERION_WRONG_CODE" not in rules(v.findings)
    assert "EXCLUSIONS_NOT_CONSIDERED" not in rules(v.findings)


# --------------------------------------------------------------------------- verifier


def test_good_annotation_is_anchored_and_clean():
    v = verified(study(annotations=[GOOD]))[0]
    assert v.status == "anchored"
    assert v.findings == []
    assert v.annotation.id == "r1" and v.annotation.source == "researcher"


def test_span_match_tolerates_case_and_quotes():
    v = verified(study(annotations=[ann(span="I HONESTLY did not  know what to do next")]))[0]
    assert v.status == "anchored"


@pytest.mark.parametrize(
    "change, rule",
    [
        ({"segment_id": "nope"}, "ANN_UNKNOWN_SEGMENT"),
        ({"code_id": "C9"}, "ANN_UNKNOWN_CODE"),
        ({"span": ""}, "SPAN_EMPTY"),
        ({"span": "I knew exactly what to do"}, "SPAN_NOT_FOUND"),
    ],
)
def test_unanchoring_rules(change, rule):
    v = verified(study(annotations=[ann(**change)]))[0]
    assert rule in rules(v.findings)
    assert v.status == "unanchored"


@pytest.mark.parametrize(
    "change, rule",
    [
        ({"segment_id": "s1", "span": "first day"}, "SPAN_INTERVIEWER"),
        ({"criteria_met": []}, "NO_CRITERION"),
        ({"criteria_met": ["C2.i1"]}, "CRITERION_WRONG_CODE"),
        ({"exclusions_considered": []}, "EXCLUSIONS_NOT_CONSIDERED"),
        ({"exclusions_considered": [{"id": "C1.x1", "reason": "r"}, {"id": "C2.x1", "reason": "r"}]},
         "EXCLUSION_WRONG_CODE"),
        ({"alternatives": ["C7"]}, "ALTERNATIVE_UNKNOWN"),
        ({"reasoning": "Fits C1."}, "REASONING_THIN"),
        ({"counter_reading": ""}, "NO_COUNTER_READING"),
    ],
)
def test_verifier_rules(change, rule):
    v = verified(study(annotations=[ann(**change)]))[0]
    assert rule in rules(v.findings)
    assert v.status == "anchored"


def test_duplicate_application():
    vs = verified(study(annotations=[GOOD, GOOD]))
    assert [rules(v.findings) for v in vs] == [[], ["DUPLICATE_APPLICATION"]]


def test_no_exclude_criteria_means_nothing_to_consider():
    doc = study()
    doc["codebook"][0]["exclude"] = []
    v = verified(study(codebook=doc["codebook"], annotations=[ann(exclusions_considered=[])]))[0]
    assert "EXCLUSIONS_NOT_CONSIDERED" not in rules(v.findings)


def test_model_and_researcher_annotations_use_same_rules():
    model = parse_annotations([ann(reasoning="short")], "m", source="model")
    r = analyze(study(annotations=[ann(reasoning="short")]), model_annotations=model)
    assert [v.annotation.source for v in r.annotations] == ["researcher", "model"]
    assert rules(r.annotations[0].findings) == rules(r.annotations[1].findings) == ["REASONING_THIN"]


def test_annotations_ordered_by_transcript_position():
    a3 = ann(segment_id="s3", code_id="C2", span="wait for IT", criteria_met=["C2.i1"],
             exclusions_considered=[{"id": "C2.x1", "reason": "r"}])
    r = analyze(study(annotations=[a3, GOOD]))
    assert [v.annotation.segment_id for v in r.annotations] == ["s2", "s3"]


# --------------------------------------------------------------------------- coverage and agreement


def test_coverage_counts_and_gaps():
    a3 = ann(segment_id="s3", code_id="C1", span="wait for IT")
    cov = analyze(study(annotations=[GOOD, a3])).coverage["researcher"]
    assert cov["code_counts"] == {"C1": 2, "C2": 0}
    assert cov["unused_codes"] == ["C2"]
    assert cov["uncoded_participant_segments"] == ["s4"]
    assert cov["single_participant_codes"] == ["C1"]


def test_coverage_absent_without_annotations():
    assert analyze(study()).coverage == {}


def test_agreement_needs_both_sources():
    assert analyze(study(annotations=[GOOD])).agreement is None


def test_agreement_table():
    model = parse_annotations([GOOD, ann(segment_id="s3", code_id="C2", span="wait for IT")], "m", source="model")
    a = analyze(study(annotations=[GOOD, ann(segment_id="s4", span="fine")]), model_annotations=model).agreement
    assert a["rows"] == [
        {"segment_id": "s2", "both": ["C1"], "model_only": [], "researcher_only": []},
        {"segment_id": "s3", "both": [], "model_only": ["C2"], "researcher_only": []},
        {"segment_id": "s4", "both": [], "model_only": [], "researcher_only": ["C1"]},
    ]
    assert a["totals"] == {"both": 1, "model_only": 1, "researcher_only": 1}


# --------------------------------------------------------------------------- review


def test_review_rules():
    bad = ann(id="r2", span="not in the text")
    doc = study(annotations=[GOOD, bad, ann(id="r3", segment_id="s3", span="wait for IT"), ann(id="r4")])
    reviews = parse_reviews([
        {"annotation_id": "r1", "decision": "accept"},
        {"annotation_id": "r2", "decision": "accept"},
        {"annotation_id": "r3", "decision": "reject"},
        {"annotation_id": "r3", "decision": "reject", "note": "Wrong code"},
        {"annotation_id": "r3", "decision": "revise"},
        {"annotation_id": "zz", "decision": "accept"},
    ])
    got = rules(analyze(doc, reviews=reviews).review)
    assert got == ["REVIEW_ACCEPTED_WITH_HIGH", "REVIEW_DUPLICATE", "REVIEW_NO_NOTE",
                   "REVIEW_UNKNOWN_ANNOTATION", "UNREVIEWED"]


def test_accept_with_note_is_the_researchers_call():
    doc = study(annotations=[ann(span="not in the text")])
    reviews = [Review("r1", "accept", note="Paraphrased from the recording; checked.")]
    assert analyze(doc, reviews=reviews).review == []


def test_no_review_findings_without_reviews():
    assert analyze(study(annotations=[GOOD])).review == []


def test_invalid_decision_rejected():
    with pytest.raises(ValueError):
        Review("r1", "approve")


# --------------------------------------------------------------------------- prompts, determinism, models


def test_prompts_for_uncoded_participant_segments():
    r = analyze(study(annotations=[GOOD]))
    assert [p["segment_id"] for p in r.prompts] == ["s3", "s4"]
    assert "include criterion" in r.prompts[0]["prompt"]


def test_engine_does_not_mutate_input_and_is_deterministic():
    doc = study(annotations=[GOOD, ann(span="nope")])
    before = copy.deepcopy(doc)
    assert analyze(doc).to_dict() == analyze(doc).to_dict()
    assert doc == before


def test_invalid_severity_and_source_rejected():
    with pytest.raises(ValueError):
        Finding(rule_id="X", severity="critical", message="m")
    with pytest.raises(ValueError):
        Annotation(id="a", segment_id="s", code_id="c", span="x", source="other")
