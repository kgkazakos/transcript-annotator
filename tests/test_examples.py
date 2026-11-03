import re
from pathlib import Path

from transcript_annotator import engine
from transcript_annotator.engine import analyze
from transcript_annotator.io import load_input, load_reviews

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ENGINE_RULES = set(re.findall(r'Finding\(\s*"([A-Z_]+)"', Path(engine.__file__).read_text()))
EMPTY_INPUT_RULES = {"TRANSCRIPT_EMPTY", "CODEBOOK_EMPTY"}  # covered in test_engine


def fired(report):
    return {f.rule_id for f in report.findings}


def test_examples_exercise_every_engine_rule():
    seen = set()
    for path in EXAMPLES.glob("*.json"):
        seen |= fired(analyze(load_input(path)))
    reviews = load_reviews(EXAMPLES / "reviews" / "flawed-annotations.reviews.json")
    seen |= fired(analyze(load_input(EXAMPLES / "flawed-annotations.json"), reviews=reviews))
    assert ENGINE_RULES - EMPTY_INPUT_RULES - seen == set()


def test_clean_examples_have_no_findings():
    for name in ("onboarding-interview.json", "researcher-coded.json"):
        assert analyze(load_input(EXAMPLES / name)).findings == [], name


def test_researcher_coded_annotations_are_anchored():
    r = analyze(load_input(EXAMPLES / "researcher-coded.json"))
    assert [v.status for v in r.annotations] == ["anchored"] * 4
