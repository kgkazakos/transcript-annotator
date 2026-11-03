"""Data models and paradigm constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TOOL_NAME = "transcript-annotator"
EXPORT_SCHEMA = "cpr.tan.export/1"

AI_PARADIGM = "LLM (reasoning / test-time deliberation)"
AI_PARADIGM_RATIONALE = (
    "Coding against a codebook is interpretive work: deciding whether a span meets a "
    "criterion, whether an exclusion rules it out, and what the strongest other reading "
    "would be. A reasoning model spending deliberate test-time compute on each segment can "
    "write that work down per code, and the written reasoning is what the researcher "
    "reviews. The deterministic layer keeps it honest: it checks every span, criterion and "
    "exclusion against the transcript and the codebook, and drops any confidence the model "
    "claims. The researcher reviews the reasoning, not just the codes."
)

SEVERITIES = ("high", "medium", "low")
SOURCES = ("model", "researcher")
DECISIONS = ("accept", "revise", "reject")
ROLES = ("participant", "interviewer")

SYNTHETIC_KEY = "_cpr_synthetic"
SYNTHETIC_THRESHOLD = 0.30
GROUNDING_FACTORS: dict[str, float] = {"seeded": 0.75, "partial": 0.5, "none": 0.0}


@dataclass(frozen=True)
class Thresholds:
    interviewer_share: float = 0.40  # interviewer share of words above this is flagged
    long_segment_words: int = 250  # segments longer than this are hard to code finely
    definition_overlap: float = 0.60  # Jaccard overlap of two code definitions
    thin_reasoning_words: int = 15  # reasoning shorter than this is thin
    synthetic_share: float = SYNTHETIC_THRESHOLD


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass(frozen=True)
class Criterion:
    id: str
    text: str


@dataclass(frozen=True)
class Code:
    id: str
    name: str
    definition: str
    include: tuple[Criterion, ...] = ()
    exclude: tuple[Criterion, ...] = ()
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Segment:
    id: str
    speaker: str
    role: str
    text: str
    syntheticity: float = 0.0  # 0.0 human, 1.0 fully synthetic, read from _cpr_synthetic


@dataclass(frozen=True)
class ExclusionNote:
    id: str
    reason: str


@dataclass(frozen=True)
class Annotation:
    id: str
    segment_id: str
    code_id: str
    span: str
    source: str = "researcher"
    criteria_met: tuple[str, ...] = ()
    exclusions_considered: tuple[ExclusionNote, ...] = ()
    reasoning: str = ""
    counter_reading: str = ""
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")


@dataclass(frozen=True)
class Review:
    annotation_id: str
    decision: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {self.decision!r}")


@dataclass
class Study:
    question: str
    codebook: list[Code]
    transcript: list[Segment]
    annotations: list[Annotation] = field(default_factory=list)


@dataclass
class VerifiedAnnotation:
    annotation: Annotation
    status: str  # anchored | unanchored
    findings: list[Finding] = field(default_factory=list)


@dataclass
class Report:
    tool: str
    version: str
    question: str = ""
    code_names: dict[str, str] = field(default_factory=dict)
    readiness: list[Finding] = field(default_factory=list)
    codebook: list[Finding] = field(default_factory=list)
    annotations: list[VerifiedAnnotation] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    agreement: Optional[dict[str, Any]] = None
    review: list[Finding] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    synthetic_share: float = 0.0
    # Model output is additive and kept apart from computed values. None under --no-llm.
    enrichment: Optional[dict[str, Any]] = None
    model_note: Optional[str] = None

    @property
    def findings(self) -> list[Finding]:
        out = list(self.readiness) + list(self.codebook)
        for v in self.annotations:
            out += v.findings
        return out + list(self.review)

    def counts(self) -> dict[str, int]:
        fs = self.findings
        return {s: sum(1 for f in fs if f.severity == s) for s in SEVERITIES}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counts"] = self.counts()
        return d
