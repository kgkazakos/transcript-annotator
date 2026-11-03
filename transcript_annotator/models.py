"""Data models and paradigm constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TOOL_NAME = "transcript-annotator"
EXPORT_SCHEMA = "cpr.tan.export/1"

AI_PARADIGM = "LLM (reasoning)"
AI_PARADIGM_RATIONALE = (
    "Qualitative coding fails quietly when codes arrive without their reasons: a list of "
    "labels hides whether the coder applied the codebook's criteria, read the segment in "
    "context, or matched on a keyword. Slow annotation asks a reasoning model to spend "
    "deliberate compute on each segment and to show its work per code: the exact span, the "
    "codebook criterion it meets, the exclusion criteria it considered, a counter-reading and "
    "the alternative codes it rejected. That is interpretive reasoning against a "
    "researcher-authored codebook, not classification, so a reasoning model fits. The "
    "deterministic layer checks every annotation against the transcript and the codebook, "
    "and the researcher reviews the reasoning, not just the codes."
)

SEVERITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    message: str
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class Report:
    tool: str
    version: str
    findings: list[Finding] = field(default_factory=list)
    # Enrichment is additive and separate from computed values. None when --no-llm or on failure.
    enrichment: Optional[str] = None

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counts"] = self.counts()
        return d
