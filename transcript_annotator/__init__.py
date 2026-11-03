"""transcript-annotator: CPR Phase II, Frame A (Test-Time Deliberation), tool 4.

Slow annotation of interview transcripts against a researcher-authored codebook. The
deterministic core (--no-llm) checks the transcript, the codebook and any annotations; the
model layer proposes annotations with the reasoning shown per code, which the core verifies.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .engine import analyze
from .models import AI_PARADIGM, AI_PARADIGM_RATIONALE, EXPORT_SCHEMA, Finding, Report

__all__ = [
    "__version__",
    "AI_PARADIGM",
    "AI_PARADIGM_RATIONALE",
    "EXPORT_SCHEMA",
    "Finding",
    "Report",
    "analyze",
]
