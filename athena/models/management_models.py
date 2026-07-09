"""Immutable data models for ATHENA's Management Quality Engine.

These dataclasses describe the textual inputs the engine reads (earnings-call
transcripts, annual reports, investor presentations), the forward-looking
commitments extracted from them, how each commitment was ultimately delivered,
and the final :class:`ManagementQualityResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DocumentType(str, Enum):
    """Source document a management commitment was extracted from."""

    EARNINGS_CALL = "Earnings Call"
    ANNUAL_REPORT = "Annual Report"
    INVESTOR_PRESENTATION = "Investor Presentation"


class CommitmentType(str, Enum):
    """Category of a detected forward-looking statement by management."""

    REVENUE_GUIDANCE = "Revenue Guidance"
    MARGIN_GUIDANCE = "Margin Guidance"
    CAPEX = "CapEx Commitment"
    DEBT_REDUCTION = "Debt Reduction Commitment"
    EXPANSION = "Expansion Plan"
    GENERAL_TARGET = "General Target"


class CommitmentStatus(str, Enum):
    """Outcome of a commitment once compared against actual results."""

    DELIVERED = "Delivered"
    DELAYED = "Delayed"
    MISSED = "Missed"
    PENDING = "Pending"
    UNVERIFIED = "Unverified"


class ManagementGrade(str, Enum):
    """Letter grade summarising overall management quality."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class ManagementDocument:
    """A single dated management communication used as engine input."""

    document_type: DocumentType
    fiscal_period: str
    text: str
    date: Optional[str] = None


@dataclass(frozen=True)
class Commitment:
    """A forward-looking promise, target or guidance detected in a document."""

    commitment_type: CommitmentType
    source_period: str
    target_period: str
    sentence: str
    promised_value: Optional[float] = None
    unit: Optional[str] = None
    document_type: Optional[DocumentType] = None


@dataclass(frozen=True)
class CommitmentOutcome:
    """A commitment paired with the actual result and a verdict."""

    commitment: Commitment
    actual_value: Optional[float]
    variance: Optional[float]
    status: CommitmentStatus


@dataclass(frozen=True)
class ManagementQualityResult:
    """Full output of the Management Quality Engine."""

    overall_score: float
    management_trust_score: float
    execution_score: float
    guidance_accuracy: float
    transparency: float
    capital_allocation: float
    communication: float
    grade: ManagementGrade
    delivered: int
    delayed: int
    missed: int
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    ai_summary: str = ""
    outcomes: list[CommitmentOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


__all__ = [
    "Commitment",
    "CommitmentOutcome",
    "CommitmentStatus",
    "CommitmentType",
    "DocumentType",
    "ManagementDocument",
    "ManagementGrade",
    "ManagementQualityResult",
]
