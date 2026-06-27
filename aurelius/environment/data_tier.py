"""Data-tier provenance — FULL_TRACE vs SUBSET vs SAMPLE_FIXTURE vs MOCK vs BLOCKED.

The canonical environment must always be honest about *which tier of data* fed a
calibration or validation. Calibration/validation default to FULL_TRACE when
available; if full ingestion is blocked we mark it explicitly (never silently
downgrade to a sample) and record the exact manual step to unblock it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Source tiers, best → worst.
FULL_TRACE = "FULL_TRACE"          # the complete available public source, ingested
VALIDATION_FIXTURE = "VALIDATION_FIXTURE"  # the complete (or proven-representative) public
#                                    trace, committed deterministically as a repo fixture so
#                                    validation is CI-reproducible without the raw download.
#                                    Real public data (never synthetic), just repo-provenance.
SUBSET_TRACE = "SUBSET_TRACE"      # a real but partial slice of the full source
SAMPLE_FIXTURE = "SAMPLE_FIXTURE"  # a small committed schema-shaped fixture
MOCK = "MOCK"                      # hand-authored, not from any trace
BLOCKED = "BLOCKED"                # full source exists but access is blocked here

TIER_ORDER = {FULL_TRACE: 0, VALIDATION_FIXTURE: 1, SUBSET_TRACE: 2,
              SAMPLE_FIXTURE: 3, MOCK: 4, BLOCKED: 5}

# Artifact-quality labels for a *computed* statistic (orthogonal to source tier):
# how exact is this aggregate relative to processing the whole source conventionally?
FULL_TRACE_EXACT = "FULL_TRACE_EXACT"    # every row processed once; mathematically exact
FULL_TRACE_APPROX = "FULL_TRACE_APPROX"  # every row processed, but the statistic is a
#                                          documented approximation (e.g. histogram percentile)
ARTIFACT_LABELS = (FULL_TRACE_EXACT, FULL_TRACE_APPROX, SUBSET_TRACE,
                   SAMPLE_FIXTURE, MOCK, BLOCKED)


@dataclass(frozen=True)
class SourceStatus:
    """Where one canonical source's data actually came from, this run."""

    source: str                    # "azure_llm", "mooncake", "alibaba_gpu_v2026", "electricity"
    tier: str                      # one of the tiers above
    path: str = ""                 # the ingested path (if any)
    n_records: int = 0
    trace_version: str = ""
    blocked_reason: str = ""       # why FULL_TRACE was unavailable (if BLOCKED)
    manual_step: str = ""          # exact next manual action to unblock

    @property
    def is_full(self) -> bool:
        return self.tier == FULL_TRACE

    @property
    def headline_safe(self) -> bool:
        return self.tier in (FULL_TRACE, VALIDATION_FIXTURE, SUBSET_TRACE)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "tier": self.tier, "path": self.path,
            "n_records": self.n_records, "trace_version": self.trace_version,
            "blocked_reason": self.blocked_reason, "manual_step": self.manual_step,
            "headline_safe": self.headline_safe,
        }


__all__ = [
    "FULL_TRACE", "VALIDATION_FIXTURE", "SUBSET_TRACE", "SAMPLE_FIXTURE", "MOCK", "BLOCKED",
    "FULL_TRACE_EXACT", "FULL_TRACE_APPROX", "ARTIFACT_LABELS",
    "TIER_ORDER", "SourceStatus",
]
