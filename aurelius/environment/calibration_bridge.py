"""CalibrationBridge — map real trace distributions into simulator parameters.

Every parameter is **distribution-derived** (computed from a real trace slice),
never a guessed constant, and carries the full provenance the build spec demands:
source dataset, table/column, fitting method, train/holdout split, trace version,
and fidelity tier. Calibration uses a time-ordered **train** split; the matching
**holdout** is handed to the ValidationSuite to prove the calibrated environment
reproduces a *held-out* distribution.

Sources & roles (never row-joined):
  * Azure   → arrival burstiness, inter-arrival CV, token distribution
  * Mooncake → KV prefix-hit rate (computed from `hash_ids` block overlap)
  * v2026   → fleet params (delegated to :class:`V2026FleetPlane.params_at`)
  * electricity → regional hourly price
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass

from .schemas import TRACE_DERIVED, CalibratedParam


def _percentile(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def time_split(raw: list, holdout_frac: float = 0.3) -> tuple:
    """Split a time-ordered ``(arrival_s, tokens)`` trace into (train, holdout)."""
    s = sorted(raw, key=lambda r: r[0])
    cut = int(len(s) * (1.0 - holdout_frac))
    return s[:cut], s[cut:]


# ---------------------------------------------------------------------------
# Azure serving-spine calibration
# ---------------------------------------------------------------------------

def azure_token_params(train: list, *, split_desc: str) -> CalibratedParam:
    toks = [t for _, t in train]
    value = {"p50": _percentile(toks, 0.50), "p95": _percentile(toks, 0.95),
             "p99": _percentile(toks, 0.99), "mean": round(statistics.mean(toks), 2) if toks else 0.0}
    return CalibratedParam(
        name="azure_token_distribution", value=value, source_dataset="azure_llm_2024",
        table_column="request.output_tokens", fitting_method="percentiles p50/p95/p99",
        train_holdout_split=split_desc, trace_version="2024", tier=TRACE_DERIVED,
        limitations="output tokens only; service time derives via TTFT+tokens·TPOT",
        safe_for_headline=True)


def azure_burstiness_params(train: list, *, split_desc: str) -> CalibratedParam:
    s = sorted(train, key=lambda r: r[0])
    gaps = [s[i + 1][0] - s[i][0] for i in range(len(s) - 1)]
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    cv = (statistics.pstdev(gaps) / mean_gap) if (gaps and mean_gap > 0) else 0.0
    value = {"interarrival_cv": round(cv, 4), "mean_gap_s": round(mean_gap, 4)}
    return CalibratedParam(
        name="azure_arrival_burstiness", value=value, source_dataset="azure_llm_2024",
        table_column="request.timestamp_s", fitting_method="inter-arrival CV",
        train_holdout_split=split_desc, trace_version="2024", tier=TRACE_DERIVED,
        limitations="single-region arrival process", safe_for_headline=True)


# ---------------------------------------------------------------------------
# Mooncake KV prefix-reuse calibration (computed from hash_ids)
# ---------------------------------------------------------------------------

def load_mooncake(path: str) -> list:
    """Return ``[(timestamp_s, [block_hash, ...]), ...]`` from a Mooncake slice."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                ts = float(r.get("timestamp_s") or 0.0)
            except ValueError:
                ts = 0.0
            blocks = (r.get("hash_ids") or "").split()
            out.append((ts, blocks))
    return out


def mooncake_prefix_hit_rate(path: str) -> CalibratedParam:
    """Prefix-hit rate = fraction of requests whose leading block was seen before.

    This is the real reuse signal Mooncake's ``hash_ids`` expose (block-level
    prefix hashes); two requests sharing a leading block shared that prefix.
    """
    reqs = sorted(load_mooncake(path), key=lambda r: r[0])
    seen: set = set()
    hits = 0
    overlaps = []
    for _, blocks in reqs:
        if blocks and blocks[0] in seen:
            hits += 1
        # block-level overlap fraction (how much of the prefix was already cached)
        if blocks:
            ov = sum(1 for b in blocks if b in seen) / len(blocks)
            overlaps.append(ov)
        seen.update(blocks)
    n = len(reqs)
    value = {
        "prefix_hit_rate": round(hits / n, 4) if n else 0.0,
        "mean_block_overlap": round(statistics.mean(overlaps), 4) if overlaps else 0.0,
        "n_requests": n,
    }
    return CalibratedParam(
        name="kv_prefix_hit_rate", value=value, source_dataset="mooncake",
        table_column="trace.hash_ids", fitting_method="leading-block overlap",
        train_holdout_split="full sample (no split — reuse signal)",
        trace_version="mooncake-sample", tier=TRACE_DERIVED,
        limitations="hit RATE is real; live eviction/memory-pressure dynamics are "
                    "simulator-only; sample is Kimi traffic, not Azure",
        safe_for_headline=True)


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBridge:
    """Holds every distribution-derived param + its provenance, for one env build."""

    params: list                      # list[CalibratedParam]
    holdout: dict                     # {"azure_tokens": [...], "azure_interarrival": [...]}

    def by_name(self, name: str) -> CalibratedParam | None:
        return next((p for p in self.params if p.name == name), None)

    def provenance(self) -> list:
        return [p.to_dict() for p in self.params]


def build_bridge(
    azure_raw: list, *, mooncake_path: str, fleet_plane, holdout_frac: float = 0.3,
) -> CalibrationBridge:
    """Calibrate from Azure (train split) + Mooncake + v2026, retaining the holdout."""
    train, hold = time_split(azure_raw, holdout_frac)
    split_desc = f"first {int((1 - holdout_frac) * 100)}% train / last {int(holdout_frac * 100)}% holdout (by time)"

    params: list = [
        azure_token_params(train, split_desc=split_desc),
        azure_burstiness_params(train, split_desc=split_desc),
        mooncake_prefix_hit_rate(mooncake_path),
    ]
    # v2026 fleet params (provenance for hour 0 as the representative set)
    if fleet_plane is not None and fleet_plane.hours():
        params.extend(fleet_plane.params_at(fleet_plane.hours()[0]))

    hold_gaps = [hold[i + 1][0] - hold[i][0] for i in range(len(hold) - 1)]
    train_gaps = [train[i + 1][0] - train[i][0] for i in range(len(train) - 1)]
    holdout = {
        "azure_tokens": [t for _, t in hold],
        "azure_interarrival": hold_gaps,
        "train_tokens": [t for _, t in train],
        "train_interarrival": train_gaps,
    }
    return CalibrationBridge(params=params, holdout=holdout)


__all__ = [
    "CalibrationBridge", "build_bridge", "time_split",
    "azure_token_params", "azure_burstiness_params",
    "mooncake_prefix_hit_rate", "load_mooncake",
]
