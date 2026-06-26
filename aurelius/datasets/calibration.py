"""Calibrate canonical-dataset parameters from REAL public traces (don't invent).

The canonical assembler (:mod:`aurelius.datasets.canonical`) needs a few
structural parameters — chiefly the **workload-class mix** (what fraction of load
is best-effort/deferrable). The first cut used an arbitrary 40% best-effort
overlay, which (we measured) inflated the compounding result. This module
replaces that guess with a ratio **derived from a real production cluster trace**.

Source: **Alibaba cluster-trace-gpu-v2023** ``openb_pod_list`` ``qos`` column —
real production QoS classes ``LS`` (latency-sensitive), ``BE`` (best-effort),
``Burstable``. This is the closest public ground truth for the *class ratio* a
real GPU fleet runs.

Honesty discipline (this is the whole point):
  * We take only the **ratio** (a distribution-level statistic), never a
    per-record join — the Alibaba trace is *training/packing* telemetry, a
    different workload from LLM serving, so its jobs cannot be merged onto the
    Azure serving spine. Ratio transfers; records do not. (This is the
    "CALIBRATE, don't join" rule from ``CANONICAL_PRODUCTION_DATASET_DESIGN.md``.)
  * We report the ratio **two ways** — by job COUNT and by GPU-WORK
    (gpu_milli·duration) — because they differ enormously (best-effort is ~20% of
    jobs but a tiny share of GPU-hours in the sample), and which one is "right"
    for a *serving* best-effort tier is genuinely unknown without pilot telemetry.
  * The committed value is from the sample fixture and tagged as such; the full
    trace's ratio should be substituted when available.

Nothing here is a production claim; it is a parameter calibration with provenance.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

# Provenance tiers (mirror signal_matrix).
TIER_MEASURED = "MEASURED_REAL"
TIER_PROXY = "PROXY"

# Alibaba qos → canonical class.
_LS = "LS"
_BE = "BE"
_BURSTABLE = "Burstable"
_BEST_EFFORT_QOS = frozenset({_BE, _BURSTABLE})


@dataclass(frozen=True)
class ClassMix:
    """A workload-class mix calibrated from a real trace, with provenance."""

    source: str
    best_effort_fraction_by_count: float
    best_effort_fraction_by_gpu_work: float
    n_jobs: int
    tier: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "best_effort_fraction_by_count": round(self.best_effort_fraction_by_count, 4),
            "best_effort_fraction_by_gpu_work": round(self.best_effort_fraction_by_gpu_work, 4),
            "n_jobs": self.n_jobs,
            "tier": self.tier,
            "note": self.note,
        }


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def alibaba_class_mix(pod_list_path: str) -> ClassMix:
    """Compute the real LS/BE/Burstable mix from an Alibaba ``openb_pod_list`` CSV.

    Returns best-effort (BE+Burstable) fraction by **job count** and by **GPU-work**
    (``gpu_milli·duration``). Distribution-level only — no per-record join.
    """
    with open(pod_list_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ClassMix(os.path.basename(pod_list_path), 0.0, 0.0, 0,
                        TIER_MEASURED, "empty pod list")

    n = len(rows)
    be_count = 0
    be_work = 0.0
    tot_work = 0.0
    for r in rows:
        qos = (r.get("qos") or "").strip()
        dur = max(0.0, _f(r.get("deletion_time")) - _f(r.get("creation_time")))
        work = _f(r.get("gpu_milli")) * dur
        tot_work += work
        if qos in _BEST_EFFORT_QOS:
            be_count += 1
            be_work += work

    return ClassMix(
        source=f"alibaba_gpu_v2023:{os.path.basename(pod_list_path)}",
        best_effort_fraction_by_count=be_count / n,
        best_effort_fraction_by_gpu_work=(be_work / tot_work) if tot_work else 0.0,
        n_jobs=n,
        tier=TIER_PROXY,  # real ratio, but from a DIFFERENT (training) workload
        note=("real production QoS ratio (LS/BE/Burstable); transfers as a "
              "distribution, not a per-record join; GPU-work share is sample-"
              "sensitive and training-not-serving — treat by-count as the anchor"),
    )


# ---------------------------------------------------------------------------
# cluster-trace-gpu-v2026 (155k-GPU ASI fleet) — the SUPERIOR calibration source
# ---------------------------------------------------------------------------
# v2026's pod_hourly table labels real SERVING workload type
# (online_inference vs offline_inference), which is the on-domain best-effort
# ratio — strictly better than the v2023 *training*-pod QoS proxy above. See
# research/results/alibaba_gpu_v2026_audit_2026-06-26.md.

V2026_JOB_TYPE = "job_type_public"
V2026_ONLINE = "online_inference"      # latency-critical serving
V2026_OFFLINE = "offline_inference"    # best-effort / batch serving (deferrable)
V2026_GPU_REQ = "gpu_request"


def alibaba_v2026_serving_class_mix(pod_hourly_path: str) -> ClassMix:
    """Best-effort serving ratio from the v2026 ``asi_opensource_pod_hourly`` table.

    Computes ``offline_inference / (online_inference + offline_inference)`` among
    **inference** pods (training/dev pods are excluded — they are not serving), by
    pod-hour COUNT and by GPU-hours (``gpu_request``·1h). This is the correct
    on-domain serving best-effort fraction — the parameter the compounding
    magnitude is bound to — grounded in real online/offline labels rather than the
    v2023 training-pod QoS proxy.

    Requires the real (multi-GB) v2026 pod_hourly CSV; no number is fabricated.
    """
    with open(pod_hourly_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    on_n = off_n = 0
    on_w = off_w = 0.0
    for r in rows:
        jt = (r.get(V2026_JOB_TYPE) or "").strip()
        w = _f(r.get(V2026_GPU_REQ))
        if jt == V2026_ONLINE:
            on_n += 1
            on_w += w
        elif jt == V2026_OFFLINE:
            off_n += 1
            off_w += w
    n_inf = on_n + off_n
    w_inf = on_w + off_w
    return ClassMix(
        source=f"alibaba_gpu_v2026:{os.path.basename(pod_hourly_path)}",
        best_effort_fraction_by_count=(off_n / n_inf) if n_inf else 0.0,
        best_effort_fraction_by_gpu_work=(off_w / w_inf) if w_inf else 0.0,
        n_jobs=n_inf,
        tier=TIER_MEASURED,  # real SERVING workload labels (on-domain, not a proxy)
        note=("offline/(online+offline) inference share from v2026 pod_hourly "
              "job_type_public — the on-domain serving best-effort ratio; "
              "hourly pod aggregates, distribution-level (not a per-record join "
              "to the token spine)"),
    )


# Default in-repo fixture path (sample — substitute the full trace when available).
_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "fixtures", "alibaba_gpu", "openb_pod_list_sample.csv",
)


def default_alibaba_class_mix() -> ClassMix:
    """The committed calibration from the Alibaba GPU sample fixture.

    Falls back to a documented literal if the fixture is absent (so the assembler
    never hard-fails on a missing optional dataset).
    """
    if os.path.exists(_FIXTURE):
        return alibaba_class_mix(_FIXTURE)
    return ClassMix(
        source="alibaba_gpu_v2023:literal_fallback",
        best_effort_fraction_by_count=0.20,
        best_effort_fraction_by_gpu_work=0.012,
        n_jobs=0, tier=TIER_PROXY,
        note="fixture absent — documented literal (~80/20 LS/BE by count)",
    )


__all__ = ["ClassMix", "alibaba_class_mix", "default_alibaba_class_mix",
           "alibaba_v2026_serving_class_mix", "TIER_MEASURED", "TIER_PROXY"]
