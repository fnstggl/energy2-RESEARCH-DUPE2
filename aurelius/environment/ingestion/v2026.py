"""Alibaba cluster-trace-gpu-v2026 — ingestion pipeline + honest BLOCKED state.

The FleetPlane reads the v2026 fact tables (``pod_hourly`` / ``server_hourly`` /
``network_hourly`` / ``job_execution_summary``). This module ingests them **if the
real parquet tables are present locally**, and otherwise returns an EXPLICIT
``BLOCKED`` status (never a silent downgrade) with the exact manual unblock step.

Discovered access reality (2026-06-26): the data is **not** on GitHub — it lives
on a public Aliyun OSS bucket as four ZIPs, and the core ``pod_hourly`` archive is
**351 GB** (`351,803,513,445` bytes), impractical to download/store/process in an
ephemeral CI container (and it is parquet; this repo is stdlib-only). So
FULL_TRACE v2026 is BLOCKED here; the FleetPlane runs on the committed
schema-shaped SAMPLE_FIXTURE, tagged as such.
"""

from __future__ import annotations

import csv
import glob
import os

from ..data_tier import BLOCKED, FULL_TRACE, SourceStatus

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(_REPO, "data", "external", "alibaba_gpu_v2026", "raw")
FIX_DIR = os.path.join(_REPO, "tests", "fixtures", "alibaba_gpu_v2026")

OSS = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/"
POD_HOURLY_BYTES = 351_803_513_445   # 351 GB — the binding blocker
MANUAL_STEP = (
    f"Download the v2026 ZIPs from {OSS} (pod_hourly.zip is 351 GB; server_hourly "
    "3 GB; network_hourly 204 MB; job_execution_summary 1.2 GB), unzip the parquet "
    f"partitions under {RAW_DIR}, and install a parquet reader. pod_hourly at 351 GB "
    "is impractical in an ephemeral container — run this on a host with the disk + "
    "a Spark/Arrow pipeline (see cluster-trace-gpu-v2026/scripts/build_aggregates.py).")

# Tables the FleetPlane consumes; ``required`` gates FULL_TRACE.
_TABLES = ("pod_hourly", "server_hourly", "network_hourly", "job_execution_summary")


def _have_full_tables() -> bool:
    """True only if real parquet partitions for every required table are present."""
    return all(
        glob.glob(os.path.join(RAW_DIR, f"asi_opensource_{t}", "**", "*.parquet"),
                  recursive=True)
        for t in ("pod_hourly", "server_hourly", "network_hourly"))


def v2026_status() -> SourceStatus:
    """Report the v2026 data tier honestly (FULL_TRACE / BLOCKED+sample)."""
    if _have_full_tables():
        return SourceStatus(
            source="alibaba_gpu_v2026", tier=FULL_TRACE, path=RAW_DIR,
            trace_version="v2026", n_records=0)
    # Blocked: core table is 351 GB and not present → run on the sample, but say so.
    return SourceStatus(
        source="alibaba_gpu_v2026", tier=BLOCKED, path=FIX_DIR,
        trace_version="v2026-sample", n_records=_sample_rows(),
        blocked_reason=(f"FULL_TRACE unavailable: pod_hourly.zip is "
                        f"{POD_HOURLY_BYTES:,} bytes (351 GB), parquet, off-GitHub on "
                        "Aliyun OSS — impractical in this ephemeral stdlib-only container"),
        manual_step=MANUAL_STEP)


def _sample_rows() -> int:
    p = os.path.join(FIX_DIR, "pod_hourly_sample.csv")
    if not os.path.exists(p):
        return 0
    with open(p, newline="") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


def fleet_sample_paths() -> dict:
    """Paths the FleetPlane uses under the BLOCKED state (committed sample fixtures)."""
    return {
        "pod_hourly": os.path.join(FIX_DIR, "pod_hourly_sample.csv"),
        "server_hourly": os.path.join(FIX_DIR, "server_hourly_sample.csv"),
        "network_hourly": os.path.join(FIX_DIR, "network_hourly_sample.csv"),
    }


__all__ = ["v2026_status", "fleet_sample_paths", "MANUAL_STEP", "POD_HOURLY_BYTES"]
