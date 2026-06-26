"""Load the v2026 streaming-calibration artifacts (the JSON the FleetPlane reads).

The heavy streaming ingestion (``v2026_stream`` / ``v2026_calibration``, pyarrow)
writes compact per-table calibration JSON to ``V2026_PROCESSED_DIR``. This stdlib
loader reads those artifacts and exposes them — plus their fidelity labels and
completeness — so the FleetPlane can consume FULL_TRACE_EXACT v2026 calibration
without touching pyarrow or the 351 GB source.
"""

from __future__ import annotations

import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = os.environ.get(
    "V2026_PROCESSED_DIR",
    os.path.join(_REPO, "data", "external", "alibaba_gpu_v2026", "processed"))

_TABLES = ("pod_hourly", "server_hourly", "network_hourly", "job_execution_summary")


def artifact_path(table: str, processed_dir: str = PROCESSED_DIR) -> str:
    return os.path.join(processed_dir, f"{table}_calibration.json")


def load_table(table: str, processed_dir: str = PROCESSED_DIR) -> dict | None:
    p = artifact_path(table, processed_dir)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_all(processed_dir: str = PROCESSED_DIR) -> dict:
    """Return ``{table: artifact}`` for every present calibration artifact."""
    return {t: a for t in _TABLES if (a := load_table(t, processed_dir)) is not None}


def coverage(processed_dir: str = PROCESSED_DIR) -> dict:
    """Per-table completeness + fidelity label (for the status doc / manifest)."""
    out: dict = {}
    for t in _TABLES:
        a = load_table(t, processed_dir)
        if a is None:
            out[t] = {"present": False}
        else:
            out[t] = {
                "present": True, "label": a.get("label"),
                "complete": a.get("complete"),
                "partitions": f"{a.get('n_partitions_done')}/{a.get('n_partitions_total')}",
                "bytes_streamed": a.get("bytes_streamed", 0),
                "categories": sorted((a.get("artifacts") or {}).keys()),
            }
    return out


__all__ = ["PROCESSED_DIR", "artifact_path", "load_table", "load_all", "coverage"]
