#!/usr/bin/env python3
"""Bounded ingest of Lightcap/agent-runtime-telemetry-small.

A small 2026-04-21 cc-by-4.0 export of MCP-style agent-runtime tool-call
execution telemetry by Faruk Alpay (Lightcap). 8 parquet configs, all
small. The two configs ingested here:

- ``operations`` — 2,262 × 33: one row per tool call with real measured
  ``duration_ms`` + ``status`` + ``stage`` + ``error_type`` +
  ``created_at_utc`` + ``updated_at_utc`` + per-call args / kwargs / result
  fingerprints. The primary tool_runtime_trace evidence.
- ``tool_summary`` — 32 × 8: per-(tool_name, status) aggregated p50 /
  median / p95 latency + first/last seen. Pre-rolled latency priors at
  the per-tool grain, useful as a routing-quality / SLA-margin prior
  without needing to recompute over operations.

Maps onto the new ``tool_runtime_trace`` canonical type
(``aurelius/traces/hf_corpus/schemas.py::ToolRuntimeRecord``).

Trust: Tier 3 (real measured execution telemetry — job-trace shape, but
the "jobs" are MCP tool calls, not GPU jobs). NOT serving telemetry —
no model_id / no input_tokens / no GPU / no queue / no replica / no
cache state. Value to Aurelius: routing-quality + failure-rate +
tail-latency priors for agent workloads.

Audit-only. Does NOT modify scheduler / controllers / robust energy
engine. Raw downloads are gitignored under
``data/external/hf/Lightcap__agent-runtime-telemetry-small/raw/``; only
schema_profile, schema_mapping, summary, statistical_rollups, the
tiny fixture, and the bounded normalized sample (cc-by-4.0 permits
redistribution) are committed.

Layout::

    data/external/hf/Lightcap__agent-runtime-telemetry-small/raw/<file>   # gitignored
    data/external/hf/Lightcap__agent-runtime-telemetry-small/<config>/processed/
        schema_profile.json
        schema_mapping.json
        summary.json
        statistical_rollups.json
        normalized_sample.jsonl                                           # committed
        analysis_sample.jsonl                                             # gitignored
    tests/fixtures/hf/Lightcap__agent-runtime-telemetry-small__<config>_sample.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_DATASET = DATASET_ID.replace("/", "__")

MAX_COMMITTED_FIXTURE_BYTES = 16 * 1024
MAX_COMMITTED_NORMALIZED_BYTES = 100 * 1024 * 1024  # 100 MiB per the policy

PER_DATASET_TIMEOUT_S = 30 * 60
PROGRESS_INTERVAL_S = 30

logger = logging.getLogger("aurelius.hf_lightcap_runtime_ingest")


# ---------------------------------------------------------------------------
# Heartbeat / timeout
# ---------------------------------------------------------------------------


class _Heartbeat:
    def __init__(self, label: str):
        self.label = label
        self.start = time.monotonic()
        self.last_log = self.start
        self.phase = "init"
        self.bytes_done = 0
        self.rows_done = 0

    def update(self, *, phase: str | None = None, bytes_done: int | None = None,
               rows_done: int | None = None, force: bool = False) -> None:
        if phase is not None:
            self.phase = phase
        if bytes_done is not None:
            self.bytes_done = bytes_done
        if rows_done is not None:
            self.rows_done = rows_done
        now = time.monotonic()
        if force or (now - self.last_log) >= PROGRESS_INTERVAL_S:
            elapsed = int(now - self.start)
            logger.info("  [hb] %s phase=%s bytes=%d rows=%d elapsed=%ds",
                        self.label, self.phase, self.bytes_done,
                        self.rows_done, elapsed)
            self.last_log = now


class _PerDatasetTimeout(Exception):  # noqa: N818
    pass


def _install_timeout(seconds: int) -> None:
    def _h(_signo, _frame):
        raise _PerDatasetTimeout(f"per-dataset timeout after {seconds}s")
    signal.signal(signal.SIGALRM, _h)
    signal.alarm(seconds)


def _clear_timeout() -> None:
    signal.alarm(0)


# ---------------------------------------------------------------------------
# Per-config target table
# ---------------------------------------------------------------------------


TARGETS: list[dict] = [
    {
        "config_name": "operations",
        "raw_file": "data/operations.parquet",
        "expected_raw_bytes": 280_000,
        "primary": True,
    },
    {
        "config_name": "tool_summary",
        "raw_file": "data/tool_summary.parquet",
        "expected_raw_bytes": 9_000,
        "primary": False,
    },
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _hf_url(dataset_id: str, raw_file: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{urllib.parse.quote(dataset_id, safe='/')}/resolve/main/"
        f"{urllib.parse.quote(raw_file)}"
    )


def _bounded_download(url: str, dest: Path, *, max_bytes: int | None,
                      token: str | None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aurelius-hf-lightcap-runtime-ingest/1.0"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{int(max_bytes - 1)}"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    written = 0
    status = None
    truncated = False
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            status = resp.getcode()
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    if max_bytes is not None:
                        remaining = max_bytes - written
                        if remaining <= 0:
                            truncated = True
                            break
                        if len(chunk) > remaining:
                            out.write(chunk[:remaining])
                            written += remaining
                            truncated = True
                            break
                    out.write(chunk)
                    written += len(chunk)
    except urllib.error.HTTPError as e:
        return {"url": url, "dest": str(dest), "status": int(e.code),
                "downloaded_bytes": 0, "truncated": False,
                "error": f"HTTPError {e.code}: {e.reason}",
                "max_bytes": max_bytes}
    except urllib.error.URLError as e:
        return {"url": url, "dest": str(dest), "status": None,
                "downloaded_bytes": 0, "truncated": False,
                "error": f"URLError: {e.reason}",
                "max_bytes": max_bytes}
    return {"url": url, "dest": str(dest), "status": status,
            "downloaded_bytes": written, "truncated": truncated,
            "max_bytes": max_bytes, "error": None}


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _iso_to_epoch_s(iso: str | None) -> float | None:
    if not iso or not isinstance(iso, str):
        return None
    try:
        from datetime import datetime
        s = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _safe_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _write_jsonl(rows: list[dict], path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    for r in rows:
        line = json.dumps(r, sort_keys=True, separators=(",", ":"),
                          default=str) + "\n"
        buf.write(line.encode("utf-8"))
    data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return len(data), hashlib.sha256(data).hexdigest()


def _statistical_sample_strength(rows: int) -> str:
    if rows >= 10_000:
        return "strong"
    if rows >= 1_000:
        return "moderate"
    if rows >= 100:
        return "weak"
    return "fixture_only"


# ---------------------------------------------------------------------------
# Normalisation: operations row → ToolRuntimeRecord-shaped dict
# ---------------------------------------------------------------------------


def _normalize_operations_row(raw: dict) -> dict:
    """Map one raw operations.parquet row → flat ToolRuntimeRecord-shaped row."""
    status = (raw.get("status") or "") or None
    stage = (raw.get("stage") or "") or None
    error_type = (raw.get("error_type") or "") or None
    created_iso = raw.get("created_at_utc")
    updated_iso = raw.get("updated_at_utc")
    created_s = _iso_to_epoch_s(created_iso)
    updated_s = _iso_to_epoch_s(updated_iso)
    duration_ms = raw.get("duration_ms")
    duration_s = None
    if isinstance(duration_ms, (int, float)):
        duration_s = float(duration_ms) / 1000.0
    is_error = (status == "error") if status is not None else None
    is_cancelled = (status == "cancelled") if status is not None else None
    return {
        # Identity + routing
        "operation_id": raw.get("operation_id"),
        "request_id": raw.get("request_id"),
        "tool_name": raw.get("tool_name"),
        "stage": stage,
        "status": status,
        "operation_mode": (raw.get("operation_mode") or "") or None,
        "backend_preference": (raw.get("backend_preference") or "") or None,
        # Timestamps
        "created_at_iso": created_iso,
        "updated_at_iso": updated_iso,
        "created_at_s": created_s,
        "updated_at_s": updated_s,
        "duration_ms": duration_ms,
        "duration_s": duration_s,
        # Failure / cancel labels
        "error_type": error_type,
        "error_message_preview": (raw.get("error_message_preview") or "") or None,
        "error_message_sha256": (raw.get("error_message_sha256") or "") or None,
        "is_error": is_error,
        "is_cancelled": is_cancelled,
        # Payload-size proxies
        "args_fingerprint": (raw.get("args_fingerprint") or "") or None,
        "args_count": raw.get("args_count"),
        "args_keys": (raw.get("args_keys") or "") or None,
        "kwargs_key_count": raw.get("kwargs_key_count"),
        "kwargs_keys": (raw.get("kwargs_keys") or "") or None,
        "result_summary_key_count": raw.get("result_summary_key_count"),
        "result_summary_keys": (raw.get("result_summary_keys") or "") or None,
        "result_type": (raw.get("result_type") or "") or None,
        "result_operation": (raw.get("result_operation") or "") or None,
        "result_payload_key_count": raw.get("result_payload_key_count"),
        "result_payload_keys": (raw.get("result_payload_keys") or "") or None,
        "result_payload_bytes": raw.get("result_payload_bytes"),
        "artifacts_bytes": raw.get("artifacts_bytes"),
        # Provenance flags
        "force_retrain": raw.get("force_retrain"),
        "include_control_sensitivities": raw.get("include_control_sensitivities"),
        "include_validation_protocols": raw.get("include_validation_protocols"),
        "has_input_provenance": raw.get("has_input_provenance"),
        "has_source_binding": raw.get("has_source_binding"),
        "series_rows_count": raw.get("series_rows_count"),
        "scenario_rows_count": raw.get("scenario_rows_count"),
    }


def _normalize_tool_summary_row(raw: dict) -> dict:
    """Map one raw tool_summary.parquet row → flat row.

    The aggregated table only has (tool_name, status, count, avg/median/p95
    durations, first/last seen). To fit the same ToolRuntimeRecord schema
    every committed row carries the same field set; aggregate-only rows
    leave per-call fields null and embed the rollup statistics as
    derived-quality duration fields ``duration_ms`` (set to median for
    schema compatibility) plus the explicit ``p95_duration_ms`` /
    ``avg_duration_ms`` reported in statistical_rollups (NOT in the
    normalized sample — those live in statistical_rollups.json).
    """
    created_iso = raw.get("first_seen_utc")
    updated_iso = raw.get("last_seen_utc")
    created_s = _iso_to_epoch_s(created_iso)
    updated_s = _iso_to_epoch_s(updated_iso)
    # We store the aggregate median as duration_ms (field_quality=derived)
    # so the row is still a valid ToolRuntimeRecord shape. The avg + p95
    # live in statistical_rollups.json, not in normalized sample rows.
    median = raw.get("median_duration_ms")
    return {
        "operation_id": None,  # aggregate row, no per-call op id
        "request_id": None,
        "tool_name": raw.get("tool_name"),
        "stage": None,
        "status": raw.get("status"),
        "operation_mode": None,
        "backend_preference": None,
        "created_at_iso": created_iso,
        "updated_at_iso": updated_iso,
        "created_at_s": created_s,
        "updated_at_s": updated_s,
        "duration_ms": median,
        "duration_s": (float(median) / 1000.0) if isinstance(median, (int, float))
                        else None,
        "error_type": None,
        "error_message_preview": None,
        "error_message_sha256": None,
        "is_error": (raw.get("status") == "error") if raw.get("status") else None,
        "is_cancelled": (raw.get("status") == "cancelled") if raw.get("status")
                        else None,
        "args_fingerprint": None,
        "args_count": None,
        "args_keys": None,
        "kwargs_key_count": None,
        "kwargs_keys": None,
        "result_summary_key_count": None,
        "result_summary_keys": None,
        "result_type": None,
        "result_operation": None,
        "result_payload_key_count": None,
        "result_payload_keys": None,
        "result_payload_bytes": None,
        "artifacts_bytes": None,
        "force_retrain": None,
        "include_control_sensitivities": None,
        "include_validation_protocols": None,
        "has_input_provenance": None,
        "has_source_binding": None,
        "series_rows_count": None,
        "scenario_rows_count": None,
    }


# ---------------------------------------------------------------------------
# Schema mapping table (raw column → normalized field + quality)
# ---------------------------------------------------------------------------


OPERATIONS_MAPPING: dict = {
    "operation_id": {
        "normalized_field": "operation_id", "field_quality": "real",
        "aurelius_signal_category": "session_id",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Tool-call identifier (e.g. 'deploy-smoke-1776349977').",
    },
    "request_id": {
        "normalized_field": "request_id", "field_quality": "real",
        "aurelius_signal_category": "session_id",
        "usable_for": ["routing_quality"],
        "notes": "MCP request UUID. Joins to audit_records.request_id.",
    },
    "tool_name": {
        "normalized_field": "tool_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality", "workload_shape_only"],
        "notes": "Tool routing key (e.g. surface_affinity, workflow_run). "
                 "22 distinct tools in this export.",
    },
    "status": {
        "normalized_field": "status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Terminal status: ok / error / cancelled. ~5.5% error rate "
                 "in this export.",
    },
    "stage": {
        "normalized_field": "stage", "field_quality": "real",
        "aurelius_signal_category": "scheduler_state",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Lifecycle stage at terminal time: completed / failed / "
                 "affinity_rejected / startup_reconciled.",
    },
    "duration_ms": {
        "normalized_field": "duration_ms", "field_quality": "real",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "Real measured end-to-end tool-call duration in ms. Closed "
                 "tool-runtime timing — NOT GPU TTFT/TPOT.",
    },
    "created_at_utc": {
        "normalized_field": "created_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "RFC3339 ISO arrival timestamp; epoch seconds derived as "
                 "created_at_s.",
    },
    "updated_at_utc": {
        "normalized_field": "updated_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_completion",
        "usable_for": ["latency_prior", "constraint_aware_backtest"],
        "notes": "RFC3339 ISO completion timestamp; epoch seconds derived as "
                 "updated_at_s.",
    },
    "args_fingerprint": {
        "normalized_field": "args_fingerprint", "field_quality": "real",
        "aurelius_signal_category": "cache_residency",
        "usable_for": ["cache_residency_evaluation"],
        "notes": "sha256 of args; same fingerprint = same args = potential "
                 "cache-reuse signal.",
    },
    "args_count": {
        "normalized_field": "args_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Positional argument count (payload-shape proxy).",
    },
    "args_keys": {
        "normalized_field": "args_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited arg key list (payload-shape proxy).",
    },
    "kwargs_key_count": {
        "normalized_field": "kwargs_key_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Keyword-argument key count (payload-shape proxy).",
    },
    "kwargs_keys": {
        "normalized_field": "kwargs_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited kwarg key list (payload-shape proxy).",
    },
    "operation_mode": {
        "normalized_field": "operation_mode", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality"],
        "notes": "Per-tool operation mode (where the tool supports modes).",
    },
    "backend_preference": {
        "normalized_field": "backend_preference", "field_quality": "real",
        "aurelius_signal_category": "routing",
        "usable_for": ["routing_quality"],
        "notes": "Routing preference recorded by the agent runtime (where set).",
    },
    "force_retrain": {
        "normalized_field": "force_retrain", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Tool-specific capability flag (granite_timeseries family).",
    },
    "include_control_sensitivities": {
        "normalized_field": "include_control_sensitivities",
        "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Tool-specific capability flag.",
    },
    "include_validation_protocols": {
        "normalized_field": "include_validation_protocols",
        "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Tool-specific capability flag.",
    },
    "has_input_provenance": {
        "normalized_field": "has_input_provenance", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Whether the operation declared input provenance.",
    },
    "has_source_binding": {
        "normalized_field": "has_source_binding", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Whether the operation declared a source binding.",
    },
    "series_rows_count": {
        "normalized_field": "series_rows_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-call input series-row count (payload-shape proxy).",
    },
    "scenario_rows_count": {
        "normalized_field": "scenario_rows_count", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Per-call input scenario-row count (payload-shape proxy).",
    },
    "result_summary_key_count": {
        "normalized_field": "result_summary_key_count",
        "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Output result summary-key count (output-shape proxy).",
    },
    "result_summary_keys": {
        "normalized_field": "result_summary_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited result summary key list.",
    },
    "result_type": {
        "normalized_field": "result_type", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Result top-level type (object / list / ...).",
    },
    "result_operation": {
        "normalized_field": "result_operation", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Result-recorded operation label (provider-specific).",
    },
    "result_payload_key_count": {
        "normalized_field": "result_payload_key_count",
        "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Output result payload-key count.",
    },
    "result_payload_keys": {
        "normalized_field": "result_payload_keys", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Pipe-delimited result payload key list.",
    },
    "result_payload_bytes": {
        "normalized_field": "result_payload_bytes", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only", "latency_prior"],
        "notes": "Real measured result payload byte count (output-size proxy).",
    },
    "artifacts_bytes": {
        "normalized_field": "artifacts_bytes", "field_quality": "real",
        "aurelius_signal_category": "tokens",
        "usable_for": ["workload_shape_only"],
        "notes": "Real measured side-effect artifact byte count.",
    },
    "error_type": {
        "normalized_field": "error_type", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality", "constraint_aware_backtest"],
        "notes": "Python exception class for status=error. 8 distinct error "
                 "types in this export (TimeoutError / RuntimeError / "
                 "ValueError / SurfaceAffinityError / ...).",
    },
    "error_message_preview": {
        "normalized_field": "error_message_preview", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Bounded error-message preview (string, redacted upstream).",
    },
    "error_message_sha256": {
        "normalized_field": "error_message_sha256", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Full-message hash for grouping; raw message not retained.",
    },
}


TOOL_SUMMARY_MAPPING: dict = {
    "tool_name": {
        "normalized_field": "tool_name", "field_quality": "real",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["routing_quality"],
        "notes": "Tool routing key.",
    },
    "status": {
        "normalized_field": "status", "field_quality": "real",
        "aurelius_signal_category": "failure_timeout",
        "usable_for": ["routing_quality"],
        "notes": "Bucket status (ok / error / cancelled).",
    },
    "operation_count": {
        "normalized_field": "result_summary_key_count",
        "field_quality": "derived",
        "aurelius_signal_category": "metadata_only",
        "usable_for": ["workload_shape_only"],
        "notes": "Number of operations rolled into this (tool, status) "
                 "bucket. Stored in statistical_rollups, not normalized "
                 "sample.",
    },
    "avg_duration_ms": {
        "normalized_field": "duration_ms",
        "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "Mean tool-call duration in ms for this (tool, status) "
                 "bucket — derived from operations rows.",
    },
    "median_duration_ms": {
        "normalized_field": "duration_ms",
        "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "Median tool-call duration in ms — committed as the "
                 "row-level duration_ms in tool_summary normalized sample.",
    },
    "p95_duration_ms": {
        "normalized_field": "duration_ms",
        "field_quality": "derived",
        "aurelius_signal_category": "latency",
        "usable_for": ["latency_prior"],
        "notes": "p95 tool-call duration in ms — recorded in "
                 "statistical_rollups.json for direct routing-quality use.",
    },
    "first_seen_utc": {
        "normalized_field": "created_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_arrival",
        "usable_for": ["workload_shape_only"],
        "notes": "First operation timestamp per (tool, status) bucket.",
    },
    "last_seen_utc": {
        "normalized_field": "updated_at_iso", "field_quality": "real",
        "aurelius_signal_category": "request_completion",
        "usable_for": ["workload_shape_only"],
        "notes": "Last operation timestamp per (tool, status) bucket.",
    },
}


CONFIG_MAPPINGS = {
    "operations": OPERATIONS_MAPPING,
    "tool_summary": TOOL_SUMMARY_MAPPING,
}

CONFIG_NORMALIZERS = {
    "operations": _normalize_operations_row,
    "tool_summary": _normalize_tool_summary_row,
}


# ---------------------------------------------------------------------------
# Schema profile (raw + normalized)
# ---------------------------------------------------------------------------


def _classify_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _profile_rows(raw_rows: list[dict], normalized: list[dict],
                  config: str, raw_file: str, file_size_bytes: int) -> dict:
    raw_cols: dict[str, dict] = {}
    n_raw = len(raw_rows)
    for r in raw_rows:
        for k, v in r.items():
            c = raw_cols.setdefault(k, {"present": 0, "types": set(),
                                        "examples": []})
            c["present"] += 1
            c["types"].add(_classify_value(v))
            if len(c["examples"]) < 3:
                if isinstance(v, (dict, list)):
                    try:
                        c["examples"].append(json.dumps(v, default=str)[:120])
                    except Exception:
                        c["examples"].append(repr(v)[:120])
                else:
                    c["examples"].append(repr(v)[:120])
    norm_cols: dict[str, dict] = {}
    for r in normalized:
        for k, v in r.items():
            c = norm_cols.setdefault(k, {"present": 0, "types": set(),
                                         "examples": []})
            # Treat None / empty string as "not present" for missing-rate computation.
            present = not (v is None or (isinstance(v, str) and v == ""))
            if present:
                c["present"] += 1
            c["types"].add(_classify_value(v))
            if len(c["examples"]) < 3:
                c["examples"].append(repr(v)[:120])
    n_norm = len(normalized)
    profile = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_files_inspected": [raw_file],
        "file_size_bytes": file_size_bytes,
        "inspected_row_count": n_norm,
        "raw_row_count": n_raw,
        "raw_columns": sorted(raw_cols.keys()),
        "raw_dtypes": {k: sorted(v["types"]) for k, v in raw_cols.items()},
        "raw_presence_rates": {k: (v["present"] / n_raw) if n_raw else 0
                                for k, v in raw_cols.items()},
        "raw_example_values": {k: v["examples"] for k, v in raw_cols.items()},
        "normalized_columns": sorted(norm_cols.keys()),
        "normalized_dtypes": {k: sorted(v["types"])
                               for k, v in norm_cols.items()},
        "presence_rates": {k: (v["present"] / n_norm) if n_norm else 0
                           for k, v in norm_cols.items()},
        "missing_rates": {k: 1 - ((v["present"] / n_norm) if n_norm else 0)
                          for k, v in norm_cols.items()},
        "example_values": {k: v["examples"] for k, v in norm_cols.items()},
    }
    return profile


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


ALL_SIGNALS = (
    "request_timestamps", "arrivals",
    "latency", "duration_measured",
    "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
    "tool_routing", "tool_failure_label", "tool_cancellation_label",
    "args_fingerprint_for_cache_reuse",
    "workload_shape", "customer_traffic_mix",
    "result_size_proxy", "artifacts_size_proxy",
    "model_load_event", "model_unload_event",
    "cost_or_region",
    "kv_block_hashes", "migration_or_cache_loss_proxy",
)


def _detect_signals_operations(profile: dict, normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["normalized_columns"])
    if {"created_at_iso", "updated_at_iso", "created_at_s"} & cols:
        out["request_timestamps"] = True
        out["arrivals"] = True
    if "duration_ms" in cols:
        out["latency"] = True
        out["duration_measured"] = True
    if "tool_name" in cols:
        out["tool_routing"] = True
        out["customer_traffic_mix"] = True
    if "is_error" in cols and any(r.get("is_error") for r in normalized):
        out["tool_failure_label"] = True
    if "is_cancelled" in cols and any(r.get("is_cancelled") for r in normalized):
        out["tool_cancellation_label"] = True
    if "args_fingerprint" in cols:
        out["args_fingerprint_for_cache_reuse"] = True
    if any(c in cols for c in ("args_count", "kwargs_key_count",
                                "series_rows_count", "scenario_rows_count",
                                "result_payload_bytes")):
        out["workload_shape"] = True
    if "result_payload_bytes" in cols:
        out["result_size_proxy"] = True
    if "artifacts_bytes" in cols:
        out["artifacts_size_proxy"] = True
    return out


def _detect_signals_tool_summary(profile: dict, normalized: list[dict]) -> dict:
    out = {s: False for s in ALL_SIGNALS}
    cols = set(profile["normalized_columns"])
    if {"created_at_iso", "updated_at_iso"} & cols:
        out["request_timestamps"] = True
    if "duration_ms" in cols:
        # Derived aggregate latency — recorded but tier 4-style prior only.
        out["latency"] = True
    if "tool_name" in cols:
        out["tool_routing"] = True
    if "is_error" in cols and any(r.get("is_error") for r in normalized):
        out["tool_failure_label"] = True
    return out


CONFIG_SIGNAL_DETECTORS = {
    "operations": _detect_signals_operations,
    "tool_summary": _detect_signals_tool_summary,
}


# ---------------------------------------------------------------------------
# Statistical rollups
# ---------------------------------------------------------------------------


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(round(q * (n - 1)))))
    return sorted_vals[idx]


def _summarize_durations(rows: list[dict],
                          dur_field: str = "duration_ms") -> dict:
    vals = [r[dur_field] for r in rows
            if isinstance(r.get(dur_field), (int, float))]
    if not vals:
        return {"count": 0}
    vals.sort()
    n = len(vals)
    return {
        "count": n,
        "min": vals[0],
        "p50": _quantile(vals, 0.50),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": vals[-1],
        "mean": sum(vals) / n,
    }


def _compute_rollups_operations(normalized: list[dict], raw_rows: list[dict]
                                 ) -> dict:
    rollups: dict = {"subgroup_counts": {}, "numeric_distributions": {}}

    # Overall + per-tool + per-(tool, status) subgroup counts.
    overall_status: dict[str, int] = {}
    for r in normalized:
        s = r.get("status") or "<unset>"
        overall_status[s] = overall_status.get(s, 0) + 1
    rollups["subgroup_counts"]["status"] = overall_status

    tool_counts: dict[str, int] = {}
    for r in normalized:
        t = r.get("tool_name") or "<unset>"
        tool_counts[t] = tool_counts.get(t, 0) + 1
    rollups["subgroup_counts"]["tool_name"] = tool_counts

    stage_counts: dict[str, int] = {}
    for r in normalized:
        s = r.get("stage") or "<unset>"
        stage_counts[s] = stage_counts.get(s, 0) + 1
    rollups["subgroup_counts"]["stage"] = stage_counts

    error_counts: dict[str, int] = {}
    for r in normalized:
        e = r.get("error_type") or ""
        if e:
            error_counts[e] = error_counts.get(e, 0) + 1
    rollups["subgroup_counts"]["error_type"] = error_counts

    tool_status_counts: dict[str, int] = {}
    for r in normalized:
        key = f"{r.get('tool_name') or '<unset>'}|{r.get('status') or '<unset>'}"
        tool_status_counts[key] = tool_status_counts.get(key, 0) + 1
    rollups["subgroup_counts"]["tool_name__status"] = tool_status_counts

    # Numeric distributions (overall + per-tool + per-status).
    rollups["numeric_distributions"]["duration_ms"] = {
        "overall": _summarize_durations(normalized, "duration_ms"),
        "per_status": {
            s: _summarize_durations([r for r in normalized
                                     if (r.get("status") or "<unset>") == s])
            for s in sorted(overall_status.keys())
        },
        "per_tool": {
            t: _summarize_durations([r for r in normalized
                                     if (r.get("tool_name") or "<unset>") == t])
            for t in sorted(tool_counts.keys())
        },
    }

    for nf in ("result_payload_bytes", "artifacts_bytes",
                "args_count", "kwargs_key_count",
                "result_payload_key_count"):
        rollups["numeric_distributions"][nf] = _summarize_durations(
            normalized, nf,
        )

    # Failure rate + cancellation rate (per-tool routing-quality priors).
    per_tool_failure = {}
    for t, n in tool_counts.items():
        errors = sum(1 for r in normalized
                     if (r.get("tool_name") or "<unset>") == t
                     and r.get("is_error"))
        cancels = sum(1 for r in normalized
                      if (r.get("tool_name") or "<unset>") == t
                      and r.get("is_cancelled"))
        per_tool_failure[t] = {
            "count": n,
            "error_count": errors,
            "cancelled_count": cancels,
            "error_rate": errors / n if n else 0.0,
            "cancelled_rate": cancels / n if n else 0.0,
        }
    rollups["per_tool_failure_rates"] = per_tool_failure

    # Overall failure / cancellation rate.
    n_total = len(normalized)
    n_err = sum(1 for r in normalized if r.get("is_error"))
    n_cancel = sum(1 for r in normalized if r.get("is_cancelled"))
    rollups["overall_failure_rates"] = {
        "count": n_total,
        "error_count": n_err,
        "cancelled_count": n_cancel,
        "error_rate": n_err / n_total if n_total else 0.0,
        "cancelled_rate": n_cancel / n_total if n_total else 0.0,
    }

    # Raw column count for cross-check.
    rollups["raw_row_count"] = len(raw_rows)
    return rollups


def _compute_rollups_tool_summary(normalized: list[dict],
                                   raw_rows: list[dict]) -> dict:
    """tool_summary rows are already aggregated. Record the per-(tool, status)
    avg/median/p95 directly so the downstream consumer doesn't have to read
    the raw parquet."""
    rollups: dict = {"subgroup_counts": {}, "per_tool_status_aggregates": {}}
    tool_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for r in raw_rows:
        t = r.get("tool_name") or "<unset>"
        s = r.get("status") or "<unset>"
        tool_counts[t] = tool_counts.get(t, 0) + 1
        status_counts[s] = status_counts.get(s, 0) + 1
        key = f"{t}|{s}"
        rollups["per_tool_status_aggregates"][key] = {
            "operation_count": r.get("operation_count"),
            "avg_duration_ms": r.get("avg_duration_ms"),
            "median_duration_ms": r.get("median_duration_ms"),
            "p95_duration_ms": r.get("p95_duration_ms"),
            "first_seen_utc": r.get("first_seen_utc"),
            "last_seen_utc": r.get("last_seen_utc"),
        }
    rollups["subgroup_counts"]["tool_name"] = tool_counts
    rollups["subgroup_counts"]["status"] = status_counts
    rollups["raw_row_count"] = len(raw_rows)
    return rollups


CONFIG_ROLLUPS = {
    "operations": _compute_rollups_operations,
    "tool_summary": _compute_rollups_tool_summary,
}


# ---------------------------------------------------------------------------
# Audit driver
# ---------------------------------------------------------------------------


def audit_one(target: dict, *, token: str | None,
              force_redownload: bool) -> dict:
    config = target["config_name"]
    hb = _Heartbeat(f"{DATASET_ID}@{config}")
    hb.update(phase="setup", force=True)

    safe_ds = HF_DIR / SAFE_DATASET
    raw_path = safe_ds / "raw" / Path(target["raw_file"]).name
    processed_dir = safe_ds / config / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    schema_profile_path = processed_dir / "schema_profile.json"
    schema_mapping_path = processed_dir / "schema_mapping.json"
    summary_path = processed_dir / "summary.json"
    rollups_path = processed_dir / "statistical_rollups.json"
    analysis_sample_path = processed_dir / "analysis_sample.jsonl"
    normalized_sample_path = processed_dir / "normalized_sample.jsonl"
    fixture_path = FIXTURES_DIR / f"{SAFE_DATASET}__{config}_sample.jsonl"

    url = _hf_url(DATASET_ID, target["raw_file"])

    hb.update(phase="download", force=True)
    if force_redownload or not raw_path.exists():
        manifest = _bounded_download(url, raw_path, max_bytes=None,
                                     token=token)
        manifest["cached"] = False
    else:
        manifest = {"url": url, "dest": str(raw_path),
                    "downloaded_bytes": raw_path.stat().st_size,
                    "status": None, "truncated": None,
                    "error": None, "max_bytes": None, "cached": True}
    if manifest.get("error"):
        return {"target": target, "manifest": manifest,
                "audit_status": "download_failed"}

    hb.update(phase="parse", force=True)
    raw_rows = _read_parquet(raw_path)
    hb.update(phase="parsed_rows", rows_done=len(raw_rows), force=True)
    if not raw_rows:
        return {"target": target, "manifest": manifest,
                "audit_status": "no_rows"}

    hb.update(phase="normalize", force=True)
    normalizer = CONFIG_NORMALIZERS[config]
    normalized = [normalizer(r) for r in raw_rows]
    hb.update(phase="normalized", rows_done=len(normalized), force=True)

    hb.update(phase="profile", force=True)
    profile = _profile_rows(raw_rows, normalized, config, target["raw_file"],
                             manifest["downloaded_bytes"])
    with open(schema_profile_path, "w") as fh:
        json.dump(profile, fh, indent=2, default=str, sort_keys=True)

    # Schema mapping: enumerate every raw column and label it.
    mapping_dict = CONFIG_MAPPINGS[config]
    accepted = [c for c in profile["raw_columns"] if c in mapping_dict]
    rejected = [c for c in profile["raw_columns"] if c not in mapping_dict]
    column_records = []
    for c in profile["raw_columns"]:
        m = mapping_dict.get(c, {})
        column_records.append({
            "raw_column_name": c,
            "raw_dtype": profile["raw_dtypes"].get(c),
            "presence_rate": profile["raw_presence_rates"].get(c),
            "normalized_field": m.get("normalized_field"),
            "field_quality": m.get("field_quality"),
            "aurelius_signal_category": m.get("aurelius_signal_category"),
            "usable_for": m.get("usable_for"),
            "notes": m.get("notes"),
        })
    mapping_doc = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "accepted_columns": sorted(accepted),
        "rejected_columns": sorted(rejected),
        "columns": column_records,
    }
    with open(schema_mapping_path, "w") as fh:
        json.dump(mapping_doc, fh, indent=2, default=str, sort_keys=True)

    hb.update(phase="rollups", force=True)
    rollups = CONFIG_ROLLUPS[config](normalized, raw_rows)
    with open(rollups_path, "w") as fh:
        json.dump(rollups, fh, indent=2, default=str, sort_keys=True)

    hb.update(phase="write_analysis_sample", force=True)
    analysis_bytes, analysis_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], analysis_sample_path,
    )

    hb.update(phase="write_normalized_sample", force=True)
    normalized_bytes, normalized_sha = _write_jsonl(
        [_safe_jsonable(r) for r in normalized], normalized_sample_path,
    )
    if normalized_bytes > MAX_COMMITTED_NORMALIZED_BYTES:
        keep = max(1, int(len(normalized) *
                          MAX_COMMITTED_NORMALIZED_BYTES / normalized_bytes))
        normalized_bytes, normalized_sha = _write_jsonl(
            [_safe_jsonable(r) for r in normalized[:keep]],
            normalized_sample_path,
        )

    hb.update(phase="write_fixture", force=True)
    fixture_rows = [_safe_jsonable(r) for r in normalized[:5]]
    fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
    while fixture_bytes > MAX_COMMITTED_FIXTURE_BYTES and fixture_rows:
        fixture_rows = fixture_rows[: max(1, len(fixture_rows) - 1)]
        fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
        if len(fixture_rows) == 1 and fixture_bytes > MAX_COMMITTED_FIXTURE_BYTES:
            slim = {k: v for k, v in fixture_rows[0].items()
                    if not isinstance(v, str) or len(v) < 200}
            fixture_rows = [slim]
            fixture_bytes, fixture_sha = _write_jsonl(fixture_rows, fixture_path)
            break

    # Signals
    signals = CONFIG_SIGNAL_DETECTORS[config](profile, normalized)
    available = sorted(k for k, v in signals.items() if v)
    missing = sorted(k for k, v in signals.items() if not v)

    derived_fields = ["created_at_s", "updated_at_s", "duration_s",
                       "is_error", "is_cancelled"]
    if config == "tool_summary":
        # In tool_summary, the duration_ms in the normalized sample is the
        # median of the aggregate bucket — that's derived, not real.
        derived_fields = derived_fields + ["duration_ms", "duration_s"]
    proxy_fields = [
        "args_count", "kwargs_key_count", "series_rows_count",
        "scenario_rows_count", "result_payload_key_count",
        "result_payload_bytes", "artifacts_bytes",
    ]

    if config == "operations":
        limitations = [
            "Real measured MCP-style agent-runtime tool-call execution telemetry exported from Faruk Alpay's Lightcap runtime (operations.parquet).",
            "Closed tool-runtime end-to-end timing — duration_ms is the tool-call wall-clock from request to response, NOT GPU TTFT/TPOT, and NOT LLM serving latency.",
            "NO model_id / NO input_tokens / NO output_tokens / NO GPU type / NO queue depth / NO replica count / NO cache state / NO LLM-serving signal. Tool-runtime trace only.",
            "Single export (one runtime, ~7 days, 2,262 operations across 22 tools). Treat as routing-quality + failure-rate + tail-latency PRIOR for agent workloads — not as a serving telemetry calibration source. Pilot telemetry remains the only Tier 1 calibration source.",
            "Raw args / kwargs / result payload bodies are NOT redistributed in this export — only fingerprints (args_fingerprint = sha256), counts (args_count, kwargs_key_count), key lists (pipe-delimited), and byte totals (result_payload_bytes, artifacts_bytes). Error messages are stored as a preview + sha256 only.",
        ]
    else:
        limitations = [
            "Pre-aggregated per-(tool_name, status) bucket summary from Faruk Alpay's Lightcap tool-runtime (tool_summary.parquet, 32 rows, 22 distinct tools).",
            "Derived aggregate latency — the normalized sample's duration_ms is the per-bucket MEDIAN computed by the upstream exporter; field_quality=derived. The exact avg / median / p95 per (tool, status) live in statistical_rollups.json::per_tool_status_aggregates.",
            "NOT a per-call trace — operations.parquet config is the per-call counterpart. Use this config as a quick per-tool latency prior; use operations for distributional analysis.",
            "NOT GPU TTFT/TPOT, NOT LLM serving telemetry — closed tool-runtime end-to-end timing only. No model_id / no input_tokens / no GPU type / no queue / no replica / no cache state.",
            "Same provenance + scope caveats as the operations config — single runtime, single ~7-day window, 22 tools.",
        ]

    strength = _statistical_sample_strength(len(normalized))

    git_sha = _git_sha()
    provenance = (
        f"{DATASET_ID}@{config}#{target['raw_file']}"
        f"#bytes={manifest['downloaded_bytes']}#git={git_sha[:7] if git_sha else 'na'}"
    )

    summary = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": "cc-by-4.0",
        "gated": False,
        "raw_schema": list(profile["raw_columns"]),
        "normalized_schema": list(profile["normalized_columns"]),
        "unknown_columns": [],  # every raw column is mapped (or explicitly rejected below)
        "rejected_columns": rejected,
        "canonical_trace_type": "tool_runtime_trace",
        "available_signals": available,
        "missing_signals": missing,
        "derived_fields": derived_fields,
        "proxy_fields": proxy_fields,
        "synthetic_fields": [],
        "provenance": provenance,
        "limitations": limitations,
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_bytes,
        "analysis_sample_rows": len(normalized),
        "analysis_sample_bytes": normalized_bytes,
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_bytes,
        "normalized_sample_rows": len(normalized),
        "normalized_sample_bytes": normalized_bytes,
        "normalized_sample_sha256": normalized_sha,
        "sample_sha256": fixture_sha,
        "sampling_method": "full_bounded",
        "stratification_keys": ["tool_name", "status", "stage", "error_type"],
        "statistical_sample_strength": strength,
        "ingestion_timestamp_s": time.time(),
        "summary_path_relative": f"data/external/hf/{SAFE_DATASET}/{config}/processed/summary.json",
        "schema_profile_path": f"data/external/hf/{SAFE_DATASET}/{config}/processed/schema_profile.json",
        "schema_mapping_path": f"data/external/hf/{SAFE_DATASET}/{config}/processed/schema_mapping.json",
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str, sort_keys=True)

    decision = promotion.evaluate_promotion(summary)

    return {
        "target": target,
        "manifest": manifest,
        "audit_status": "ok",
        "summary_path": str(summary_path),
        "profile_path": str(schema_profile_path),
        "mapping_path": str(schema_mapping_path),
        "rollups_path": str(rollups_path),
        "fixture_path": str(fixture_path),
        "normalized_sample_path": str(normalized_sample_path),
        "analysis_sample_path": str(analysis_sample_path),
        "decision": decision,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=None,
                        help="Limit to one or more configs (default: all).")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Re-download even if raw file exists.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.info("HF_TOKEN not set — public-only access.")
    requested = set(args.config) if args.config else None

    DISC_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    _install_timeout(PER_DATASET_TIMEOUT_S)
    try:
        per_config: list[dict] = []
        for tgt in TARGETS:
            if requested and tgt["config_name"] not in requested:
                continue
            logger.info("=== %s :: %s ===", DATASET_ID, tgt["config_name"])
            result = audit_one(tgt, token=token,
                                force_redownload=args.force_redownload)
            d = result.get("decision") or {}
            logger.info("  %s -> state=%s tags=%s",
                        tgt["config_name"], d.get("state"),
                        d.get("promotion_tags"))
            per_config.append({
                "config": tgt["config_name"],
                "audit_status": result.get("audit_status"),
                "manifest": result.get("manifest"),
                "summary_path": result.get("summary_path"),
                "decision_state": d.get("state"),
                "decision_tags": d.get("promotion_tags"),
            })

        summary_out = {
            "dataset_id": DATASET_ID,
            "wrote_at_s": time.time(),
            "configs": per_config,
        }
        out_path = (DISC_DIR /
                    "lightcap_runtime_telemetry_ingest_summary.json")
        with open(out_path, "w") as fh:
            json.dump(summary_out, fh, indent=2, default=str, sort_keys=True)
        logger.info("Wrote %s (%d configs)", out_path, len(per_config))
    finally:
        _clear_timeout()
    return 0


if __name__ == "__main__":
    sys.exit(main())
