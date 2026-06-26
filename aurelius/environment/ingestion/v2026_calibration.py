"""v2026 streaming calibration — wire the exact aggregators to each fact table.

Maps every FleetPlane calibration category to mergeable exact aggregators
(:mod:`v2026_stream`) and the per-partition fold over the real parquet columns
(verified against the live trace). Driven by :func:`v2026_stream.stream_archive`,
this computes FULL_TRACE_EXACT calibration artifacts incrementally, bounded disk,
resumable.

Categories (build spec) → table.column (real v2026 schema):
  * GPU utilization        ← pod_hourly.avg_gpu_sm_util            (stats + hist)
  * GPU memory             ← pod_hourly.avg_memory_util/avg_gpu_mem_gib (stats + hist)
  * priority mix           ← pod_hourly.priority_class             (exact counter)
  * job/model type mix     ← pod_hourly.job_type_public/model_type_public + is_genai_request
  * queue-delay            ← pod_hourly.schedule_delay_sec/ready_delay_sec (stats+hist)
  * capacity envelope      ← server_hourly.gpu_count (sum) / pod_hourly.gpu_request
  * placement/fragmentation← pod_hourly.gpu_request vs server_gpu_count (sums)
  * GPU type mix           ← server_hourly.gpu_spec_public / pod_hourly.gpu_spec_public
  * rack / asw locality    ← server_hourly.asw_id                  (counter)
  * network rx/tx          ← network_hourly.rx/tx_gibps_avg        (stats + hist)
  * job durations          ← job_execution_summary.duration_hours  (stats + hist)
"""

from __future__ import annotations

import json

from ..data_tier import FULL_TRACE_EXACT, SUBSET_TRACE
from .v2026_stream import (
    ARCHIVES,
    PREFETCH_MAX_BYTES,
    ExactCounter,
    ExactHistogram,
    ExactStats,
    StreamResult,
    head_size,
    stream_archive_parallel,
    stream_archive_prefetched,
)


def _col(table, name):
    """A column's python values, or [] if the column is absent."""
    return table.column(name).to_pylist() if name in table.column_names else []


# --- pod_hourly (the core fleet table) -------------------------------------

def _pod_aggs():
    return {
        "gpu_sm_util": ExactStats(), "gpu_sm_util_hist": ExactHistogram(0.0, 100.0, 100),
        "gpu_mem_gib": ExactStats(), "gpu_mem_util": ExactStats(),
        "gpu_mem_util_hist": ExactHistogram(0.0, 1.0, 100),
        "cpu_util": ExactStats(),
        "schedule_delay_s": ExactStats(), "schedule_delay_hist": ExactHistogram(0.0, 300.0, 150),
        "ready_delay_s": ExactStats(),
        "gpu_request": ExactStats(), "gpu_mem_request": ExactStats(),
        "used_gpu_hours": ExactStats(), "server_gpu_count": ExactStats(),
        "priority_class": ExactCounter(), "job_type_public": ExactCounter(),
        "model_type_public": ExactCounter(), "is_genai_request": ExactCounter(),
        "gpu_spec_public": ExactCounter(), "state_public": ExactCounter(),
    }


def _pod_fold(aggs, t):
    util = _col(t, "avg_gpu_sm_util")
    aggs["gpu_sm_util"].update(util)
    aggs["gpu_sm_util_hist"].update(util)
    aggs["gpu_mem_gib"].update(_col(t, "avg_gpu_mem_gib"))
    mem = _col(t, "avg_memory_util")
    aggs["gpu_mem_util"].update(mem)
    aggs["gpu_mem_util_hist"].update(mem)
    aggs["cpu_util"].update(_col(t, "avg_cpu_request_util"))
    sd = _col(t, "schedule_delay_sec")
    aggs["schedule_delay_s"].update(sd)
    aggs["schedule_delay_hist"].update(sd)
    aggs["ready_delay_s"].update(_col(t, "ready_delay_sec"))
    aggs["gpu_request"].update(_col(t, "gpu_request"))
    aggs["gpu_mem_request"].update(_col(t, "gpu_mem_request"))
    aggs["used_gpu_hours"].update(_col(t, "used_gpu_hours"))
    aggs["server_gpu_count"].update(_col(t, "server_gpu_count"))
    aggs["priority_class"].update(_col(t, "priority_class"))
    aggs["job_type_public"].update(_col(t, "job_type_public"))
    aggs["model_type_public"].update(_col(t, "model_type_public"))
    aggs["is_genai_request"].update(_col(t, "is_genai_request"))
    aggs["gpu_spec_public"].update(_col(t, "gpu_spec_public"))
    aggs["state_public"].update(_col(t, "state_public"))


# --- server_hourly (inventory + topology) ----------------------------------

def _server_aggs():
    return {"gpu_type": ExactCounter(), "asw_locality": ExactCounter(),
            "cluster": ExactCounter(), "gpu_count": ExactStats(),
            "cpu_capacity_cores": ExactStats()}


def _server_fold(aggs, t):
    aggs["gpu_type"].update(_col(t, "gpu_spec_public"))
    aggs["asw_locality"].update(_col(t, "asw_id"))
    aggs["cluster"].update(_col(t, "cluster_id"))
    aggs["gpu_count"].update(_col(t, "gpu_count"))
    aggs["cpu_capacity_cores"].update(_col(t, "cpu_capacity_cores"))


# --- network_hourly (macro traffic) ----------------------------------------

def _network_aggs():
    # most rx/tx values are < 1 gibps with a long tail → fine-grained low range
    return {"rx_gibps": ExactStats(), "rx_hist": ExactHistogram(0.0, 5.0, 100),
            "tx_gibps": ExactStats(), "tx_hist": ExactHistogram(0.0, 5.0, 100)}


def _network_fold(aggs, t):
    rx = _col(t, "rx_gibps_avg")
    aggs["rx_gibps"].update(rx)
    aggs["rx_hist"].update(rx)
    tx = _col(t, "tx_gibps_avg")
    aggs["tx_gibps"].update(tx)
    aggs["tx_hist"].update(tx)


# --- job_execution_summary (per-job durations + mixes) ---------------------

def _job_aggs():
    return {"duration_hours": ExactStats(), "duration_hist": ExactHistogram(0.0, 168.0, 168),
            "schedule_delay_s": ExactStats(), "schedule_delay_hist": ExactHistogram(0.0, 300.0, 150),
            "ready_delay_s": ExactStats(), "gpu_request": ExactStats(),
            "priority_class": ExactCounter(), "job_type_public": ExactCounter(),
            "model_type_public": ExactCounter(), "is_genai_request": ExactCounter(),
            "schedule_status": ExactCounter(), "ready_status": ExactCounter()}


def _job_fold(aggs, t):
    dur = _col(t, "duration_hours")
    aggs["duration_hours"].update(dur)
    aggs["duration_hist"].update(dur)
    sd = _col(t, "schedule_delay_sec")
    aggs["schedule_delay_s"].update(sd)
    aggs["schedule_delay_hist"].update(sd)
    aggs["ready_delay_s"].update(_col(t, "ready_delay_sec"))
    aggs["gpu_request"].update(_col(t, "gpu_request"))
    aggs["priority_class"].update(_col(t, "priority_class"))
    aggs["job_type_public"].update(_col(t, "job_type_public"))
    aggs["model_type_public"].update(_col(t, "model_type_public"))
    aggs["is_genai_request"].update(_col(t, "is_genai_request"))
    aggs["schedule_status"].update(_col(t, "schedule_status"))
    aggs["ready_status"].update(_col(t, "ready_status"))


_WIRING = {
    "pod_hourly": (_pod_aggs, _pod_fold),
    "server_hourly": (_server_aggs, _server_fold),
    "network_hourly": (_network_aggs, _network_fold),
    "job_execution_summary": (_job_aggs, _job_fold),
}


def calibrate_table(
    table: str, *, work_dir: str, manifest_path: str, max_partitions=None,
    workers: int = 8,
):
    """Stream one v2026 archive → exact calibration artifacts. Resumable."""
    if table not in _WIRING:
        raise ValueError(f"no wiring for table {table!r}; have {sorted(_WIRING)}")
    build, fold = _WIRING[table]
    url = ARCHIVES[table]
    # Archives that fit the disk budget: download whole once (fast) then stream
    # locally. pod_hourly (351 GB) exceeds the cap → parallel range-stream
    # partition-by-partition (download-bound → a small thread pool fetches
    # concurrently while the main thread folds). All paths are exact, resumable,
    # bounded-disk; the merge is order-independent so parallelism stays EXACT.
    if head_size(url) <= PREFETCH_MAX_BYTES:
        return stream_archive_prefetched(
            url, build, fold, work_dir=work_dir,
            manifest_path=manifest_path, max_partitions=max_partitions)
    return stream_archive_parallel(
        url, build, fold, work_dir=work_dir, manifest_path=manifest_path,
        workers=workers, max_partitions=max_partitions)


def materialize_artifact(table: str, manifest_path: str, *, n_partitions_total=None):
    """Rebuild the calibration artifact dict from a checkpoint manifest WITHOUT
    re-streaming. A full pod_hourly pass is egress-time-bound (hours) and spans
    container sessions; this lets each session publish an accurate artifact —
    SUBSET_TRACE while in progress, FULL_TRACE_EXACT once every partition is in —
    from the committed manifest alone (exact aggregator state, every row counted)."""
    if table not in _WIRING:
        raise ValueError(f"no wiring for table {table!r}; have {sorted(_WIRING)}")
    build, _ = _WIRING[table]
    with open(manifest_path) as f:
        m = json.load(f)
    aggs = build()
    state = m.get("state", {})
    for name, agg in aggs.items():
        if name in state:
            agg.__dict__.update(type(agg).from_state(state[name]).__dict__)
    n_done = len(m.get("processed", []))
    total = m.get("n_partitions_total") or n_partitions_total or n_done
    label = FULL_TRACE_EXACT if total and n_done == total else SUBSET_TRACE
    return StreamResult(
        archive=m.get("archive", ARCHIVES.get(table, "")),
        n_partitions_total=total, n_partitions_done=n_done,
        artifacts={k: v.to_dict() for k, v in aggs.items()},
        label=label, bytes_streamed=m.get("bytes_streamed", 0)).to_dict()


__all__ = ["calibrate_table", "materialize_artifact"]
