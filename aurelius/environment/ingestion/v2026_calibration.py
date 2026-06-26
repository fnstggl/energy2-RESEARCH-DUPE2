"""v2026 streaming calibration — wire the exact aggregators to each fact table.

Maps every FleetPlane calibration category to mergeable exact aggregators
(:mod:`v2026_stream`) and the per-partition fold over the real parquet columns.
Driven by :func:`v2026_stream.stream_archive`, this computes FULL_TRACE_EXACT
calibration artifacts incrementally, with bounded disk, resumable.

Categories (build spec) → table.column:
  * GPU utilization        ← pod_hourly.avg_gpu_sm_util            (stats + hist)
  * GPU memory             ← pod_hourly.avg_memory_util/avg_gpu_mem_gib (stats + hist)
  * priority mix           ← pod_hourly.priority_class             (exact counter)
  * job/model type mix     ← pod_hourly.job_type_public/model_type_public (counter)
  * queue-delay            ← pod_hourly.schedule_delay_sec/ready_delay_sec (stats+hist)
  * capacity / gpu_request ← pod_hourly.gpu_request                (stats)
  * GPU type mix           ← server_hourly.gpu_spec_public         (counter)
  * rack / asw locality    ← server_hourly.asw_id                  (counter)
  * server inventory       ← server_hourly.gpu_count               (stats)
  * network rx/tx          ← network_hourly.rx/tx_gibps_avg        (stats + hist)
  * job durations          ← job_execution_summary.* delays        (stats + hist)
"""

from __future__ import annotations

from .v2026_stream import ARCHIVES, ExactCounter, ExactHistogram, ExactStats, stream_archive


def _col(table, name):
    """Return a column's python values, or [] if the column is absent."""
    return table.column(name).to_pylist() if name in table.column_names else []


# --- per-table aggregator factories + fold functions -----------------------

def _pod_aggs():
    return {
        "gpu_sm_util": ExactStats(), "gpu_sm_util_hist": ExactHistogram(0.0, 100.0, 50),
        "gpu_mem_util": ExactStats(), "gpu_mem_util_hist": ExactHistogram(0.0, 1.0, 50),
        "schedule_delay_s": ExactStats(), "schedule_delay_hist": ExactHistogram(0.0, 120.0, 60),
        "gpu_request": ExactStats(),
        "priority_class": ExactCounter(), "job_type_public": ExactCounter(),
        "model_type_public": ExactCounter(),
    }


def _pod_fold(aggs, t):
    util = _col(t, "avg_gpu_sm_util")
    aggs["gpu_sm_util"].update(util)
    aggs["gpu_sm_util_hist"].update(util)
    mem = _col(t, "avg_memory_util")
    aggs["gpu_mem_util"].update(mem)
    aggs["gpu_mem_util_hist"].update(mem)
    sd = _col(t, "schedule_delay_sec")
    aggs["schedule_delay_s"].update(sd)
    aggs["schedule_delay_hist"].update(sd)
    aggs["gpu_request"].update(_col(t, "gpu_request"))
    aggs["priority_class"].update(_col(t, "priority_class"))
    aggs["job_type_public"].update(_col(t, "job_type_public"))
    aggs["model_type_public"].update(_col(t, "model_type_public"))


def _server_aggs():
    return {"gpu_type": ExactCounter(), "asw_locality": ExactCounter(),
            "gpu_count": ExactStats()}


def _server_fold(aggs, t):
    aggs["gpu_type"].update(_col(t, "gpu_spec_public"))
    aggs["asw_locality"].update(_col(t, "asw_id"))
    aggs["gpu_count"].update(_col(t, "gpu_count"))


def _network_aggs():
    return {"rx_gibps": ExactStats(), "rx_hist": ExactHistogram(0.0, 50.0, 50),
            "tx_gibps": ExactStats(), "tx_hist": ExactHistogram(0.0, 50.0, 50)}


def _network_fold(aggs, t):
    rx = _col(t, "rx_gibps_avg")
    aggs["rx_gibps"].update(rx)
    aggs["rx_hist"].update(rx)
    tx = _col(t, "tx_gibps_avg")
    aggs["tx_gibps"].update(tx)
    aggs["tx_hist"].update(tx)


_WIRING = {
    "pod_hourly": (_pod_aggs, _pod_fold),
    "server_hourly": (_server_aggs, _server_fold),
    "network_hourly": (_network_aggs, _network_fold),
}


def calibrate_table(
    table: str, *, work_dir: str, manifest_path: str, max_partitions=None,
):
    """Stream one v2026 archive → exact calibration artifacts. Resumable."""
    if table not in _WIRING:
        raise ValueError(f"no wiring for table {table!r}; have {sorted(_WIRING)}")
    build, fold = _WIRING[table]
    return stream_archive(
        ARCHIVES[table], build, fold,
        work_dir=work_dir, manifest_path=manifest_path, max_partitions=max_partitions)


__all__ = ["calibrate_table"]
