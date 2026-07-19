"""End-to-end self-serve pipeline: file → mapping → validation → replay → report.

The replay step calls the LOCKED evaluation infrastructure only:

- serving  → ``aurelius.traces.backtest.run_backtest`` (policies incl. the
             ``constraint_aware`` Aurelius policy, ``sla_aware`` headline
             baseline, ``fifo`` sanity baseline),
- gpu_jobs → ``aurelius.traces.gpu_scheduling.run_backtest`` (packing +
             scheduling baselines, ``best_fit`` headline).

Nothing here re-implements physics, policies, or the economics KPI, and the
replay is REFUSED (not fudged) when the validator says the data cannot
support it. Cost basis note: the replay prices GPU-hours with the public
default priors in ``aurelius/benchmarks/economics.py`` — the report labels
this; a pilot substitutes the customer's actual procurement rate
(``docs/RESULTS.md`` §8 gate).
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .analyze import describe
from .extract import extract
from .fields import GPU_JOBS, SERVING
from .fingerprint import build_fingerprint
from .sniff import MappingPlan, RawTable, TableReadError, build_mapping_plan, read_table
from .validate import TIER_INSUFFICIENT, Check, ReadinessReport, validate

DEFAULT_GPUS_PER_NODE = 8
_SERVING_REPLAY_ROW_CAP = 1_500_000


@dataclass
class RunArtifacts:
    """Everything one pipeline run produced (all local)."""

    run_id: str
    created_unix: float
    source_file: str
    ok: bool
    error: Optional[str]
    mapping: Optional[MappingPlan]
    readiness: Optional[ReadinessReport]
    descriptive: dict = field(default_factory=dict)
    replay_summary: Optional[dict] = None
    replay_skipped_reason: Optional[str] = None
    fleet_assumption: Optional[dict] = None
    fingerprint: dict = field(default_factory=dict)
    timings_s: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_unix": self.created_unix,
            "source_file": self.source_file,
            "ok": self.ok,
            "error": self.error,
            "mapping": self.mapping.to_dict() if self.mapping else None,
            "readiness": self.readiness.to_dict() if self.readiness else None,
            "descriptive": self.descriptive,
            "replay_summary": self.replay_summary,
            "replay_skipped_reason": self.replay_skipped_reason,
            "fleet_assumption": self.fleet_assumption,
            "fingerprint": self.fingerprint,
            "timings_s": {k: round(v, 2) for k, v in self.timings_s.items()},
        }


def inspect_file(
    path: str | Path,
    *,
    workload_class: Optional[str] = None,
    overrides: Optional[dict[str, str]] = None,
) -> tuple[RawTable, MappingPlan]:
    """Sniff + map only (the UI's first step). Raises TableReadError on I/O."""
    table = read_table(path)
    plan = build_mapping_plan(
        table, workload_class=workload_class, overrides=overrides
    )
    return table, plan


def _records_to_serving_requests(records: list[dict]):
    from aurelius.traces.schema import NormalizedLLMRequest

    t0 = min(r["timestamp"] for r in records)
    out = []
    for i, r in enumerate(records):
        prompt = r.get("prompt_tokens")
        output = r.get("output_tokens")
        if prompt is None and output is None:
            continue
        prompt = int(prompt or 0)
        output = int(output or 0)
        if prompt < 0 or output < 0:
            continue
        out.append(NormalizedLLMRequest(
            request_id=str(r.get("request_id") or i),
            timestamp_s=float(r["timestamp"] - t0),
            session_id=r.get("session_id"),
            model=str(r.get("model") or "unknown-model"),
            prompt_tokens=prompt,
            output_tokens=output,
            total_tokens=prompt + output,
            elapsed_s=r.get("elapsed"),
            log_type="API log",
            is_failure=bool(r.get("is_failure") is True),
            cache_affinity_key=r.get("session_id"),
        ))
    return out


def _records_to_gpu_jobs(records: list[dict]):
    from aurelius.traces.schema import NormalizedGPUJob

    out = []
    for i, r in enumerate(records):
        out.append(NormalizedGPUJob(
            job_id=str(r.get("job_id") or i),
            submit_time_s=r.get("submit_time"),
            start_time_s=r.get("start_time"),
            end_time_s=r.get("end_time"),
            duration_s=r.get("duration"),
            gpu_count=int(r.get("gpu_count") or 0),
            gpu_type=r.get("gpu_type"),
            gpu_memory_gb=None,
            status=r.get("status"),
            user_or_group=r.get("user_or_group"),
            workload_type=r.get("workload_type"),
            priority=r.get("priority"),
            placement_nodes=None,
            placement_gpus=None,
            is_failed=bool(r.get("is_failure") is True),
            queue_wait_s=r.get("queue_wait"),
        ))
    return out


def _build_fleet(
    descriptive: dict,
    *,
    fleet_gpus: Optional[int],
    gpus_per_node: int,
    gpu_type: Optional[str],
) -> tuple[list, dict]:
    """Build GPUNode fleet from user spec or the peak-demand inference."""
    from aurelius.traces.gpu_packing import GPUNode

    inferred = False
    if fleet_gpus is None:
        peak = int(descriptive.get("peak_concurrent_gpu_demand") or 0)
        max_job = int(
            (descriptive.get("gpu_request", {}).get("dist", {}) or {}).get("max")
            or 1
        )
        fleet_gpus = max(peak, max_job, gpus_per_node)
        inferred = True
    n_nodes = max(1, math.ceil(fleet_gpus / gpus_per_node))
    model = gpu_type or "unspecified-gpu"
    nodes = [
        GPUNode(
            node_id=f"node-{k}",
            gpu_count=gpus_per_node,
            gpu_model=model,
            cpu_milli=0,
            memory_mib=0,
        )
        for k in range(n_nodes)
    ]
    assumption = {
        "fleet_gpus": n_nodes * gpus_per_node,
        "nodes": n_nodes,
        "gpus_per_node": gpus_per_node,
        "gpu_model": model,
        "inferred_from_peak_demand": inferred,
        "note": (
            "fleet inferred from the log's peak concurrent GPU demand — "
            "pass your real fleet size to replace this assumption"
            if inferred else "fleet supplied by user"
        ),
    }
    return nodes, assumption


def run_pipeline(
    path: str | Path,
    *,
    workload_class: Optional[str] = None,
    overrides: Optional[dict[str, str]] = None,
    fleet_gpus: Optional[int] = None,
    gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    gpu_type: Optional[str] = None,
    tick_seconds: float = 60.0,
    max_rows: int = 2_000_000,
) -> RunArtifacts:
    """The whole self-serve flow. Never raises on data problems — only on bugs."""
    run_id = uuid.uuid4().hex[:12]
    created = time.time()
    timings: dict[str, float] = {}

    try:
        t = time.time()
        table, plan = inspect_file(
            path, workload_class=workload_class, overrides=overrides
        )
        timings["sniff"] = time.time() - t
    except TableReadError as exc:
        return RunArtifacts(
            run_id=run_id, created_unix=created, source_file=str(path),
            ok=False, error=str(exc), mapping=None, readiness=None,
        )

    t = time.time()
    ext = extract(table, plan, max_rows=max_rows)
    timings["extract"] = time.time() - t

    t = time.time()
    readiness = validate(plan, ext)
    timings["validate"] = time.time() - t

    t = time.time()
    descriptive = describe(ext) if ext.records else {"empty": True}
    timings["describe"] = time.time() - t

    replay_summary: Optional[dict] = None
    skipped: Optional[str] = None
    fleet_assumption: Optional[dict] = None

    if readiness.tier == TIER_INSUFFICIENT:
        skipped = (
            "replay refused: " + "; ".join(readiness.refusal_reasons)
            if readiness.refusal_reasons
            else "replay refused: validation failed"
        )
    else:
        t = time.time()
        try:
            if ext.workload_class == SERVING:
                from aurelius.traces.backtest import run_backtest

                requests = _records_to_serving_requests(ext.records)
                if len(requests) > _SERVING_REPLAY_ROW_CAP:
                    requests = requests[:_SERVING_REPLAY_ROW_CAP]
                    readiness.checks.append(Check(
                        "serving_row_cap", "warn",
                        f"replay used the first {_SERVING_REPLAY_ROW_CAP:,} "
                        "requests",
                    ))
                result = run_backtest(requests, tick_seconds=tick_seconds)
                replay_summary = result.to_summary_dict()
            elif ext.workload_class == GPU_JOBS:
                from aurelius.traces.gpu_scheduling import run_backtest

                jobs = _records_to_gpu_jobs(ext.records)
                nodes, fleet_assumption = _build_fleet(
                    descriptive, fleet_gpus=fleet_gpus,
                    gpus_per_node=gpus_per_node, gpu_type=gpu_type,
                )
                result = run_backtest(jobs, nodes)
                replay_summary = result.to_summary_dict()
            else:  # pragma: no cover - insufficient tier already refused
                skipped = "replay refused: unknown workload class"
        except Exception as exc:  # noqa: BLE001 - self-serve must fail legibly
            skipped = (
                "replay failed inside the engine — this is a tool bug, not a "
                f"data problem ({type(exc).__name__}: {exc}); please share "
                "the fingerprint so we can reproduce"
            )
        timings["replay"] = time.time() - t

    fp = build_fingerprint(table, plan, ext, readiness)
    return RunArtifacts(
        run_id=run_id,
        created_unix=created,
        source_file=str(Path(path).name),
        ok=True,
        error=None,
        mapping=plan,
        readiness=readiness,
        descriptive=descriptive,
        replay_summary=replay_summary,
        replay_skipped_reason=skipped,
        fleet_assumption=fleet_assumption,
        fingerprint=fp,
        timings_s=timings,
    )
