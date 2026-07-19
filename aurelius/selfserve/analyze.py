"""Descriptive analytics computed directly from the customer's own log.

Everything in this module is *measured from their data* — no simulator, no
counterfactual, no Aurelius policy. It is deliberately the first substantive
section of the report: it is unconditionally true, useful on its own (the
"fleet analyzer" layer), and it establishes credibility before any replay
number appears.

Stdlib only, deterministic.
"""

from __future__ import annotations

import math
from collections import Counter

from .extract import ExtractionResult
from .fields import SERVING


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def _dist(values: list[float]) -> dict:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {}
    return {
        "count": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "p50": round(_percentile(vals, 0.50), 3),
        "p95": round(_percentile(vals, 0.95), 3),
        "p99": round(_percentile(vals, 0.99), 3),
        "max": round(vals[-1], 3),
    }


def _top(counter: Counter, n: int = 8) -> list[dict]:
    total = sum(counter.values()) or 1
    return [
        {"value": str(k)[:80], "count": c, "share_pct": round(100 * c / total, 1)}
        for k, c in counter.most_common(n)
    ]


def describe(ext: ExtractionResult) -> dict:
    """Build the descriptive-analytics dict for the report."""
    recs = ext.records
    if not recs:
        return {"empty": True}
    if ext.workload_class == SERVING:
        return _describe_serving(recs)
    return _describe_jobs(recs)


def _describe_serving(recs: list[dict]) -> dict:
    ts = sorted(r["timestamp"] for r in recs if r.get("timestamp") is not None)
    t0, t1 = ts[0], ts[-1]
    span_s = max(1.0, t1 - t0)

    per_min: Counter = Counter()
    for t in ts:
        per_min[int((t - t0) // 60)] += 1
    n_minutes = int(span_s // 60) + 1
    rpm = [per_min.get(i, 0) for i in range(n_minutes)]
    idle_minutes = sum(1 for x in rpm if x == 0)

    prompt = [r["prompt_tokens"] for r in recs if r.get("prompt_tokens") is not None]
    output = [r["output_tokens"] for r in recs if r.get("output_tokens") is not None]
    elapsed = [r["elapsed"] for r in recs if r.get("elapsed") is not None]
    failures = sum(1 for r in recs if r.get("is_failure") is True)
    models = Counter(r["model"] for r in recs if r.get("model"))
    sessions = Counter(r["session_id"] for r in recs if r.get("session_id"))
    repeat_requests = sum(c for c in sessions.values() if c > 1)

    mean_rpm = len(ts) / (span_s / 60.0)
    peak_rpm = max(rpm) if rpm else 0
    return {
        "class": SERVING,
        "requests": len(recs),
        "span_hours": round(span_s / 3600.0, 2),
        "arrival": {
            "mean_rpm": round(mean_rpm, 2),
            "p95_rpm": round(_percentile(sorted(rpm), 0.95), 1),
            "peak_rpm": peak_rpm,
            "burstiness_peak_over_mean": round(peak_rpm / mean_rpm, 1)
            if mean_rpm else 0.0,
            "idle_minutes": idle_minutes,
            "idle_share_pct": round(100 * idle_minutes / max(1, n_minutes), 1),
        },
        "tokens": {
            "prompt": _dist([float(x) for x in prompt]),
            "output": _dist([float(x) for x in output]),
            "total_prompt": sum(prompt),
            "total_output": sum(output),
        },
        "latency_s": _dist([float(x) for x in elapsed]) if elapsed else {},
        "failure_rate_pct": round(100 * failures / len(recs), 2),
        "model_mix": _top(models),
        "session": {
            "distinct_sessions": len(sessions),
            "requests_in_repeat_sessions_pct": round(
                100 * repeat_requests / len(recs), 1) if sessions else 0.0,
        },
        "burst_note": (
            "peak arrivals are "
            f"{round(peak_rpm / mean_rpm, 1) if mean_rpm else 0}× the mean — "
            "static provisioning for the peak idles most of the fleet; "
            "provisioning for the mean burns the SLA during bursts"
            if mean_rpm and peak_rpm / mean_rpm >= 3 else ""
        ),
    }


def _describe_jobs(recs: list[dict]) -> dict:
    sub = sorted(r["submit_time"] for r in recs if r.get("submit_time") is not None)
    t0, t1 = sub[0], sub[-1]
    span_s = max(1.0, t1 - t0)

    durations = [r["duration"] for r in recs if r.get("duration")]
    gpus = [r["gpu_count"] for r in recs if r.get("gpu_count")]
    waits = [r["queue_wait"] for r in recs if r.get("queue_wait") is not None]
    failures = sum(1 for r in recs if r.get("is_failure") is True)
    users = Counter(r["user_or_group"] for r in recs if r.get("user_or_group"))
    types = Counter(r["workload_type"] for r in recs if r.get("workload_type"))
    gpu_types = Counter(r["gpu_type"] for r in recs if r.get("gpu_type"))

    size_class: Counter = Counter()
    for g in gpus:
        if g <= 1:
            size_class["1"] += 1
        elif g <= 4:
            size_class["2-4"] += 1
        elif g <= 8:
            size_class["5-8"] += 1
        else:
            size_class[">8"] += 1

    total_gpu_hours = sum(
        (r["gpu_count"] or 0) * (r["duration"] or 0) / 3600.0 for r in recs
    )

    # Peak concurrent GPU demand via event sweep on (start|submit, end).
    events: list[tuple[float, int]] = []
    for r in recs:
        start = r.get("start_time")
        if start is None:
            start = r["submit_time"]
        dur = r.get("duration") or 0.0
        g = r.get("gpu_count") or 0
        if g and dur > 0:
            events.append((start, g))
            events.append((start + dur, -g))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)

    long_waits = sum(1 for w in waits if w > 6 * 3600)
    return {
        "class": "gpu_jobs",
        "jobs": len(recs),
        "span_days": round(span_s / 86400.0, 2),
        "jobs_per_day": round(len(recs) / max(span_s / 86400.0, 1 / 24), 1),
        "gpu_request": {
            "dist": _dist([float(g) for g in gpus]),
            "size_class_mix": dict(size_class),
        },
        "duration_hours": _dist([d / 3600.0 for d in durations]),
        "queue_wait_hours": _dist([w / 3600.0 for w in waits]) if waits else {},
        "long_wait_jobs_over_6h": long_waits,
        "total_gpu_hours_demanded": round(total_gpu_hours, 1),
        "peak_concurrent_gpu_demand": peak,
        "failure_rate_pct": round(100 * failures / len(recs), 2),
        "top_users": _top(users, 5),
        "workload_type_mix": _top(types, 8),
        "gpu_type_mix": _top(gpu_types, 6),
        "wait_note": (
            f"{long_waits} jobs waited over 6 hours in queue — head-of-line "
            "blocking or fragmentation; exactly what the replay compares "
            "policies on"
            if long_waits else ""
        ),
    }
