#!/usr/bin/env python3
"""AUDIT HARNESS — re-run the run-g SRTF serving backtest against stronger
baselines than FIFO.

This is **audit / reporting infrastructure**, not a runtime optimization and not
a new Aurelius scheduler. It exists only to answer the audit question:

    The run-g headline is "+323% SLA-safe goodput/$ vs FIFO". FIFO is the
    weakest possible baseline. What is the delta versus an SLA-aware baseline
    (earliest-deadline-first) and a constraint-aware baseline?

It deliberately REUSES the exact, unmodified run-g service physics, trace
loader, time-warp calibration and goodput/$ definition from
``aurelius.benchmarks.srtf_serving_backtest`` so every number is directly
comparable to the committed run-g result. It only adds extra *queue
disciplines* (baselines) to the same discrete-event M/G/c simulator:

    fifo          — run-g baseline (serve in arrival order).
    sla_aware     — earliest-deadline-first (EDF). deadline_i = arrival_i + SLA.
                    This is the textbook SLA-aware queue discipline.
    srtf_perfect  — run-g optimized (shortest *predicted*=actual first).
    srtf_forecast — run-g optimized (shortest predicted, 30%-CV noisy prior).
    srpt          — preemptive shortest-remaining-processing-time. A REFERENCE
                    upper bound (textbook-optimal mean response); included only
                    to show the long-tail starvation of non-preemptive SRTF is
                    not intrinsic. NOT an Aurelius optimization.

KEY AUDIT FINDING this harness demonstrates empirically:
  Because the Azure LLM 2024 trace carries **no per-request SLA-class / request-
  type label** (its only columns are TIMESTAMP, ContextTokens, GeneratedTokens),
  every request is assigned the *same* SLA budget. Under a uniform SLA budget,
  earliest-deadline-first (sla_aware) is mathematically identical to FIFO
  (deadline order == arrival order). So on THIS trace "sla_aware" collapses onto
  FIFO and "+323% vs FIFO" == "+323% vs (degenerate) sla_aware". A *differentiated*
  SLA-aware baseline would require synthesizing class labels the trace does not
  contain — which this harness refuses to do for any headline number.

  ``constraint_aware`` is a provisioning / region / energy-timing policy. Its
  decision surface (how many replicas, which region, when to run) is ORTHOGONAL
  to single-queue request ordering and has no expression inside a fixed-c
  single-queue simulator. Δ vs constraint_aware is therefore reported as N/A —
  the two policies do not act on the same decision surface.

Honesty (mirrors docs/RESULTS.md §8): simulator / public-trace directional
result, NOT production savings. The denominator (GPU busy-seconds) is identical
across every discipline by construction, so the reported "goodput/$" delta is
purely an SLA-attainment (latency) effect at constant cost — not a cost
reduction.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import statistics
import time

# Reuse the EXACT run-g physics so results are directly comparable.
from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_SLA_S,
    GPU_HOUR_USD,
    _Request,
    _service_time_s,
    calibrate_time_warp,
    load_serving_requests,
)

# ---------------------------------------------------------------------------
# Generalized non-preemptive M/G/c simulator (adds EDF; fifo/srtf identical to
# run-g). Kept structurally identical to run-g's simulate_queue so fifo/srtf
# reproduce the committed numbers exactly.
# ---------------------------------------------------------------------------

def simulate_nonpreemptive(requests, servers: int, discipline: str, sla_s: float):
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    busy: list[float] = []
    ready: list[tuple] = []
    seq = 0
    response: dict[int, float] = {}
    wait: dict[int, float] = {}
    ai = 0
    INF = float("inf")

    def _key(req: _Request):
        if discipline == "srtf":
            return (req.predicted_tokens, seq)
        if discipline == "sla_aware":
            # earliest-deadline-first: deadline = arrival + per-request SLA budget.
            return (req.arrival_s + sla_s, seq)
        return (seq,)  # fifo

    def _push(req: _Request):
        nonlocal seq
        heapq.heappush(ready, (_key(req), req.idx, req))
        seq += 1

    while ai < n or busy or ready:
        next_arrival = by_arrival[ai].arrival_s if ai < n else INF
        next_completion = busy[0] if busy else INF
        t = min(next_arrival, next_completion)
        if t == INF:
            break
        while busy and busy[0] <= t:
            heapq.heappop(busy)
        while ai < n and by_arrival[ai].arrival_s <= t:
            _push(by_arrival[ai])
            ai += 1
        while ready and len(busy) < servers:
            _, _, req = heapq.heappop(ready)
            wait[req.idx] = t - req.arrival_s
            comp = t + req.service_s
            response[req.idx] = comp - req.arrival_s
            heapq.heappush(busy, comp)
    return response, wait


# ---------------------------------------------------------------------------
# Preemptive SRPT (reference upper bound only — NOT an Aurelius optimization).
# ---------------------------------------------------------------------------

def simulate_srpt(requests, servers: int):
    """Preemptive shortest-remaining-processing-time on c servers.

    Event-driven: between events the c in-system jobs with the smallest remaining
    service run in parallel. Total service work is conserved (no preemption
    overhead modelled), so GPU busy-seconds are identical to the non-preemptive
    disciplines — keeping the goodput/$ denominator identical across all rows.
    """
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    n = len(by_arrival)
    remaining = {r.idx: r.service_s for r in by_arrival}
    response: dict[int, float] = {}
    in_system: list[int] = []        # idxs present, not yet finished
    svc = {r.idx: r for r in by_arrival}
    ai = 0
    t = by_arrival[0].arrival_s
    INF = float("inf")

    while ai < n or in_system:
        if not in_system:
            t = by_arrival[ai].arrival_s
            in_system.append(by_arrival[ai].idx)
            ai += 1
            while ai < n and by_arrival[ai].arrival_s <= t:
                in_system.append(by_arrival[ai].idx)
                ai += 1
            continue
        # c jobs with smallest remaining run now
        in_system.sort(key=lambda i: (remaining[i], svc[i].arrival_s, i))
        running = in_system[:servers]
        min_remaining = min(remaining[i] for i in running)
        next_arrival = by_arrival[ai].arrival_s if ai < n else INF
        dt = min(min_remaining, next_arrival - t)
        if dt <= 0:
            dt = 0.0
        for i in running:
            remaining[i] -= dt
        t += dt
        finished = [i for i in running if remaining[i] <= 1e-9]
        for i in finished:
            response[i] = t - svc[i].arrival_s
            in_system.remove(i)
        while ai < n and by_arrival[ai].arrival_s <= t:
            in_system.append(by_arrival[ai].idx)
            ai += 1
    return response, {}


# ---------------------------------------------------------------------------
# KPIs (run-g definitions; denominator identical across disciplines)
# ---------------------------------------------------------------------------

def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, int(math.ceil(p / 100.0 * len(s)) - 1))
    return s[max(0, k)]


def kpis(requests, response, wait, servers, sla_s, runtime_s):
    resp = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait[r.idx] for r in requests if r.idx in wait] if wait else []
    busy_gpu_hours = sum(r.service_s for r in requests) / 3600.0
    infra = max(busy_gpu_hours, 1e-9) * GPU_HOUR_USD
    sla_safe_tokens = sum(
        r.actual_tokens for r in requests
        if r.idx in response and response[r.idx] <= sla_s
    )
    viol = sum(1 for r in requests if r.idx in response and response[r.idx] > sla_s)
    pred = sorted(r.predicted_tokens for r in requests)
    median_pred = pred[len(pred) // 2] if pred else 0.0
    short_resp = [response[r.idx] for r in requests
                  if r.idx in response and r.predicted_tokens <= median_pred]
    long_resp = [response[r.idx] for r in requests
                 if r.idx in response and r.predicted_tokens > median_pred]
    return {
        "sla_safe_goodput_per_dollar": sla_safe_tokens / infra,
        "gpu_hours": busy_gpu_hours,
        "cost_usd": infra,
        "sla_violations": viol,
        "sla_violation_pct": 100.0 * viol / len(resp) if resp else 0.0,
        "mean_response_s": statistics.mean(resp) if resp else 0.0,
        "p99_response_s": _pct(resp, 99),
        "short_p90_response_s": _pct(short_resp, 90),
        "long_p99_response_s": _pct(long_resp, 99),
        "mean_queue_wait_s": statistics.mean(waits) if waits else 0.0,
        "p99_queue_wait_s": _pct(waits, 99) if waits else 0.0,
        "runtime_s": runtime_s,
    }


def run(servers=4, target_rho=0.85, sla_s=DEFAULT_SLA_S, forecast_cv=0.30,
        seed=20260201, fixture=DEFAULT_AZURE_FIXTURE, job_limit=None):
    raw = load_serving_requests(fixture, limit=job_limit)
    if len(raw) < 2:
        raise SystemExit("need >=2 requests")
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)
    sigma = math.sqrt(math.log(1.0 + forecast_cv ** 2)) if forecast_cv > 0 else 0.0

    def build(mode, rng=None):
        out = []
        for i, (arr, tok) in enumerate(raw):
            if mode == "forecast" and sigma > 0:
                pred = max(1.0, tok * math.exp(rng.gauss(0.0, sigma)))
            else:
                pred = float(tok)
            out.append(_Request(idx=i, arrival_s=arr / warp, actual_tokens=tok,
                                 predicted_tokens=pred, service_s=_service_time_s(tok)))
        return out

    rows = {}

    def _timed(name, reqs, fn):
        t0 = time.perf_counter()
        resp, wait = fn(reqs)
        rt = time.perf_counter() - t0
        rows[name] = kpis(reqs, resp, wait, servers, sla_s, rt)

    base = build("perfect")
    _timed("fifo", base, lambda r: simulate_nonpreemptive(r, servers, "fifo", sla_s))
    _timed("sla_aware_edf", base, lambda r: simulate_nonpreemptive(r, servers, "sla_aware", sla_s))
    _timed("srtf_perfect", base, lambda r: simulate_nonpreemptive(r, servers, "srtf", sla_s))
    fc = build("forecast", random.Random(seed + 1))
    _timed("srtf_forecast", fc, lambda r: simulate_nonpreemptive(r, servers, "srtf", sla_s))
    _timed("srpt_reference", base, lambda r: simulate_srpt(r, servers))

    return {
        "config": {
            "servers": servers, "target_rho": target_rho, "sla_s": sla_s,
            "time_warp": warp, "total_requests": len(raw),
            "forecast_cv": forecast_cv, "fixture": os.path.basename(fixture),
        },
        "rows": rows,
        "constraint_aware": "N/A — provisioning/region policy, no single-queue "
                            "ordering analog (orthogonal decision surface)",
        "shadow_tag": "shadow_only_simulator_result_not_production_savings",
    }


def _delta(new, base):
    return (new - base) / base * 100.0 if base else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.85)
    ap.add_argument("--sla", type=float, default=DEFAULT_SLA_S)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    res = run(servers=args.servers, target_rho=args.rho, sla_s=args.sla)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return

    rows = res["rows"]
    fifo_g = rows["fifo"]["sla_safe_goodput_per_dollar"]
    edf_g = rows["sla_aware_edf"]["sla_safe_goodput_per_dollar"]
    ca_note = "N/A"
    print(f"\nSRTF stronger-baseline audit — Azure LLM 2024 sample, "
          f"c={res['config']['servers']}, rho={res['config']['target_rho']}, "
          f"SLA={res['config']['sla_s']}s, warp={res['config']['time_warp']:.2f}x")
    print("(denominator = GPU busy-seconds, IDENTICAL across rows → delta is "
          "pure SLA-attainment at constant cost)\n")
    hdr = (f"{'policy':<16}{'goodput/$':>12}{'Δ FIFO':>10}{'Δ SLA-aw':>10}"
           f"{'Δ CA':>7}{'SLAviol%':>10}{'qP99 s':>9}{'cost$':>9}{'GPUh':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name in ["fifo", "sla_aware_edf", "srtf_perfect", "srtf_forecast", "srpt_reference"]:
        r = rows[name]
        g = r["sla_safe_goodput_per_dollar"]
        print(f"{name:<16}{g:>12.1f}{_delta(g, fifo_g):>9.1f}%"
              f"{_delta(g, edf_g):>9.1f}%{ca_note:>7}"
              f"{r['sla_violation_pct']:>9.2f}%{r['p99_queue_wait_s']:>9.1f}"
              f"{r['cost_usd']:>9.2f}{r['gpu_hours']:>8.2f}")
    print("\nlong-request p99 response (starvation cost):")
    for name in ["fifo", "sla_aware_edf", "srtf_perfect", "srpt_reference"]:
        print(f"  {name:<16}{rows[name]['long_p99_response_s']:>10.1f} s")
    print(f"\nconstraint_aware: {res['constraint_aware']}")
    print(f"sla_aware_edf == fifo? "
          f"{abs(edf_g - fifo_g) < 1e-6} (uniform SLA → EDF degenerates to FIFO)")


if __name__ == "__main__":
    main()
