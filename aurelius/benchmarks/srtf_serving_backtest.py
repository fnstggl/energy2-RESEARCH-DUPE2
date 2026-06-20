"""SRTF serving-queue backtest — the request-level evaluation of shortest-job-
first ordering on a real LLM serving trace (arXiv:2604.06970), extended with
SRTF-with-Aging anti-starvation guard [run 2026-06-20-i].

Why this module exists
----------------------
Run 2026-06-20-f wired a ``predicted_output_tokens`` sort key into the
*batch* ``JobScheduler`` and showed it neutral on the 26-day energy trace.
Run 2026-06-20-g (the SRTF-under-contention probe) then showed the batch
scheduler **cannot** express the SRTF benefit at all: its greedy placement has
no queue-wait semantics — when capacity is exhausted it falls back to
``earliest_start`` rather than making a request *wait*, so processing order
never changes a completion time.  The analytical Erlang-C model in
``simulation/cluster/serving.py`` is likewise an aggregate M/M/c formula with no
per-request ordering.

The SRTF result from the literature ("Scheduling the Unschedulable",
arXiv:2604.06970: +32% p90 for short requests vs FIFO) is a **request-level
queue-discipline** effect.  Demonstrating it honestly requires a discrete-event
queue that processes individual requests through a finite server pool under a
chosen ordering.  That is exactly what this module is.

What is real vs. modelled
-------------------------
- **Real:** per-request output-token counts come from the Azure LLM 2024 public
  trace (heavy-tailed: p50≈90, p99≈479, max≈1346).  Inter-arrival *shape*
  (burstiness) comes from the trace timestamps.
- **Documented model (identical across every discipline):**
    * service time  s_i = TTFT_BASE_S + actual_output_tokens · TPOT_S
      (continuous-batching decode physics; the same per-token rate the engine
      uses).
    * ``c`` homogeneous replicas behind one queue (M/G/c, non-preemptive).
    * arrivals are time-warped by a single scalar so cluster utilization hits a
      realistic ``target_rho`` — the public sample is downsampled and its raw
      RPS would leave the pool 85% idle.  The warp preserves the real token
      distribution and burst shape; it is applied identically to FIFO and SRTF.
- **Leakage guard:** the SRTF discipline orders by *predicted* output tokens.
  Service time always uses the *actual* token count.  With a noisy forecast the
  ordering key and the physics are genuinely decoupled.

Disciplines compared through the identical simulator:
  ``fifo``           — serve waiting requests in arrival order.
  ``srtf_perfect``   — serve shortest *predicted* job first; prior = actual tokens.
  ``srtf_forecast``  — shortest predicted first; prior = actual × lognormal noise.
  ``aging_srtf``     — SRTF with aging: key(r,t) = predicted / (1 + α·wait_s).
                       Long requests gain priority as wait grows, bounding
                       starvation while preserving most of the SRTF short-request
                       gain.  Research basis: Astraea (arXiv:2512.14142) aging-
                       based promotion; FlowPrefill (arXiv:2602.16603) preemptive
                       HoL mitigation.

AGING_ALPHA calibration (AGING_ALPHA_DEFAULT = 0.05):
  A p99-length Azure 2024 request (479 tokens) reaches parity with the median
  (90 tokens) after ≈87 seconds of waiting.  Beyond that threshold the long job
  wins priority over any newly-arriving short request — starvation is bounded.

BurstGPT cross-validation [run 2026-06-20-i]:
  ``load_burstgpt_serving_requests`` + ``run_burstgpt_aging_backtest`` replay the
  BurstGPT fixture (avg response ≈ 340 tokens, heavier tail than Azure 2024)
  to cross-validate SRTF and aging_SRTF gain across a second real LLM trace.

Honesty / non-goals (``docs/RESULTS.md`` §8):
- Simulator / public-trace directional result — **not** production savings.
- The server pool ``c`` and the time-warp are identical across disciplines, so
  the infra-dollar denominator is identical and every delta comes purely from
  the **queue ordering**.
"""

from __future__ import annotations

import heapq
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Documented service-physics constants (identical across all disciplines).
# Mirror the engine's serving baselines (traces/backtest.py BASE_TTFT_MS /
# BASE_TPOT_MS) expressed in seconds.
# ---------------------------------------------------------------------------

TTFT_BASE_S: float = 0.150          # fixed prefill / time-to-first-token component
TPOT_S: float = 0.020               # per-output-token decode time (50 tok/s/seq)
GPU_HOUR_USD: float = 2.0           # replica cost for the infra-dollar denominator

# E2E response-time SLA for an interactive request (seconds).  A request is
# "SLA-safe" iff its total response time (queue wait + service) is within this.
DEFAULT_SLA_S: float = 10.0

# BurstGPT SLA: heavier output distribution (avg ~340 tokens → service ~6.95s at
# idle) so the response-time budget is set higher.
DEFAULT_BURSTGPT_SLA_S: float = 30.0

# Aging decay constant: key(r, t) = predicted_tokens / (1 + alpha * wait_s).
# At alpha=0.05 a p99-length Azure 2024 request (479 tok) reaches parity with
# the p50 (90 tok) after ≈87 seconds — bounding starvation without eliminating
# the SRTF short-request benefit.
AGING_ALPHA_DEFAULT: float = 0.05

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AZURE_FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv"
)
DEFAULT_BURSTGPT_FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv"
)


# ---------------------------------------------------------------------------
# Real trace loading
# ---------------------------------------------------------------------------

@dataclass
class _Request:
    idx: int
    arrival_s: float
    actual_tokens: int
    predicted_tokens: float
    service_s: float


def _service_time_s(output_tokens: int) -> float:
    return TTFT_BASE_S + output_tokens * TPOT_S


def load_serving_requests(
    path: str = DEFAULT_AZURE_FIXTURE,
    limit: Optional[int] = None,
) -> list[tuple[float, int]]:
    """Return real ``(arrival_s, output_tokens)`` from the Azure LLM 2024 trace.

    Arrival seconds are relative to the first request; failures (zero output)
    are excluded.  Sorted by arrival time.
    """
    from ..traces.azure_llm import load_csv

    reqs = load_csv(path, include_failures=False)
    reqs.sort(key=lambda r: (r.timestamp_s, r.request_id))
    if not reqs:
        return []
    t0 = reqs[0].timestamp_s
    out = [(r.timestamp_s - t0, r.output_tokens) for r in reqs if r.output_tokens > 0]
    if limit is not None:
        out = out[:limit]
    return out


def load_burstgpt_serving_requests(
    path: str = DEFAULT_BURSTGPT_FIXTURE,
    limit: Optional[int] = None,
) -> list[tuple[float, int]]:
    """Return real ``(arrival_s, output_tokens)`` from a BurstGPT CSV.

    BurstGPT ``Timestamp`` is in seconds (integer, relative to trace start).
    ``Response tokens`` is the output token count.  Failures (zero response)
    are excluded.  Results sorted by arrival time with t0 normalized to 0.
    """
    import csv as _csv

    rows: list[tuple[float, int]] = []
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["Timestamp"])
                resp = int(float(row.get("Response tokens") or 0))
            except (KeyError, ValueError, TypeError):
                continue
            if resp > 0:
                rows.append((ts, resp))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return []
    t0 = rows[0][0]
    out = [(ts - t0, resp) for ts, resp in rows]
    if limit is not None:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# Utilization calibration
# ---------------------------------------------------------------------------

def calibrate_time_warp(
    arrivals: list[tuple[float, int]],
    servers: int,
    target_rho: float,
) -> float:
    """Return the scalar arrival time-warp that yields ``target_rho`` on ``c``.

    rho = lambda_warped * E[S] / c, and lambda_warped = lambda_raw * warp.
    => warp = target_rho * c / (lambda_raw * E[S]).
    """
    if len(arrivals) < 2:
        return 1.0
    span = arrivals[-1][0] - arrivals[0][0]
    if span <= 0:
        return 1.0
    lam_raw = len(arrivals) / span
    mean_service = statistics.mean(_service_time_s(tok) for _, tok in arrivals)
    if lam_raw <= 0 or mean_service <= 0:
        return 1.0
    return target_rho * servers / (lam_raw * mean_service)


# ---------------------------------------------------------------------------
# Discrete-event M/G/c simulator (non-preemptive)
# ---------------------------------------------------------------------------

def simulate_queue(
    requests: list[_Request],
    servers: int,
    discipline: str,
    aging_alpha: float = AGING_ALPHA_DEFAULT,
) -> tuple[dict, dict, dict]:
    """Run a non-preemptive M/G/c discrete-event simulation.

    ``discipline``:
      ``fifo``        — ready requests served in arrival order.
      ``srtf``        — ready requests served shortest *predicted* service first.
      ``aging_srtf``  — SRTF with aging: at dispatch time t the effective key is
                        ``predicted_tokens / (1 + aging_alpha * wait_so_far)``.
                        As wait grows the key falls, giving long-waiting requests
                        higher priority and bounding starvation.

    ``aging_alpha`` only affects the ``aging_srtf`` discipline.  The default
    (0.05) is calibrated so an Azure-2024 p99 request (479 tok) reaches parity
    with the median (90 tok) after ≈87 s of waiting.

    Returns ``(summary, response_map, wait_map)`` where the maps are
    ``{request_idx: seconds}``.  The simulation is deterministic given the
    inputs; ties break on arrival sequence (request index).
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))

    busy: list[float] = []   # min-heap of server completion times

    # For fifo/srtf: priority heap keyed by (discipline_key, idx, request).
    # For aging_srtf: plain list re-evaluated at dispatch time.
    ready_heap: list[tuple] = []
    ready_list: list[_Request] = []
    seq = 0

    response: dict[int, float] = {}
    wait: dict[int, float] = {}

    ai = 0
    INF = float("inf")
    _use_aging = (discipline == "aging_srtf")

    def _push_ready(req: _Request) -> None:
        nonlocal seq
        if _use_aging:
            ready_list.append(req)
        else:
            # SRTF: shortest predicted tokens first; FIFO: arrival seq.
            key = (req.predicted_tokens, seq) if discipline == "srtf" else (seq,)
            heapq.heappush(ready_heap, (key, req.idx, req))
        seq += 1

    def _has_ready() -> bool:
        return bool(ready_list) if _use_aging else bool(ready_heap)

    while ai < n or busy or _has_ready():
        next_arrival = by_arrival[ai].arrival_s if ai < n else INF
        next_completion = busy[0] if busy else INF
        t = min(next_arrival, next_completion)
        if t == INF:
            break

        # 1. free any servers completing at or before t
        while busy and busy[0] <= t:
            heapq.heappop(busy)
        # 2. admit all arrivals at t
        while ai < n and by_arrival[ai].arrival_s <= t:
            _push_ready(by_arrival[ai])
            ai += 1
        # 3. dispatch ready requests to free servers (per discipline)
        if _use_aging:
            # Re-evaluate effective key at current time t for every queued
            # request. O(|ready|) per dispatch event — correct for non-heap aging.
            while ready_list and len(busy) < servers:
                best_i = min(
                    range(len(ready_list)),
                    key=lambda i: (
                        ready_list[i].predicted_tokens
                        / (1.0 + aging_alpha * max(0.0, t - ready_list[i].arrival_s)),
                        ready_list[i].idx,  # stable tiebreak
                    ),
                )
                req = ready_list.pop(best_i)
                wait[req.idx] = t - req.arrival_s
                comp = t + req.service_s
                response[req.idx] = comp - req.arrival_s
                heapq.heappush(busy, comp)
        else:
            while ready_heap and len(busy) < servers:
                _, _, req = heapq.heappop(ready_heap)
                wait[req.idx] = t - req.arrival_s
                comp = t + req.service_s
                response[req.idx] = comp - req.arrival_s
                heapq.heappush(busy, comp)

    resp = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait[r.idx] for r in requests if r.idx in wait]
    summary = _summarize(requests, response, wait, resp, waits, servers)
    return summary, response, wait


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, int(math.ceil(pct / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[max(0, k)]


def _summarize(requests, response, wait, resp, waits, servers) -> dict:
    resp_sorted = sorted(resp)
    wait_sorted = sorted(waits)

    # Short-request subset: predicted size at or below the median.
    # Long-request subset: predicted size above the median.
    # SRTF protects short requests at the cost of deferring long ones; both
    # metrics are needed to assess the starvation trade-off.
    pred = sorted(r.predicted_tokens for r in requests)
    median_pred = pred[len(pred) // 2] if pred else 0.0
    short_resp = sorted(
        response[r.idx] for r in requests
        if r.idx in response and r.predicted_tokens <= median_pred
    )
    long_resp = sorted(
        response[r.idx] for r in requests
        if r.idx in response and r.predicted_tokens > median_pred
    )

    sim_end = max(response.values()) if response else 0.0
    return {
        "requests": len(resp),
        "servers": servers,
        "sim_horizon_s": sim_end,
        "mean_response_s": statistics.mean(resp) if resp else 0.0,
        "p50_response_s": _percentile(resp_sorted, 50),
        "p90_response_s": _percentile(resp_sorted, 90),
        "p99_response_s": _percentile(resp_sorted, 99),
        "mean_wait_s": statistics.mean(waits) if waits else 0.0,
        "p90_wait_s": _percentile(wait_sorted, 90),
        "p99_wait_s": _percentile(wait_sorted, 99),
        "short_p90_response_s": _percentile(short_resp, 90),
        "short_p99_response_s": _percentile(short_resp, 99),
        "long_p90_response_s": _percentile(long_resp, 90),
        "long_p99_response_s": _percentile(long_resp, 99),
        "max_response_s": resp_sorted[-1] if resp_sorted else 0.0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class SRTFServingReport:
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float

    fifo: dict
    srtf_perfect: dict
    srtf_forecast: dict

    # Headline deltas (positive = SRTF better)
    short_p90_improvement_pct: float       # reduction in short-request p90 response
    mean_response_improvement_pct: float
    sla_goodput_delta_pct: float           # increase in SLA-safe goodput/$
    forecast_short_p90_improvement_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "time_warp": round(self.time_warp, 4),
            "sla_s": self.sla_s,
            "fifo": _r(self.fifo),
            "srtf_perfect": _r(self.srtf_perfect),
            "srtf_forecast": _r(self.srtf_forecast),
            "short_p90_improvement_pct": round(self.short_p90_improvement_pct, 2),
            "mean_response_improvement_pct": round(self.mean_response_improvement_pct, 2),
            "sla_goodput_delta_pct": round(self.sla_goodput_delta_pct, 2),
            "forecast_short_p90_improvement_pct": round(self.forecast_short_p90_improvement_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


@dataclass
class SRTFAgingReport:
    """Multi-discipline comparison on a real LLM serving trace [run 2026-06-20-i].

    Compares FIFO, SRTF-perfect, and Aging-SRTF side-by-side, quantifying:
    - how much short-request p90 benefit aging_srtf preserves vs pure SRTF
    - how much long-request p99 starvation aging_srtf recovers vs pure SRTF
    - SLA-safe goodput/$ delta vs FIFO for both disciplines

    Research basis:
    - Astraea (arXiv:2512.14142): aging-based promotion mechanism for LLM serving.
    - FlowPrefill (arXiv:2602.16603): operator-level preemption to mitigate HoL.
    - Equinox (arXiv:2508.16646): holistic fairness with dual-counter starvation
      prevention.
    """
    trace: str                  # "azure_llm_2024" or "burstgpt"
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float
    aging_alpha: float

    fifo: dict
    srtf_perfect: dict
    aging_srtf: dict

    # Short-request p90 response reduction vs FIFO (positive = better)
    srtf_short_p90_improvement_pct: float
    aging_short_p90_improvement_pct: float

    # Long-request p99 response change vs FIFO (negative = regression, positive = improvement)
    srtf_long_p99_delta_pct: float       # typically large negative (starvation)
    aging_long_p99_delta_pct: float      # should be much less negative than srtf

    # SLA-safe goodput/$ delta vs FIFO
    srtf_goodput_delta_pct: float
    aging_goodput_delta_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "time_warp": round(self.time_warp, 4),
            "sla_s": self.sla_s,
            "aging_alpha": self.aging_alpha,
            "fifo": _r(self.fifo),
            "srtf_perfect": _r(self.srtf_perfect),
            "aging_srtf": _r(self.aging_srtf),
            "srtf_short_p90_improvement_pct": round(self.srtf_short_p90_improvement_pct, 2),
            "aging_short_p90_improvement_pct": round(self.aging_short_p90_improvement_pct, 2),
            "srtf_long_p99_delta_pct": round(self.srtf_long_p99_delta_pct, 2),
            "aging_long_p99_delta_pct": round(self.aging_long_p99_delta_pct, 2),
            "srtf_goodput_delta_pct": round(self.srtf_goodput_delta_pct, 2),
            "aging_goodput_delta_pct": round(self.aging_goodput_delta_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _sla_safe_goodput(requests: list[_Request], response: dict, sla_s: float) -> float:
    """Output-token-equivalents of requests completing within the SLA budget."""
    return float(sum(
        r.actual_tokens for r in requests
        if r.idx in response and response[r.idx] <= sla_s
    ))


def _sla_safe_goodput_per_dollar(
    requests: list[_Request], response: dict, sla_s: float, servers: int
) -> float:
    """SLA-safe goodput per infrastructure dollar.

    Denominator is total GPU busy-time (sum of service seconds) priced at
    ``GPU_HOUR_USD`` — this is **identical** across disciplines because every
    discipline processes the same request set on the same servers, so the only
    thing that moves the metric is how many SLA-safe tokens the ordering
    delivers.  (Using max-response horizon instead would let SRTF's starved
    long-request tail unfairly inflate its own denominator.)
    """
    busy_gpu_hours = sum(r.service_s for r in requests) / 3600.0
    infra = max(busy_gpu_hours, 1e-9) * GPU_HOUR_USD
    return _sla_safe_goodput(requests, response, sla_s) / infra


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_srtf_serving_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    forecast_noise_cv: float = 0.30,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
    seed: int = 20260201,
) -> SRTFServingReport:
    """Evaluate SRTF vs FIFO request ordering on the real Azure LLM 2024 trace.

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization; sets the arrival time-warp.
        job_limit: Optional cap on number of real requests.
        forecast_noise_cv: Lognormal CV of the realistic forecast prior.
        sla_s: E2E response-time SLA budget (seconds) for goodput.
        azure_fixture: Real Azure LLM 2024 CSV path.
        seed: Forecast-noise seed.

    Returns:
        ``SRTFServingReport`` with FIFO / SRTF-perfect / SRTF-forecast KPIs
        and headline improvement percentages.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests to simulate a queue")

    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    rng = random.Random(seed)
    sigma = math.sqrt(math.log(1.0 + forecast_noise_cv ** 2)) if forecast_noise_cv > 0 else 0.0

    def _build(prior_mode: str) -> list[_Request]:
        reqs: list[_Request] = []
        for i, (arr, tok) in enumerate(raw):
            if prior_mode == "perfect":
                pred = float(tok)
            elif prior_mode == "forecast":
                pred = max(1.0, tok * math.exp(rng.gauss(0.0, sigma))) if sigma > 0 else float(tok)
            else:  # fifo doesn't use the prior; keep actual for the short-cohort split
                pred = float(tok)
            reqs.append(_Request(
                idx=i,
                arrival_s=arr / warp,    # warp>1 compresses time → higher RPS
                actual_tokens=tok,
                predicted_tokens=pred,
                service_s=_service_time_s(tok),
            ))
        return reqs

    fifo_reqs = _build("fifo")
    perfect_reqs = _build("perfect")
    # rebuild rng so forecast noise is independent & reproducible
    rng = random.Random(seed + 1)
    forecast_reqs = _build("forecast")

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    perfect_sim, perfect_resp, _ = simulate_queue(perfect_reqs, servers, "srtf")
    forecast_sim, forecast_resp, _ = simulate_queue(forecast_reqs, servers, "srtf")

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_perfect = _sla_safe_goodput_per_dollar(perfect_reqs, perfect_resp, sla_s, servers)
    gp_forecast = _sla_safe_goodput_per_dollar(forecast_reqs, forecast_resp, sla_s, servers)

    def _impr(base, new):
        return (base - new) / base * 100.0 if base > 0 else 0.0

    short_p90_impr = _impr(fifo_sim["short_p90_response_s"], perfect_sim["short_p90_response_s"])
    fc_short_p90_impr = _impr(fifo_sim["short_p90_response_s"], forecast_sim["short_p90_response_s"])
    mean_impr = _impr(fifo_sim["mean_response_s"], perfect_sim["mean_response_s"])
    gp_delta = (gp_perfect - gp_fifo) / gp_fifo * 100.0 if gp_fifo > 0 else 0.0

    # attach goodput/$ to the per-discipline dicts for transparency
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    perfect_sim["sla_safe_goodput_per_dollar"] = gp_perfect
    forecast_sim["sla_safe_goodput_per_dollar"] = gp_forecast

    return SRTFServingReport(
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        time_warp=warp,
        sla_s=sla_s,
        fifo=fifo_sim,
        srtf_perfect=perfect_sim,
        srtf_forecast=forecast_sim,
        short_p90_improvement_pct=short_p90_impr,
        mean_response_improvement_pct=mean_impr,
        sla_goodput_delta_pct=gp_delta,
        forecast_short_p90_improvement_pct=fc_short_p90_impr,
    )


def _run_aging_backtest_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
) -> SRTFAgingReport:
    """Internal helper: build, simulate, and report for a given trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build(prior_mode: str) -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # perfect prior for aging comparison
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs = _build("perfect")
    srtf_reqs = _build("perfect")
    aging_reqs = _build("perfect")

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    srtf_sim, srtf_resp, _ = simulate_queue(srtf_reqs, servers, "srtf")
    aging_sim, aging_resp, _ = simulate_queue(
        aging_reqs, servers, "aging_srtf", aging_alpha=aging_alpha
    )

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_srtf = _sla_safe_goodput_per_dollar(srtf_reqs, srtf_resp, sla_s, servers)
    gp_aging = _sla_safe_goodput_per_dollar(aging_reqs, aging_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    srtf_sim["sla_safe_goodput_per_dollar"] = gp_srtf
    aging_sim["sla_safe_goodput_per_dollar"] = gp_aging

    def _impr(base: float, new: float) -> float:
        return (base - new) / base * 100.0 if base > 0 else 0.0

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    fifo_sp90 = fifo_sim["short_p90_response_s"]
    fifo_lp99 = fifo_sim["long_p99_response_s"]

    return SRTFAgingReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        time_warp=warp,
        sla_s=sla_s,
        aging_alpha=aging_alpha,
        fifo=fifo_sim,
        srtf_perfect=srtf_sim,
        aging_srtf=aging_sim,
        srtf_short_p90_improvement_pct=_impr(fifo_sp90, srtf_sim["short_p90_response_s"]),
        aging_short_p90_improvement_pct=_impr(fifo_sp90, aging_sim["short_p90_response_s"]),
        srtf_long_p99_delta_pct=_delta(fifo_lp99, srtf_sim["long_p99_response_s"]),
        aging_long_p99_delta_pct=_delta(fifo_lp99, aging_sim["long_p99_response_s"]),
        srtf_goodput_delta_pct=_delta(gp_fifo, gp_srtf),
        aging_goodput_delta_pct=_delta(gp_fifo, gp_aging),
    )


def run_aging_srtf_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = AGING_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> SRTFAgingReport:
    """Compare FIFO, SRTF-perfect, and Aging-SRTF on the Azure LLM 2024 trace.

    The aging discipline uses key(r, t) = predicted_tokens / (1 + alpha * wait_s).
    This bounds long-request starvation while preserving the bulk of SRTF's
    short-request latency benefit.  Research basis: Astraea (arXiv:2512.14142),
    FlowPrefill (arXiv:2602.16603), Equinox (arXiv:2508.16646).

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on number of real requests.
        aging_alpha: Aging decay constant (default 0.05).
        sla_s: E2E response-time SLA budget.
        azure_fixture: Azure LLM 2024 CSV path.

    Returns:
        ``SRTFAgingReport`` with FIFO / SRTF / aging_SRTF KPIs and deltas.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_aging_backtest_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s
    )


def run_burstgpt_aging_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = AGING_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> SRTFAgingReport:
    """Cross-validate SRTF + Aging-SRTF on the BurstGPT trace.

    BurstGPT has a heavier output-token distribution (avg ~340 tokens vs Azure
    2024's ~104), so the SLA budget is set higher (DEFAULT_BURSTGPT_SLA_S=30s).
    The same simulator, warp calibration, and aging formula are used as in the
    Azure 2024 backtest, making the results directly comparable.

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on number of requests.
        aging_alpha: Aging decay constant (default 0.05).
        sla_s: SLA budget (default 30s for BurstGPT's longer service times).
        burstgpt_fixture: BurstGPT CSV path.

    Returns:
        ``SRTFAgingReport`` with FIFO / SRTF / aging_SRTF KPIs and deltas.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_aging_backtest_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s
    )
