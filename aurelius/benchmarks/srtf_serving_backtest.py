"""SRTF serving-queue backtest — the request-level evaluation of shortest-job-
first ordering on a real LLM serving trace (arXiv:2604.06970).

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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AZURE_FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv"
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
) -> tuple[dict, dict, dict]:
    """Run a non-preemptive M/G/c discrete-event simulation.

    ``discipline``:
      ``fifo`` — ready requests served in arrival order.
      ``srtf`` — ready requests served shortest *predicted* service first.

    Returns ``(summary, response_map, wait_map)`` where the maps are
    ``{request_idx: seconds}``.  The simulation is deterministic given the
    inputs; ties in the ready queue break on arrival sequence.
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))

    busy: list[float] = []          # min-heap of server completion times
    # ready queue: heap keyed by discipline priority -> (key, idx, request)
    ready: list[tuple] = []
    seq = 0                          # stable tiebreaker / FIFO order key

    response: dict[int, float] = {}
    wait: dict[int, float] = {}

    ai = 0
    INF = float("inf")

    def _push_ready(req: _Request):
        nonlocal seq
        # SRTF orders by predicted service (shortest first); FIFO by arrival seq.
        key = (req.predicted_tokens, seq) if discipline == "srtf" else (seq,)
        heapq.heappush(ready, (key, req.idx, req))
        seq += 1

    while ai < n or busy or ready:
        next_arrival = by_arrival[ai].arrival_s if ai < n else INF
        next_completion = busy[0] if busy else INF
        t = min(next_arrival, next_completion)
        if t == INF:
            break

        # 1. free any servers completing at t
        while busy and busy[0] <= t:
            heapq.heappop(busy)
        # 2. admit any arrivals at t
        while ai < n and by_arrival[ai].arrival_s <= t:
            _push_ready(by_arrival[ai])
            ai += 1
        # 3. dispatch ready requests to free servers (per discipline)
        while ready and len(busy) < servers:
            _, _, req = heapq.heappop(ready)
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

    # Short-request subset: predicted size below the median (the cohort SRTF is
    # meant to protect from head-of-line blocking by long requests).
    pred = sorted(r.predicted_tokens for r in requests)
    median_pred = pred[len(pred) // 2] if pred else 0.0
    short_resp = sorted(
        response[r.idx] for r in requests
        if r.idx in response and r.predicted_tokens <= median_pred
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
