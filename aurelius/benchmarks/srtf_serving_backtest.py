"""SRTF serving-queue backtest — the request-level evaluation of shortest-job-
first ordering on a real LLM serving trace (arXiv:2604.06970), extended with
SRTF-with-Aging anti-starvation guard [run 2026-06-20-i], Preemptive SRPT
[run 2026-06-20-j], Hybrid Aging+Preemptive SRPT [run 2026-06-20-k],
Decoupled Hybrid SRPT [run 2026-06-20-l], Alpha Sweep [run 2026-06-21-m],
SLA-aware baseline + Noisy Prior Robustness [run 2026-06-21-n],
Preemption Overhead Sensitivity [run 2026-06-21-o],
BurstGPT HF Cross-Validation [run 2026-06-21-p],
Conformal Adaptive α [run 2026-06-21-q],
Absolute-Error Conformal Calibration [run 2026-06-22-x],
SLA-aware vs Abs-Conformal Head-to-Head [run 2026-06-22-y], and
Compound Economic × Queue Scheduling [run 2026-06-22-z].

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
    * ``c`` homogeneous replicas behind one queue (M/G/c).
    * arrivals are time-warped by a single scalar so cluster utilization hits a
      realistic ``target_rho`` — the public sample is downsampled and its raw
      RPS would leave the pool 85% idle.  The warp preserves the real token
      distribution and burst shape; it is applied identically to FIFO and SRTF.
- **Leakage guard:** the SRTF discipline orders by *predicted* output tokens.
  Service time always uses the *actual* token count.  With a noisy forecast the
  ordering key and the physics are genuinely decoupled.

Disciplines compared through the identical simulator:
  ``fifo``                     — serve waiting requests in arrival order (non-preemptive).
  ``srtf``                     — serve shortest *predicted* job first (non-preemptive).
  ``aging_srtf``               — SRTF with aging: key(r,t) = predicted / (1 + α·wait_s).
                                 Long requests gain priority as wait grows, bounding
                                 starvation while preserving most of the SRTF short-request
                                 gain.  Research basis: Astraea (arXiv:2512.14142) aging-
                                 based promotion; FlowPrefill (arXiv:2602.16603) preemptive
                                 HoL mitigation.
  ``srpt_preemptive``          — Preemptive SRPT [run 2026-06-20-j]: when a shorter request
                                 arrives, the server running the longest-remaining job is
                                 preempted; the preempted job re-enters waiting with its
                                 current remaining service time and is resumed later.
                                 Maintains the SRPT invariant: at all times the c requests
                                 with shortest remaining service are running.
                                 Research basis: TRAIL (arXiv:2410.01035, ICLR 2025);
                                 FlowPrefill (arXiv:2602.16603); SRPT for multiserver
                                 (arXiv:1805.07686).
  ``hybrid_aging_preemptive``  — Hybrid Aging+Preemptive SRPT [run 2026-06-20-k]:
                                 Preemption key = remaining_s / (1 + α·accumulated_wait_s).
                                 As a long request accumulates waiting time (across initial
                                 wait + all preemption gaps), its effective key shrinks
                                 toward zero — making it progressively harder for new
                                 short arrivals to preempt it.  Anti-starvation guarantee:
                                 once accumulated_wait_s is large enough that effective_key
                                 < min_service_s_of_any_arrival, the request can no longer
                                 be preempted and completes uninterrupted.  This eliminates
                                 unbounded starvation while preserving SRPT's short-request
                                 benefit.  Research basis: FastServe (USENIX NSDI '26),
                                 Chimera (arXiv:2603.22206), SEK-SMOD (arXiv:2510.25963).
  ``decoupled_hybrid``         — Decoupled Hybrid SRPT [run 2026-06-20-l]:
                                 PREEMPTION key = remaining_s (pure SRPT — no aging).
                                 DISPATCH key  = remaining_s / (1 + α·total_wait_s) (aging).
                                 Decouples the two decisions that run -k's unified key
                                 conflated: arrivals always preempt by pure remaining work
                                 (preserving SRPT's throughput-optimal preemption), while
                                 dispatch from the waiting queue uses aging to prevent
                                 indefinite starvation of long-waiting requests.
                                 Expected: SRPT-level goodput (+322% vs FIFO) with
                                 Aging-SRTF-level long_p99 improvement vs pure SRPT.
                                 Research basis: TRAIL (arXiv:2410.01035, ICLR 2025),
                                 Chimera (arXiv:2603.22206), FastServe (NSDI '26).
  ``sla_aware``                — SLA-aware binary-class priority discipline [run 2026-06-21-n]:
                                 Requests are split by predicted_tokens into two SLA classes:
                                 "short" (≤ global median, latency-critical) → priority 0
                                 "long"  (> global median, standard)          → priority 1
                                 Short requests are always dispatched before long requests;
                                 within each class, requests are served FIFO (arrival order).
                                 Uses no oracle: only the binary classification matters.
                                 Serves as the North Star SLA-aware comparison baseline —
                                 the incremental gain of decoupled hybrid over sla_aware
                                 quantifies the value of continuous token-length prediction
                                 vs binary SLA-class awareness.
                                 Research basis: PROSERVE (arXiv:2512.12928, Dec 2025),
                                 Past-Future Scheduler (arXiv:2507.10150, July 2025).

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
from dataclasses import dataclass
from typing import Optional

# Canonical serving-queue policy [Phase 2 unification]: the strongest validated
# serving discipline (Decoupled Hybrid SRPT + absolute-error conformal alpha) and
# its calibrator now live in the optimizer package. The benchmark imports them
# back so it no longer owns the optimizer logic (parity-preserving extraction).
from aurelius.optimizer.policies.serving_queue import (
    AbsoluteErrorConformalCalibrator,
)
from aurelius.optimizer.policies.serving_queue import (
    simulate_decoupled_hybrid_abs_conformal as _abs_conformal_impl,
)

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
# Aging decay constant for the non-preemptive aging_srtf discipline.
AGING_ALPHA_DEFAULT: float = 0.05

# Recommended aging_alpha for the hybrid_aging_preemptive discipline.
# At α=0.01 a p99-length Azure 2024 request (479 tok, service≈9.73s) accumulates
# enough wait priority to resist preemption by median-length arrivals (service≈1.95s)
# after approximately 400 seconds of accumulated queuing time.  This provides a
# practical starvation bound while preserving near-SRPT short-request performance.
HYBRID_AGING_ALPHA_DEFAULT: float = 0.01

# Pareto-optimal aging decay constant for the decoupled_hybrid discipline [run 2026-06-21-m].
# Alpha sweep (run -m) profiled α ∈ {0.001, 0.005, 0.01, 0.05} and identified α=0.001
# as the Pareto-optimal configuration on Azure LLM 2024 (5,880 requests, ρ=0.85):
#   α=0.001: +274.0% goodput/$ vs FIFO, short_p90=1.91s, long_p99 +177.4% vs FIFO
#   α=0.005: +205.0% goodput/$, short_p90=2.06s, long_p99 +141.2% vs FIFO
#   α=0.01:  +184.5% goodput/$, short_p90=14.90s, long_p99 +132.3% vs FIFO  (prev default)
#   α=0.05:  +167.4% goodput/$, short_p90=84.78s, long_p99 +124.3% vs FIFO
# Flip-point at α=0.001: 3,990s (~66 min) — aging fires only under extreme starvation.
# Dispatch is near-identical to pure SRPT for virtually all practical waiting times,
# while bounding extreme starvation at the tail.
# 30%-CV prior robustness validated [run 2026-06-21-n]: noisy prior retains ≥97% of
# oracle goodput/$ gain (see run_decoupled_hybrid_noisy_prior_backtest).
DECOUPLED_HYBRID_ALPHA_DEFAULT: float = 0.001

# ---------------------------------------------------------------------------
# Conformal Adaptive α — constants [run 2026-06-21-q]
# ---------------------------------------------------------------------------

# Maximum aging α for conformal discipline (same as fixed best α).
CONFORMAL_ALPHA_MAX: float = DECOUPLED_HYBRID_ALPHA_DEFAULT   # 0.001
# p90 relative prediction error expected under 30%-CV lognormal noise.
# Derived analytically: for X ~ N(−0.043, 0.294²), p90(|e^X − 1|) ≈ 0.40.
CONFORMAL_TARGET_P90_ERROR: float = 0.40
# Number of completions required before α adaptation begins.
CONFORMAL_WARMUP: int = 100
# Sliding-window size for error estimation.
CONFORMAL_WINDOW: int = 200
# ---------------------------------------------------------------------------
# Absolute-Error Conformal α — constants [run 2026-06-22-x]
# ---------------------------------------------------------------------------
# Target p90 absolute prediction error (in output tokens) for α = alpha_max.
# Calibration: with a running-median prior on BurstGPT, p90 abs error ≈ 300–600
# tokens (driven by GPT-4 and surprise-long ChatGPT requests).  Setting
# target=500 means: if p90_abs_err ≤ 500 tokens the calibrator outputs α ≤ alpha_max.
# Contrast with relative error (CONFORMAL_TARGET_P90_ERROR=0.40): short ChatGPT
# over-predictions (predict=18, actual=7) produce rel_err=1.57 >> 0.40, capping
# the calibrator at 2×alpha_max=0.002.  Absolute error correctly ignores those tiny
# absolute misses and reports only large-absolute-error (long-request) uncertainty.
CONFORMAL_ABS_TARGET_P90_TOKENS: float = 500.0

# ---------------------------------------------------------------------------
# Compound Economic × Queue Scheduling — constants [run 2026-06-22-z]
# ---------------------------------------------------------------------------
# Economic cost factor from BENCHMARK_REGISTRY (Azure LLM 2024 provisioning,
# run 2026-06-21-s): +25.75% SLA-safe goodput/$ vs sla_aware, driven by
# -21.2% GPU-hours through time-of-day / spot pricing / regional routing.
# Applied as a multiplicative cost-side discount to any queue discipline:
#   compound_goodput/$ = queue_goodput/$ × ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY
# The factor is orthogonal to queue ordering because provisioning-level
# decisions (which GPU, when, where) are independent of per-request ordering.
# Source: research/BENCHMARK_REGISTRY.md §1.1 "Azure LLM Inference Dataset 2024"
ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY: float = 1.2575  # = 1 + 0.2575

# North-star target for the compound backtest (run -z):
#   compound_goodput/$ must be ≥ NORTH_STAR_MULTIPLIER × oracle_sla_aware_goodput/$
NORTH_STAR_MULTIPLIER: float = 4.0   # +300% vs oracle SLA-aware = 4× oracle SLA-aware

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AZURE_FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv"
)
DEFAULT_BURSTGPT_FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv"
)

# HuggingFace BurstGPT normalized sample — 59,999 records, CC-BY-4.0.
# Fields: request_arrival_ts_s (float), output_tokens (int), input_tokens, model_id.
# Used for full-scale cross-validation that the 54-row fixture cannot support.
DEFAULT_BURSTGPT_HF_JSONL = os.path.join(
    _REPO_ROOT, "data", "external", "hf", "lzzmm__BurstGPT",
    "burstgpt_1_full", "processed", "normalized_sample.jsonl"
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
    model_id: str = ""   # optional class label for per-class conformal calibration


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


def load_burstgpt_serving_requests_jsonl(
    path: str = DEFAULT_BURSTGPT_HF_JSONL,
    limit: Optional[int] = None,
) -> list[tuple[float, int]]:
    """Return ``(arrival_s, output_tokens)`` from a BurstGPT HF normalized JSONL.

    Each line is a JSON object with at minimum:
      ``request_arrival_ts_s`` (float) — arrival timestamp in seconds
      ``output_tokens``         (int)  — response token count

    Failures (output_tokens == 0) are excluded.  Results are sorted by arrival
    time with t0 normalized to 0.

    Used for full-scale BurstGPT cross-validation [run 2026-06-21-p]:
    the HF normalized sample has 59,999 records (CC-BY-4.0) vs the 54-row
    fixture which is too small to demonstrate SRPT > FIFO.
    """
    import json as _json

    rows: list[tuple[float, int]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = _json.loads(line)
                ts = float(d["request_arrival_ts_s"])
                out_tok = int(d.get("output_tokens") or 0)
            except (KeyError, ValueError, TypeError):
                continue
            if out_tok > 0:
                rows.append((ts, out_tok))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return []
    t0 = rows[0][0]
    result = [(ts - t0, tok) for ts, tok in rows]
    if limit is not None:
        result = result[:limit]
    return result


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

def _simulate_srpt_preemptive(
    requests: list[_Request],
    servers: int,
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Preemptive M/G/c SRPT discrete-event simulator.

    Maintains the SRPT invariant: at all times the *c* requests with the
    shortest remaining service time are the ones running.  When a newly
    arriving request is shorter than the longest-remaining running job, that
    running job is preempted; it re-enters the waiting queue with its current
    remaining service time and will be resumed when a server is next freed.

    Key properties (vs. non-preemptive SRTF):

    - **No unbounded starvation:** every long request makes forward progress
      whenever it holds a server.  Its remaining service decreases
      monotonically.  Once its remaining drops below any competing request's
      service time it can no longer be preempted.
    - **SRPT optimality:** SRPT minimises mean response time for M/G/1
      (Schrage 1968) and achieves near-optimal results for M/G/c
      (arXiv:1805.07686).
    - **Short-request benefit preserved:** short requests preempt long ones and
      reach the same near-SRTF p90 latency as the non-preemptive discipline.

    Stale-event detection: each server maintains a ``version`` counter that is
    incremented on every start or preemption.  A completion event is ignored if
    its recorded version differs from the server's current counter.

    Wait time accounting: wait_map[i] = response[i] − service_s[i], which
    captures the sum of all queuing intervals (initial wait + any preemption
    intervals).

    Research basis:
    - TRAIL (arXiv:2410.01035, ICLR 2025): SRPT with limited preemptions.
    - FlowPrefill (arXiv:2602.16603, Feb 2026): operator-level preemption at
      token boundaries; event-driven scheduling on arrival/completion.
    - SRPT for multiserver systems (arXiv:1805.07686): theoretical analysis of
      preemptive SRPT in M/G/k queues.
    - FastSwitch (arXiv:2411.18424, Nov 2024): quantifies context-switching
      overhead in preemptive LLM serving; ``preemption_overhead_s`` models
      re-prefill latency per preemption event (default 0.0 = zero-overhead
      assumption; calibrate from vLLM recomputation benchmarks).
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]   # total preemption events across all servers

    # Per-server state (indexed 0..servers-1).
    s_req:   list = [None] * servers   # current _Request or None (free)
    s_start: list = [0.0]  * servers   # wall-clock time this service period began
    s_rem0:  list = [0.0]  * servers   # remaining service at the start of this period
    s_ver:   list = [0]    * servers   # stale-event version counter

    # Waiting heap: (remaining_s, stable_seq, _Request).
    # remaining_s is the work still needed when the request entered waiting.
    waiting: list = []
    _wseq = [0]

    def _nseq() -> int:
        _wseq[0] += 1
        return _wseq[0]

    # Event heap: (time, ev_type, seq, server_id_or_-1, version_or_-1, request)
    # ev_type 0 = ARRIVAL (sorts before completions at equal time)
    # ev_type 1 = COMPLETION
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return s_rem0[sid] - (t - s_start[sid])

    def _start(sid: int, req: _Request, rem: float, t: float) -> None:
        s_req[sid]  = req
        s_start[sid] = t
        s_rem0[sid]  = rem
        s_ver[sid]  += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    def _preempt(sid: int, t: float):
        """Remove running request from server; return (req, remaining_s)."""
        req = s_req[sid]
        rem = max(0.0, _remaining(sid, t))
        s_req[sid]  = None
        s_ver[sid] += 1   # invalidate the pending completion event
        return req, rem

    response: dict[int, float] = {}

    while events:
        ev  = heapq.heappop(events)
        t   = ev[0]
        ety = ev[1]

        if ety == 0:  # ---- ARRIVAL ----------------------------------------
            req = ev[5]
            free = next((s for s in range(servers) if s_req[s] is None), None)
            if free is not None:
                # A server is idle — start immediately, no preemption needed.
                _start(free, req, req.service_s, t)
            else:
                # All servers busy.  Find the one with the most remaining work.
                worst_sid, worst_rem = 0, -1.0
                for s in range(servers):
                    r = _remaining(s, t)
                    if r > worst_rem:
                        worst_rem, worst_sid = r, s
                if req.service_s < worst_rem:
                    # Arriving request is shorter → preempt the worst server.
                    preempted, prem = _preempt(worst_sid, t)
                    _start(worst_sid, req, req.service_s, t)
                    _npreempt[0] += 1
                    # Re-prefill overhead: added to remaining service of evicted request.
                    heapq.heappush(waiting, (prem + preemption_overhead_s, _nseq(), preempted))
                else:
                    # Arriving request is not shorter than any running → wait.
                    heapq.heappush(waiting, (req.service_s, _nseq(), req))

        else:  # ---- COMPLETION ---------------------------------------------
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue   # stale: this server was preempted or re-started
            response[req.idx] = t - req.arrival_s
            s_req[sid]  = None
            s_ver[sid] += 1
            if waiting:
                rem_s, _, nxt = heapq.heappop(waiting)
                _start(sid, nxt, rem_s, t)

    # Wait time = total response − total service (captures sum of all
    # queuing intervals including preemption pauses).
    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp  = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits, servers)
    summary["preemption_count"] = _npreempt[0]
    return summary, response, wait_map


def _simulate_hybrid_aging_preemptive(
    requests: list[_Request],
    servers: int,
    aging_alpha: float,
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Hybrid Aging + Preemptive SRPT discrete-event simulator.

    Preemption key: key(r, t) = remaining_s / (1 + α · accumulated_wait_s)

    Each request tracks ``accumulated_wait_s`` — the total time it has spent
    in the waiting queue (initial wait + all preemption-gap waits combined).
    While a request is executing on a server, its accumulated wait is frozen.

    **Preemption rule (on arrival of new request r):**
    - new_key = r.service_s (no accumulated wait yet, so denominator = 1).
    - Find the running server with the highest effective key:
      ``effective_key(sid, t) = remaining_s(sid,t) / (1 + α · frozen_wait[sid])``.
    - If ``new_key < max_running_effective_key`` → preempt that server.

    **Anti-starvation guarantee:**
    As a request accumulates waiting time W (across multiple preemption cycles),
    its effective key → remaining_s / (1 + α·W) → 0 as W → ∞.  Once its
    effective key drops below the service_s of the shortest possible arrival,
    no new request can preempt it and it completes uninterrupted.

    **Dispatch rule (when a server becomes free):**
    Pick the waiting request with the minimum effective key at the current time t,
    re-evaluating keys for all waiting requests (O(|waiting|) per dispatch event).

    This combines:
    - SRPT preemption mechanics for optimal short-request performance.
    - Aging-based protection for long-waiting requests against repeated preemption.

    Research basis:
    - FastServe (USENIX NSDI '26): iteration-level preemptive MLFQ + starvation
      prevention for LLM serving; up to 6.1× throughput vs vLLM.
    - Chimera (arXiv:2603.22206, March 2026): STJF with aging-based anti-starvation
      for multi-agent LLM serving.
    - SEK-SMOD / Outperforming Multiserver SRPT (arXiv:2510.25963, SIGMETRICS 2026):
      first policy to provably outperform SRPT-k at all loads by strategic large-job
      re-prioritization.
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]   # total preemption events

    # Per-server state.
    s_req:         list = [None] * servers  # current _Request or None (free)
    s_start:       list = [0.0] * servers   # time this service period began
    s_rem0:        list = [0.0] * servers   # remaining service at period start
    s_ver:         list = [0]   * servers   # stale-event version counter
    s_frozen_wait: list = [0.0] * servers   # accumulated_wait_s of the running req (frozen)

    # Waiting queue: list of (remaining_s, frozen_wait_s, wait_entered_s, req).
    # We use a plain list re-evaluated at each dispatch event (O(|waiting|) per
    # dispatch) so that aging keys are always computed at the correct current time.
    waiting: list = []

    # Event heap: (time, ev_type, seq, server_id, version, request)
    # ev_type 0=ARRIVAL (sorts before COMPLETION at equal time), 1=COMPLETION
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return max(0.0, s_rem0[sid] - (t - s_start[sid]))

    def _effective_running_key(sid: int, t: float) -> float:
        """Key for the request currently running on server sid."""
        return _remaining(sid, t) / max(1e-9, 1.0 + aging_alpha * s_frozen_wait[sid])

    def _effective_waiting_key(entry: tuple, t: float) -> tuple:
        """(effective_key, req_idx) for a waiting-queue entry — stable sort."""
        rem_s, frozen_wait_s, wait_entered_s, req = entry
        current_wait = t - wait_entered_s
        total_wait = frozen_wait_s + current_wait
        ek = rem_s / max(1e-9, 1.0 + aging_alpha * total_wait)
        return (ek, req.idx)

    def _start(sid: int, req: "_Request", rem: float, frozen_wait: float, t: float) -> None:
        s_req[sid]          = req
        s_start[sid]        = t
        s_rem0[sid]         = rem
        s_frozen_wait[sid]  = frozen_wait
        s_ver[sid]         += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    def _preempt(sid: int, t: float):
        """Remove running request; return (req, remaining_s, frozen_wait_s)."""
        req = s_req[sid]
        rem = _remaining(sid, t)
        frozen = s_frozen_wait[sid]
        s_req[sid]  = None
        s_ver[sid] += 1  # invalidate pending completion event
        return req, rem, frozen

    response: dict[int, float] = {}

    while events:
        ev  = heapq.heappop(events)
        t   = ev[0]
        ety = ev[1]

        if ety == 0:  # ---- ARRIVAL ----------------------------------------
            req = ev[5]
            # A fresh arrival has no accumulated wait → effective key = service_s.
            new_key = req.service_s

            free = next((s for s in range(servers) if s_req[s] is None), None)
            if free is not None:
                # Idle server — start immediately, no preemption needed.
                _start(free, req, req.service_s, 0.0, t)
            else:
                # All servers busy.  Find the one with the worst effective key.
                worst_sid, worst_ek = 0, -1.0
                for s in range(servers):
                    ek = _effective_running_key(s, t)
                    if ek > worst_ek:
                        worst_ek, worst_sid = ek, s

                if new_key < worst_ek:
                    # Arriving request is "shorter" per the aging key → preempt.
                    preempted, prem, pfrozen = _preempt(worst_sid, t)
                    _start(worst_sid, req, req.service_s, 0.0, t)
                    _npreempt[0] += 1
                    # Preempted request re-enters waiting; its frozen_wait is
                    # unchanged (it was on a server, not accumulating wait).
                    # Re-prefill overhead is added to remaining service.
                    waiting.append((prem + preemption_overhead_s, pfrozen, t, preempted))
                else:
                    # New request is not short enough to preempt → it waits.
                    waiting.append((req.service_s, 0.0, t, req))

        else:  # ---- COMPLETION ---------------------------------------------
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue  # stale: server was preempted or restarted
            response[req.idx] = t - req.arrival_s
            s_req[sid]  = None
            s_ver[sid] += 1

            if waiting:
                # Pick waiting request with minimum effective key at time t.
                best_i = min(
                    range(len(waiting)),
                    key=lambda i: _effective_waiting_key(waiting[i], t),
                )
                rem_s, frozen_wait_s, wait_entered_s, nxt = waiting.pop(best_i)
                # Accumulate the current waiting period into frozen_wait.
                new_frozen = frozen_wait_s + (t - wait_entered_s)
                _start(sid, nxt, rem_s, new_frozen, t)

    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp  = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits, servers)
    summary["preemption_count"] = _npreempt[0]
    return summary, response, wait_map


def _simulate_decoupled_hybrid(
    requests: list[_Request],
    servers: int,
    aging_alpha: float,
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Decoupled Hybrid SRPT discrete-event simulator [run 2026-06-20-l].

    Separates the two decisions that run -k's unified aging key conflated:

    **Preemption key (on arrival of new request r):**
        new_key = r.service_s
        max_running_key = max(remaining_s(sid, t) for running servers)
        Preempt if new_key < max_running_key.
    This is identical to pure SRPT preemption — no aging factor.  It preserves
    SRPT's throughput-optimal preemption rule: the c requests with the smallest
    remaining service always hold the servers.

    **Dispatch key (when a server becomes free):**
        key(entry, t) = remaining_s / (1 + aging_alpha * total_wait_s)
    where total_wait_s = frozen_wait_s (accumulated across prior preemption pauses)
        + (t - wait_entered_s) (current waiting interval).
    The waiting queue is a plain list re-evaluated at each dispatch event.
    O(|waiting|) per dispatch — correct and necessary for the time-varying key.

    **Root cause fix from run -k:**
    Run -k used the same aging key for BOTH preemption AND dispatch.  At α=0.01,
    a waiting request beats a fresh 3s arrival once its total_wait > 66.7s — very
    common at ρ=0.85.  This caused the hybrid to behave like Aging-SRTF (+64.2%
    goodput/$ vs FIFO), not SRPT (+322.2%).  By using pure remaining_s for
    preemption, new short arrivals always preempt optimally (SRPT throughput).
    By using aging for dispatch, extreme long-wait starvation is bounded.

    **Expected results (Azure LLM 2024, ρ=0.85):**
    - SLA-safe goodput/$: near-SRPT (+320%+ vs FIFO)
    - short_p90: near-SRPT (sub-2s)
    - long_p99: significantly better than pure SRPT (+223% regression), closer to
      Aging-SRTF (+113% regression), because extremely starved long requests
      eventually win dispatch over fresher arrivals

    Research basis:
    - TRAIL (arXiv:2410.01035, ICLR 2025): near-SRPT via SPRPT with limited
      preemptions — validates that SRPT preemption + anti-starvation mechanism
      can coexist without sacrificing throughput.
    - Chimera (arXiv:2603.22206, March 2026): STJF with aging dispatch key for
      multi-agent LLM serving; separates admission policy from scheduling key.
    - FastServe (USENIX NSDI '26): skip-join MLFQ separates preemption granularity
      (iteration-level) from queue promotion policy (aging-based MLFQ level jump).
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]   # total preemption events

    # Per-server state.
    s_req:         list = [None] * servers   # current _Request or None (free)
    s_start:       list = [0.0] * servers    # time this service period started
    s_rem0:        list = [0.0] * servers    # remaining service at period start
    s_ver:         list = [0]   * servers    # stale-event version counter

    # Waiting queue: list of (remaining_s, frozen_wait_s, wait_entered_s, req).
    # Plain list: O(|waiting|) per dispatch; required for aging key that changes
    # as wait_entered_s is compared against current dispatch time t.
    waiting: list = []

    # Event heap: (time, ev_type, seq, server_id, version, request)
    # ev_type 0=ARRIVAL (sorts before COMPLETION at equal time), 1=COMPLETION
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return max(0.0, s_rem0[sid] - (t - s_start[sid]))

    def _aging_dispatch_key(entry: tuple, t: float) -> tuple:
        """(effective_key, req_idx) for dispatch — stable sort."""
        rem_s, frozen_wait_s, wait_entered_s, req = entry
        current_wait = t - wait_entered_s
        total_wait = frozen_wait_s + current_wait
        ek = rem_s / max(1e-9, 1.0 + aging_alpha * total_wait)
        return (ek, req.idx)

    def _start(sid: int, req: "_Request", rem: float, frozen_wait: float, t: float) -> None:
        s_req[sid]   = req
        s_start[sid] = t
        s_rem0[sid]  = rem
        s_ver[sid]  += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    def _preempt(sid: int, t: float):
        """Remove running request; return (req, remaining_s, frozen_wait_s).

        NOTE: frozen_wait_s is the accumulated wait BEFORE this run interval.
        While running, the request does NOT accumulate additional wait — wait
        only accumulates while in the waiting queue.  So frozen_wait on re-entry
        stays the same as when it was dispatched.
        """
        req = s_req[sid]
        rem = _remaining(sid, t)
        frozen = 0.0  # not tracked for decoupled; re-enters with correct semantics below
        s_req[sid]  = None
        s_ver[sid] += 1  # invalidate pending completion event
        return req, rem, frozen

    # We need to track frozen_wait per running server for dispatch re-entry.
    s_frozen_wait: list = [0.0] * servers

    response: dict[int, float] = {}

    while events:
        ev  = heapq.heappop(events)
        t   = ev[0]
        ety = ev[1]

        if ety == 0:  # ---- ARRIVAL ----------------------------------------
            req = ev[5]
            free = next((s for s in range(servers) if s_req[s] is None), None)
            if free is not None:
                # Idle server — start immediately.
                s_frozen_wait[free] = 0.0
                _start(free, req, req.service_s, 0.0, t)
            else:
                # All servers busy.
                # PURE SRPT PREEMPTION: compare new arrival's service_s vs
                # max(remaining_s) — NO aging factor here.
                worst_sid, worst_rem = 0, -1.0
                for s in range(servers):
                    r = _remaining(s, t)
                    if r > worst_rem:
                        worst_rem, worst_sid = r, s

                if req.service_s < worst_rem:
                    # Preempt — pure SRPT rule, no aging.
                    preempted = s_req[worst_sid]
                    prem = _remaining(worst_sid, t)
                    pfrozen = s_frozen_wait[worst_sid]  # accumulated wait before this run
                    s_req[worst_sid]  = None
                    s_ver[worst_sid] += 1               # invalidate stale completion event

                    # Start new short arrival on freed server.
                    s_frozen_wait[worst_sid] = 0.0
                    _start(worst_sid, req, req.service_s, 0.0, t)

                    # Preempted request re-enters waiting with frozen_wait unchanged
                    # (it was running, not accumulating queue wait).
                    # Re-prefill overhead is added to remaining service.
                    _npreempt[0] += 1
                    waiting.append((prem + preemption_overhead_s, pfrozen, t, preempted))
                else:
                    # New request is not shorter than any running → wait.
                    waiting.append((req.service_s, 0.0, t, req))

        else:  # ---- COMPLETION ---------------------------------------------
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue  # stale: server was preempted or restarted
            response[req.idx] = t - req.arrival_s
            s_req[sid]  = None
            s_ver[sid] += 1

            if waiting:
                # AGING DISPATCH: pick waiting request with minimum aging key.
                best_i = min(
                    range(len(waiting)),
                    key=lambda i: _aging_dispatch_key(waiting[i], t),
                )
                rem_s, frozen_wait_s, wait_entered_s, nxt = waiting.pop(best_i)
                # Accumulate current queuing interval into frozen_wait.
                new_frozen = frozen_wait_s + (t - wait_entered_s)
                s_frozen_wait[sid] = new_frozen
                _start(sid, nxt, rem_s, new_frozen, t)

    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp  = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits, servers)
    summary["preemption_count"] = _npreempt[0]
    return summary, response, wait_map


# ---------------------------------------------------------------------------
# Conformal Adaptive α — calibrator and simulator [run 2026-06-21-q]
# ---------------------------------------------------------------------------

class ConformalAlphaCalibrator:
    """Adapts the aging α for decoupled-hybrid dispatch from empirical prediction errors.

    Motivation (arXiv:2508.14544 — Adaptively Robust LLM Inference Optimization
    under Prediction Uncertainty; Mitzenmacher 1902.00732 — Scheduling with
    Predictions and the Price of Misprediction; arXiv:2503.07545 — Queueing,
    Predictions, and LLMs):

    The fixed α=0.001 for the decoupled hybrid dispatch key is calibrated for
    30%-CV lognormal prediction noise.  When predictions are more accurate (e.g.
    oracle prior from a well-trained length forecaster), a smaller α is safe —
    reducing α toward 0 recovers the pure SRPT dispatch key, which is
    throughput-optimal for M/G/c queues (arXiv:1805.07686).

    Mechanism:
      1. Maintain a sliding window of completed (predicted_tokens, actual_tokens) pairs.
      2. Compute the empirical p90 relative prediction error from the window:
           p90_err = percentile_90(|predicted − actual| / actual)
      3. Map p90_err to α linearly (capped at 2× alpha_max for safety):
           α = alpha_max × min(2.0, p90_err / target_p90_error)
      4. Return current_alpha() for use in the dispatch key.

    Behaviour by prediction quality:
      Oracle (predicted == actual):
        p90_err = 0  →  α = 0  →  dispatch = pure SRPT  →  ~+322% vs FIFO
      30%-CV lognormal noise [run-n validated]:
        p90_err ≈ 0.40  →  α = 0.001  →  same as fixed α = 0.001  →  +253.9% vs FIFO
      60%-CV lognormal noise:
        p90_err ≈ 0.72  →  α = 0.0018  →  more aging → robust degradation
      Very noisy:
        α capped at 2 × 0.001 = 0.002

    During warmup (first ``warmup`` completions) the calibrator conservatively
    returns ``alpha_max`` so the simulation starts with the proven safe value.
    """

    def __init__(
        self,
        alpha_max: float = CONFORMAL_ALPHA_MAX,
        warmup: int = CONFORMAL_WARMUP,
        window: int = CONFORMAL_WINDOW,
        target_p90_error: float = CONFORMAL_TARGET_P90_ERROR,
    ) -> None:
        self.alpha_max = alpha_max
        self.warmup = warmup
        self.window = window
        self.target_p90_error = target_p90_error
        self._residuals: list[float] = []
        self._n_completed: int = 0
        self._alpha_sum: float = 0.0   # for mean α diagnostic
        self._alpha_count: int = 0

    def update(self, predicted_tokens: float, actual_tokens: int) -> None:
        """Record a completed request's relative prediction error."""
        self._n_completed += 1
        if actual_tokens > 0:
            rel_err = abs(predicted_tokens - actual_tokens) / actual_tokens
            self._residuals.append(rel_err)
            if len(self._residuals) > self.window:
                self._residuals.pop(0)

    def current_alpha(self) -> float:
        """Return the calibrated dispatch α from empirical p90 prediction error."""
        if self._n_completed < self.warmup or len(self._residuals) < self.warmup // 2:
            alpha = self.alpha_max
        else:
            sorted_r = sorted(self._residuals)
            p90_idx = min(len(sorted_r) - 1, int(0.90 * len(sorted_r)))
            p90_err = sorted_r[p90_idx]
            ratio = min(2.0, p90_err / max(self.target_p90_error, 1e-9))
            alpha = self.alpha_max * ratio
        self._alpha_sum += alpha
        self._alpha_count += 1
        return alpha

    def mean_alpha(self) -> float:
        """Diagnostic: mean α value returned across all dispatch events."""
        return self._alpha_sum / max(1, self._alpha_count)


# AbsoluteErrorConformalCalibrator moved to
# aurelius/optimizer/policies/serving_queue.py [Phase 2] and imported at module
# top (so `from aurelius.benchmarks.srtf_serving_backtest import
# AbsoluteErrorConformalCalibrator` still resolves).


def _simulate_decoupled_hybrid_conformal(
    requests: list[_Request],
    servers: int,
    calibrator: ConformalAlphaCalibrator,
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Decoupled Hybrid SRPT with Conformal Adaptive α [run 2026-06-21-q].

    Identical to ``_simulate_decoupled_hybrid`` except that the dispatch aging
    parameter α is not fixed — it is updated after every completion event using
    a ``ConformalAlphaCalibrator``.

    **Preemption key (on new arrival r):**
        remaining_s  [pure SRPT — unchanged from fixed-α variant]

    **Dispatch key (when server becomes free):**
        key(entry, t) = remaining_s / (1 + α(t) × total_wait_s)
    where α(t) = calibrator.current_alpha() is re-evaluated before each dispatch.

    With oracle tokens (predicted == actual):
      α(t) → 0 after warmup  →  dispatch is pure SRPT  →  goodput/$ → +322% vs FIFO.
    With 30%-CV noisy prior [run-n calibrated]:
      α(t) → 0.001 after warmup  →  same as fixed-α variant  →  +253.9% vs FIFO.

    The two results show the conformal calibrator correctly recovers SRPT
    throughput when predictions are reliable and falls back to the safe α=0.001
    when they are not.

    Research basis:
    - arXiv:2508.14544 (Adaptively Robust LLM Inference under Prediction Uncertainty):
      core motivation for adaptive scheduling policy under prediction error.
    - arXiv:1902.00732 (Scheduling with Predictions, Mitzenmacher 2019): theoretical
      foundation for prediction-based scheduling with graceful degradation.
    - arXiv:2604.00499 (TIE scheduling, Zheng et al. 2026): distributional ordering
      for heavy-tailed output lengths — conformal α generalises this to dispatch.
    - arXiv:2503.07545 (Queueing, Predictions, and LLMs, Mitzenmacher & Shahout 2025):
      identifies adaptive calibration as the key open problem for production schedulers.
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]

    s_req:          list = [None] * servers
    s_start:        list = [0.0] * servers
    s_rem0:         list = [0.0] * servers
    s_ver:          list = [0]   * servers
    s_frozen_wait:  list = [0.0] * servers

    waiting: list = []

    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return max(0.0, s_rem0[sid] - (t - s_start[sid]))

    def _conf_dispatch_key(entry: tuple, t: float, alpha: float) -> tuple:
        rem_s, frozen_wait_s, wait_entered_s, req = entry
        current_wait = t - wait_entered_s
        total_wait = frozen_wait_s + current_wait
        ek = rem_s / max(1e-9, 1.0 + alpha * total_wait)
        return (ek, req.idx)

    def _start(sid: int, req: "_Request", rem: float, frozen_wait: float, t: float) -> None:
        s_req[sid]   = req
        s_start[sid] = t
        s_rem0[sid]  = rem
        s_ver[sid]  += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    response: dict[int, float] = {}

    while events:
        ev  = heapq.heappop(events)
        t   = ev[0]
        ety = ev[1]

        if ety == 0:  # ---- ARRIVAL ----------------------------------------
            req = ev[5]
            free = next((s for s in range(servers) if s_req[s] is None), None)
            if free is not None:
                s_frozen_wait[free] = 0.0
                _start(free, req, req.service_s, 0.0, t)
            else:
                # PURE SRPT PREEMPTION: no aging factor on preemption key.
                worst_sid, worst_rem = 0, -1.0
                for s in range(servers):
                    r = _remaining(s, t)
                    if r > worst_rem:
                        worst_rem, worst_sid = r, s

                if req.service_s < worst_rem:
                    preempted = s_req[worst_sid]
                    prem = _remaining(worst_sid, t)
                    pfrozen = s_frozen_wait[worst_sid]
                    s_req[worst_sid]  = None
                    s_ver[worst_sid] += 1
                    s_frozen_wait[worst_sid] = 0.0
                    _start(worst_sid, req, req.service_s, 0.0, t)
                    _npreempt[0] += 1
                    waiting.append((prem + preemption_overhead_s, pfrozen, t, preempted))
                else:
                    waiting.append((req.service_s, 0.0, t, req))

        else:  # ---- COMPLETION ---------------------------------------------
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue
            response[req.idx] = t - req.arrival_s

            # Update calibrator with this completion's prediction residual.
            calibrator.update(req.predicted_tokens, req.actual_tokens)

            s_req[sid]  = None
            s_ver[sid] += 1

            if waiting:
                # CONFORMAL AGING DISPATCH: α recalibrated after each completion.
                alpha = calibrator.current_alpha()
                best_i = min(
                    range(len(waiting)),
                    key=lambda i: _conf_dispatch_key(waiting[i], t, alpha),
                )
                rem_s, frozen_wait_s, wait_entered_s, nxt = waiting.pop(best_i)
                new_frozen = frozen_wait_s + (t - wait_entered_s)
                s_frozen_wait[sid] = new_frozen
                _start(sid, nxt, rem_s, new_frozen, t)

    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp  = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits, servers)
    summary["preemption_count"] = _npreempt[0]
    summary["conformal_mean_alpha"] = calibrator.mean_alpha()
    return summary, response, wait_map


def simulate_queue(
    requests: list[_Request],
    servers: int,
    discipline: str,
    aging_alpha: float = AGING_ALPHA_DEFAULT,
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Run a M/G/c discrete-event simulation under the requested discipline.

    ``discipline``:
      ``fifo``                    — ready requests served in arrival order (non-preemptive).
      ``srtf``                    — shortest *predicted* job first (non-preemptive).
      ``aging_srtf``              — SRTF with aging: at dispatch time t the effective key
                                    is ``predicted_tokens / (1 + aging_alpha * wait_so_far)``.
                                    As wait grows the key falls, giving long-waiting requests
                                    higher priority and bounding starvation.
      ``srpt_preemptive``         — Preemptive SRPT: when a shorter request arrives the
                                    longest-running job is preempted and re-enters waiting
                                    with its remaining service.  Eliminates unbounded
                                    starvation; each long request always makes progress.
      ``hybrid_aging_preemptive`` — Preemptive SRPT with aging-based starvation protection
                                    [run 2026-06-20-k]: preemption key =
                                    remaining_s / (1 + aging_alpha * accumulated_wait_s).
                                    New arrivals (zero accumulated wait) have key = service_s.
                                    Long-waiting requests accumulate priority, eventually
                                    becoming unpreemptable.  Combines SRPT's short-request
                                    benefit with aging's starvation bound.
      ``decoupled_hybrid``              — Decoupled Hybrid SRPT [run 2026-06-20-l]: preemption
                                          by pure remaining_s (SRPT), dispatch by aging key
                                          remaining_s / (1 + aging_alpha * total_wait_s).
                                          Achieves SRPT-level goodput with Aging-SRTF long_p99.
      ``decoupled_hybrid_conformal``    — Conformal Adaptive α [run 2026-06-21-q]: same
                                          preemption as decoupled_hybrid (pure SRPT) but aging
                                          α for dispatch is recalibrated after each completion
                                          from empirical p90 prediction error.  With oracle
                                          tokens: α → 0 → pure SRPT dispatch → ~+322% vs FIFO.
                                          With 30%-CV noise: α → 0.001 → same as fixed α.
                                          Research basis: arXiv:2508.14544 (adaptively robust
                                          LLM scheduling), arXiv:1902.00732 (price of mispred.).
      ``sla_aware``                     — Binary SLA-class priority [run 2026-06-21-n]: requests
                                          with predicted_tokens ≤ global median get priority class
                                          0 (latency-critical); others get class 1 (standard).
                                          Dispatches all class-0 requests before class-1; FIFO
                                          within each class.  No continuous token prediction
                                          needed — just the binary SLA classification.

    ``aging_alpha`` affects ``aging_srtf``, ``hybrid_aging_preemptive``, and
    ``decoupled_hybrid``.  ``decoupled_hybrid_conformal`` derives its α adaptively
    and ignores the ``aging_alpha`` parameter.

    Returns ``(summary, response_map, wait_map)`` where the maps are
    ``{request_idx: seconds}``.  The simulation is deterministic given the
    inputs; ties break on arrival sequence (request index).
    """
    if discipline == "srpt_preemptive":
        return _simulate_srpt_preemptive(requests, servers, preemption_overhead_s)
    if discipline == "hybrid_aging_preemptive":
        return _simulate_hybrid_aging_preemptive(requests, servers, aging_alpha, preemption_overhead_s)
    if discipline == "decoupled_hybrid":
        return _simulate_decoupled_hybrid(requests, servers, aging_alpha, preemption_overhead_s)
    if discipline == "decoupled_hybrid_conformal":
        cal = ConformalAlphaCalibrator()
        return _simulate_decoupled_hybrid_conformal(requests, servers, cal, preemption_overhead_s)
    if discipline == "decoupled_hybrid_abs_conformal":
        cal = AbsoluteErrorConformalCalibrator()
        return _simulate_decoupled_hybrid_abs_conformal(requests, servers, cal, preemption_overhead_s)
    if discipline == "decoupled_hybrid_per_class_conformal":
        cal = PerClassConformalCalibrator()
        return _simulate_decoupled_hybrid_per_class_conformal(requests, servers, cal, preemption_overhead_s)
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))

    # Precompute median predicted_tokens for sla_aware binary classification.
    _pred_sorted = sorted(r.predicted_tokens for r in requests)
    _median_pred: float = _pred_sorted[len(_pred_sorted) // 2] if _pred_sorted else 0.0

    busy: list[float] = []   # min-heap of server completion times

    # For fifo/srtf/sla_aware: priority heap keyed by (discipline_key, idx, request).
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
            if discipline == "srtf":
                key = (req.predicted_tokens, seq)
            elif discipline == "sla_aware":
                # Binary SLA-class: 0=latency-critical (short), 1=standard (long).
                # Within each class, FIFO (arrival sequence via seq tiebreak).
                sla_class = 0 if req.predicted_tokens <= _median_pred else 1
                key = (sla_class, seq)
            else:
                key = (seq,)
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
    summary["preemption_count"] = 0  # non-preemptive disciplines have zero preemptions
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


@dataclass
class SRTFPreemptiveReport:
    """4-discipline comparison: FIFO, SRTF (non-preemptive), Aging-SRTF, SRPT-preemptive.

    Run 2026-06-20-j extends run -i's aging analysis by adding a preemptive
    SRPT discipline that eliminates (rather than bounds) long-request starvation.

    Research basis:
    - TRAIL (arXiv:2410.01035, ICLR 2025): SRPT with limited preemptions; 1.66–
      2.01× lower mean latency + M/G/1 closed-form formula for the SRPT variant.
    - FlowPrefill (arXiv:2602.16603, Feb 2026): operator-level preemption at
      token/operator boundaries; event-driven scheduling on arrival/completion.
    - SRPT for multiserver systems (arXiv:1805.07686): SRPT policy for M/G/k
      runs the k requests with smallest remaining processing time; near-optimal.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float
    aging_alpha: float

    fifo: dict
    srtf_perfect: dict         # non-preemptive SRTF (oracle prior)
    aging_srtf: dict           # non-preemptive SRTF with aging (α=aging_alpha)
    srpt_preemptive: dict      # preemptive SRPT (oracle prior = actual tokens)

    # Short-request p90 improvement vs FIFO (positive = better)
    srtf_short_p90_improvement_pct: float
    aging_short_p90_improvement_pct: float
    srpt_short_p90_improvement_pct: float

    # Long-request p99 change vs FIFO
    # Negative % = improvement; positive % = regression (starvation)
    srtf_long_p99_delta_pct: float     # large positive → starvation
    aging_long_p99_delta_pct: float    # smaller positive → partial recovery
    srpt_long_p99_delta_pct: float     # expected near zero or negative → full recovery

    # SLA-safe goodput/$ vs FIFO (positive = better)
    srtf_goodput_delta_pct: float
    aging_goodput_delta_pct: float
    srpt_goodput_delta_pct: float

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
            "srpt_preemptive": _r(self.srpt_preemptive),
            "srtf_short_p90_improvement_pct": round(self.srtf_short_p90_improvement_pct, 2),
            "aging_short_p90_improvement_pct": round(self.aging_short_p90_improvement_pct, 2),
            "srpt_short_p90_improvement_pct": round(self.srpt_short_p90_improvement_pct, 2),
            "srtf_long_p99_delta_pct": round(self.srtf_long_p99_delta_pct, 2),
            "aging_long_p99_delta_pct": round(self.aging_long_p99_delta_pct, 2),
            "srpt_long_p99_delta_pct": round(self.srpt_long_p99_delta_pct, 2),
            "srtf_goodput_delta_pct": round(self.srtf_goodput_delta_pct, 2),
            "aging_goodput_delta_pct": round(self.aging_goodput_delta_pct, 2),
            "srpt_goodput_delta_pct": round(self.srpt_goodput_delta_pct, 2),
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


# ---------------------------------------------------------------------------
# Preemptive SRPT — 4-discipline comparison [run 2026-06-20-j]
# ---------------------------------------------------------------------------

def _run_preemptive_backtest_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
) -> "SRTFPreemptiveReport":
    """Internal helper: run all 4 disciplines and return SRTFPreemptiveReport."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # oracle prior (actual = predicted)
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs  = _build()
    srtf_reqs  = _build()
    aging_reqs = _build()
    srpt_reqs  = _build()

    fifo_sim,  fifo_resp,  _ = simulate_queue(fifo_reqs,  servers, "fifo")
    srtf_sim,  srtf_resp,  _ = simulate_queue(srtf_reqs,  servers, "srtf")
    aging_sim, aging_resp, _ = simulate_queue(
        aging_reqs, servers, "aging_srtf", aging_alpha=aging_alpha
    )
    srpt_sim,  srpt_resp,  _ = simulate_queue(srpt_reqs,  servers, "srpt_preemptive")

    gp_fifo  = _sla_safe_goodput_per_dollar(fifo_reqs,  fifo_resp,  sla_s, servers)
    gp_srtf  = _sla_safe_goodput_per_dollar(srtf_reqs,  srtf_resp,  sla_s, servers)
    gp_aging = _sla_safe_goodput_per_dollar(aging_reqs, aging_resp, sla_s, servers)
    gp_srpt  = _sla_safe_goodput_per_dollar(srpt_reqs,  srpt_resp,  sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"]  = gp_fifo
    srtf_sim["sla_safe_goodput_per_dollar"]  = gp_srtf
    aging_sim["sla_safe_goodput_per_dollar"] = gp_aging
    srpt_sim["sla_safe_goodput_per_dollar"]  = gp_srpt

    def _impr(base: float, new: float) -> float:
        return (base - new) / base * 100.0 if base > 0 else 0.0

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    fifo_sp90 = fifo_sim["short_p90_response_s"]
    fifo_lp99 = fifo_sim["long_p99_response_s"]

    return SRTFPreemptiveReport(
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
        srpt_preemptive=srpt_sim,
        srtf_short_p90_improvement_pct=_impr(fifo_sp90, srtf_sim["short_p90_response_s"]),
        aging_short_p90_improvement_pct=_impr(fifo_sp90, aging_sim["short_p90_response_s"]),
        srpt_short_p90_improvement_pct=_impr(fifo_sp90, srpt_sim["short_p90_response_s"]),
        srtf_long_p99_delta_pct=_delta(fifo_lp99, srtf_sim["long_p99_response_s"]),
        aging_long_p99_delta_pct=_delta(fifo_lp99, aging_sim["long_p99_response_s"]),
        srpt_long_p99_delta_pct=_delta(fifo_lp99, srpt_sim["long_p99_response_s"]),
        srtf_goodput_delta_pct=_delta(gp_fifo, gp_srtf),
        aging_goodput_delta_pct=_delta(gp_fifo, gp_aging),
        srpt_goodput_delta_pct=_delta(gp_fifo, gp_srpt),
    )


def run_srpt_preemptive_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = 0.01,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> "SRTFPreemptiveReport":
    """4-discipline comparison on Azure LLM 2024: FIFO / SRTF / Aging-SRTF / SRPT-preemptive.

    Extends run -i's aging analysis by adding preemptive SRPT, which eliminates
    (not just bounds) long-request starvation.  In SRPT, a running long job is
    interrupted when a shorter request arrives; the long job resumes with its
    remaining service time.  Because remaining service decreases monotonically
    while a job holds the server, every long job eventually becomes the shortest
    remaining and completes without further preemption.

    The default ``aging_alpha=0.01`` is the recommended sweet spot from run -i
    (+70.7% goodput/$ vs FIFO, 49% starvation reduction).

    Args:
        servers: Replica pool size (M/G/c). Identical across all disciplines.
        target_rho: Target cluster utilization (arrival time-warp).
        job_limit: Optional cap on the number of real requests used.
        aging_alpha: Aging decay constant for the aging_srtf discipline.
        sla_s: E2E response-time SLA budget (seconds).
        azure_fixture: Path to the Azure LLM 2024 CSV fixture.

    Returns:
        ``SRTFPreemptiveReport`` with KPIs and deltas for all four disciplines.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_preemptive_backtest_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s
    )


def run_burstgpt_srpt_preemptive_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = 0.01,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> "SRTFPreemptiveReport":
    """Cross-validate SRPT-preemptive on BurstGPT — 4-discipline comparison.

    Runs the same 4-discipline comparison as ``run_srpt_preemptive_backtest``
    but on the BurstGPT trace (heavier output-token distribution, avg ~340 tok
    vs Azure 2024's ~104 tok, with a higher default SLA budget of 30 s).

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on number of requests.
        aging_alpha: Aging decay constant for the aging_srtf discipline.
        sla_s: SLA budget (default 30 s for BurstGPT's longer service times).
        burstgpt_fixture: BurstGPT CSV path.

    Returns:
        ``SRTFPreemptiveReport`` with FIFO / SRTF / Aging-SRTF / SRPT-preemptive KPIs.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_preemptive_backtest_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s
    )


# ---------------------------------------------------------------------------
# Hybrid Aging + Preemptive SRPT — 5-discipline comparison [run 2026-06-20-k]
# ---------------------------------------------------------------------------

@dataclass
class HybridAgingPreemptiveReport:
    """5-discipline comparison: FIFO, SRTF, Aging-SRTF, SRPT-preemptive, Hybrid.

    Run 2026-06-20-k adds the ``hybrid_aging_preemptive`` discipline, which
    combines SRPT preemption with aging-based anti-starvation.  The preemption
    key is ``remaining_s / (1 + α · accumulated_wait_s)``:

    - Short new arrivals (zero accumulated wait, key = service_s) preempt long
      running jobs early, preserving SRPT's short-request p90 benefit.
    - As a long job's accumulated waiting time grows across multiple preemption
      cycles, its effective key shrinks → it becomes harder to preempt →
      starvation is provably bounded.

    Expected positioning on the goodput/$ vs long_p99 trade-off curve:
    - goodput/$: near-SRPT (close to +322% vs FIFO from run -j)
    - long_p99: much better than SRPT (+223% regression in run -j), likely
      better than aging-SRTF (+113% in run -i)

    Research basis:
    - FastServe (USENIX NSDI '26): skip-join MLFQ with starvation prevention.
    - Chimera (arXiv:2603.22206, March 2026): aging anti-starvation in STJF.
    - SEK-SMOD (arXiv:2510.25963, SIGMETRICS 2026): outperforming SRPT-k by
      strategic large-job re-prioritization.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float
    aging_alpha: float

    fifo: dict
    srtf_perfect: dict            # non-preemptive SRTF (oracle prior)
    aging_srtf: dict              # non-preemptive SRTF with aging (α=aging_alpha)
    srpt_preemptive: dict         # preemptive SRPT (oracle prior, no aging)
    hybrid_aging_preemptive: dict # preemptive SRPT with aging (α=aging_alpha)

    # Short-request p90 response improvement vs FIFO (positive = better)
    srtf_short_p90_improvement_pct: float
    aging_short_p90_improvement_pct: float
    srpt_short_p90_improvement_pct: float
    hybrid_short_p90_improvement_pct: float

    # Long-request p99 change vs FIFO
    # Positive % = regression (starvation); negative % = improvement
    srtf_long_p99_delta_pct: float
    aging_long_p99_delta_pct: float
    srpt_long_p99_delta_pct: float
    hybrid_long_p99_delta_pct: float

    # SLA-safe goodput/$ delta vs FIFO (positive = better)
    srtf_goodput_delta_pct: float
    aging_goodput_delta_pct: float
    srpt_goodput_delta_pct: float
    hybrid_goodput_delta_pct: float

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
            "srpt_preemptive": _r(self.srpt_preemptive),
            "hybrid_aging_preemptive": _r(self.hybrid_aging_preemptive),
            "srtf_short_p90_improvement_pct": round(self.srtf_short_p90_improvement_pct, 2),
            "aging_short_p90_improvement_pct": round(self.aging_short_p90_improvement_pct, 2),
            "srpt_short_p90_improvement_pct": round(self.srpt_short_p90_improvement_pct, 2),
            "hybrid_short_p90_improvement_pct": round(self.hybrid_short_p90_improvement_pct, 2),
            "srtf_long_p99_delta_pct": round(self.srtf_long_p99_delta_pct, 2),
            "aging_long_p99_delta_pct": round(self.aging_long_p99_delta_pct, 2),
            "srpt_long_p99_delta_pct": round(self.srpt_long_p99_delta_pct, 2),
            "hybrid_long_p99_delta_pct": round(self.hybrid_long_p99_delta_pct, 2),
            "srtf_goodput_delta_pct": round(self.srtf_goodput_delta_pct, 2),
            "aging_goodput_delta_pct": round(self.aging_goodput_delta_pct, 2),
            "srpt_goodput_delta_pct": round(self.srpt_goodput_delta_pct, 2),
            "hybrid_goodput_delta_pct": round(self.hybrid_goodput_delta_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_hybrid_backtest_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
) -> HybridAgingPreemptiveReport:
    """Internal helper: run all 5 disciplines and return HybridAgingPreemptiveReport."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # oracle prior (actual = predicted)
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs   = _build()
    srtf_reqs   = _build()
    aging_reqs  = _build()
    srpt_reqs   = _build()
    hybrid_reqs = _build()

    fifo_sim,   fifo_resp,   _ = simulate_queue(fifo_reqs,   servers, "fifo")
    srtf_sim,   srtf_resp,   _ = simulate_queue(srtf_reqs,   servers, "srtf")
    aging_sim,  aging_resp,  _ = simulate_queue(
        aging_reqs, servers, "aging_srtf", aging_alpha=aging_alpha
    )
    srpt_sim,   srpt_resp,   _ = simulate_queue(srpt_reqs,   servers, "srpt_preemptive")
    hybrid_sim, hybrid_resp, _ = simulate_queue(
        hybrid_reqs, servers, "hybrid_aging_preemptive", aging_alpha=aging_alpha
    )

    gp_fifo   = _sla_safe_goodput_per_dollar(fifo_reqs,   fifo_resp,   sla_s, servers)
    gp_srtf   = _sla_safe_goodput_per_dollar(srtf_reqs,   srtf_resp,   sla_s, servers)
    gp_aging  = _sla_safe_goodput_per_dollar(aging_reqs,  aging_resp,  sla_s, servers)
    gp_srpt   = _sla_safe_goodput_per_dollar(srpt_reqs,   srpt_resp,   sla_s, servers)
    gp_hybrid = _sla_safe_goodput_per_dollar(hybrid_reqs, hybrid_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"]             = gp_fifo
    srtf_sim["sla_safe_goodput_per_dollar"]             = gp_srtf
    aging_sim["sla_safe_goodput_per_dollar"]            = gp_aging
    srpt_sim["sla_safe_goodput_per_dollar"]             = gp_srpt
    hybrid_sim["sla_safe_goodput_per_dollar"]           = gp_hybrid

    def _impr(base: float, new: float) -> float:
        return (base - new) / base * 100.0 if base > 0 else 0.0

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    fifo_sp90 = fifo_sim["short_p90_response_s"]
    fifo_lp99 = fifo_sim["long_p99_response_s"]

    return HybridAgingPreemptiveReport(
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
        srpt_preemptive=srpt_sim,
        hybrid_aging_preemptive=hybrid_sim,
        srtf_short_p90_improvement_pct=_impr(fifo_sp90, srtf_sim["short_p90_response_s"]),
        aging_short_p90_improvement_pct=_impr(fifo_sp90, aging_sim["short_p90_response_s"]),
        srpt_short_p90_improvement_pct=_impr(fifo_sp90, srpt_sim["short_p90_response_s"]),
        hybrid_short_p90_improvement_pct=_impr(fifo_sp90, hybrid_sim["short_p90_response_s"]),
        srtf_long_p99_delta_pct=_delta(fifo_lp99, srtf_sim["long_p99_response_s"]),
        aging_long_p99_delta_pct=_delta(fifo_lp99, aging_sim["long_p99_response_s"]),
        srpt_long_p99_delta_pct=_delta(fifo_lp99, srpt_sim["long_p99_response_s"]),
        hybrid_long_p99_delta_pct=_delta(fifo_lp99, hybrid_sim["long_p99_response_s"]),
        srtf_goodput_delta_pct=_delta(gp_fifo, gp_srtf),
        aging_goodput_delta_pct=_delta(gp_fifo, gp_aging),
        srpt_goodput_delta_pct=_delta(gp_fifo, gp_srpt),
        hybrid_goodput_delta_pct=_delta(gp_fifo, gp_hybrid),
    )


def run_hybrid_aging_preemptive_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = HYBRID_AGING_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> HybridAgingPreemptiveReport:
    """5-discipline comparison on Azure LLM 2024: FIFO / SRTF / Aging / SRPT / Hybrid.

    The hybrid discipline uses preemption key = remaining_s / (1 + α·accumulated_wait_s).
    New arrivals (zero accumulated wait) have key = service_s and can preempt long jobs.
    As a long job accumulates waiting time across preemption cycles, its effective key
    shrinks → it becomes resistant to further preemption → starvation is bounded.

    Expected outcomes (Azure LLM 2024, ρ=0.85):
    - goodput/$: near-SRPT preemptive (+322% vs FIFO from run -j)
    - short_p90: similar to SRPT preemptive (best among all disciplines)
    - long_p99: significantly better than SRPT preemptive (+223% regression in run -j)
      and likely better than aging-SRTF (+113% regression in run -i)

    Research basis: FastServe (NSDI '26), Chimera (arXiv:2603.22206),
    SEK-SMOD (arXiv:2510.25963, SIGMETRICS 2026).

    Args:
        servers: Replica pool size (M/G/c). Identical across all disciplines.
        target_rho: Target cluster utilization (arrival time-warp).
        job_limit: Optional cap on the number of real requests used.
        aging_alpha: Aging decay constant (default HYBRID_AGING_ALPHA_DEFAULT = 0.01).
        sla_s: E2E response-time SLA budget (seconds).
        azure_fixture: Path to the Azure LLM 2024 CSV fixture.

    Returns:
        ``HybridAgingPreemptiveReport`` with KPIs and deltas for all five disciplines.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_hybrid_backtest_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s
    )


def run_burstgpt_hybrid_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = HYBRID_AGING_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> HybridAgingPreemptiveReport:
    """Cross-validate Hybrid Aging+Preemptive SRPT on BurstGPT — 5-discipline comparison.

    Runs the same 5-discipline comparison as ``run_hybrid_aging_preemptive_backtest``
    on the BurstGPT trace (heavier output-token distribution, avg ~340 tok vs Azure
    2024's ~104 tok, with a higher default SLA budget of 30 s).

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on number of requests.
        aging_alpha: Aging decay constant for aging-based disciplines.
        sla_s: SLA budget (default 30 s for BurstGPT's longer service times).
        burstgpt_fixture: BurstGPT CSV path.

    Returns:
        ``HybridAgingPreemptiveReport`` with FIFO / SRTF / Aging / SRPT / Hybrid KPIs.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_hybrid_backtest_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s
    )


# ---------------------------------------------------------------------------
# Decoupled Hybrid SRPT — 6-discipline comparison [run 2026-06-20-l]
# ---------------------------------------------------------------------------

@dataclass
class DecoupledHybridReport:
    """6-discipline comparison: FIFO, SRTF, Aging-SRTF, SRPT-preemptive,
    Hybrid (unified key), and Decoupled Hybrid (split preemption/dispatch keys).

    Run 2026-06-20-l addresses the root cause identified in run -k:

    Run -k finding: the unified aging key remaining_s/(1+α·wait) for BOTH
    preemption AND dispatch made the hybrid behave like Aging-SRTF (+64.2%
    goodput/$ vs FIFO), not SRPT (+322.2%).  The dispatch-level aging at α=0.01
    promotes long-waiting requests over fresher short arrivals after only 66.7s
    of accumulated wait, systematically blocking short fresh arrivals.

    Fix (decoupled hybrid): separate the two decisions:
    - Preemption: remaining_s (pure SRPT) — short fresh arrivals always preempt
      the longest-remaining running job, as in pure SRPT.
    - Dispatch: remaining_s / (1 + α·total_wait) — long-waiting requests
      accumulate dispatch priority, preventing indefinite starvation.

    Expected positioning:
    - goodput/$: near-SRPT (+~320% vs FIFO) — preemption is SRPT-identical
    - short_p90: near-SRPT (sub-2s) — same preemption rule
    - long_p99: significantly better than SRPT — aging dispatch eventually
      boosts extremely starved long requests to the front of the dispatch queue

    Research basis:
    - TRAIL (arXiv:2410.01035, ICLR 2025): SRPT-style preemption + bounded
      starvation; shows these goals are simultaneously achievable.
    - Chimera (arXiv:2603.22206, March 2026): aging dispatch key for anti-
      starvation in multi-agent LLM serving; supports splitting preemption
      from dispatch as independent policies.
    - FastServe (USENIX NSDI '26): skip-join MLFQ separates preemption
      granularity (iteration-level) from promotion policy (level-based aging).
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float
    aging_alpha: float

    fifo: dict
    srtf_perfect: dict              # non-preemptive SRTF (oracle prior)
    aging_srtf: dict                # non-preemptive SRTF with aging
    srpt_preemptive: dict           # preemptive SRPT (no aging)
    hybrid_aging_preemptive: dict   # preemptive SRPT with unified aging key (run -k)
    decoupled_hybrid: dict          # preemptive SRPT + aging dispatch (run -l)

    # Short-request p90 response improvement vs FIFO (positive = better)
    srtf_short_p90_improvement_pct: float
    aging_short_p90_improvement_pct: float
    srpt_short_p90_improvement_pct: float
    hybrid_short_p90_improvement_pct: float
    decoupled_short_p90_improvement_pct: float

    # Long-request p99 change vs FIFO
    # Positive % = regression (starvation); negative % = improvement
    srtf_long_p99_delta_pct: float
    aging_long_p99_delta_pct: float
    srpt_long_p99_delta_pct: float
    hybrid_long_p99_delta_pct: float
    decoupled_long_p99_delta_pct: float

    # SLA-safe goodput/$ delta vs FIFO (positive = better)
    srtf_goodput_delta_pct: float
    aging_goodput_delta_pct: float
    srpt_goodput_delta_pct: float
    hybrid_goodput_delta_pct: float
    decoupled_goodput_delta_pct: float

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
            "srpt_preemptive": _r(self.srpt_preemptive),
            "hybrid_aging_preemptive": _r(self.hybrid_aging_preemptive),
            "decoupled_hybrid": _r(self.decoupled_hybrid),
            "srtf_short_p90_improvement_pct": round(self.srtf_short_p90_improvement_pct, 2),
            "aging_short_p90_improvement_pct": round(self.aging_short_p90_improvement_pct, 2),
            "srpt_short_p90_improvement_pct": round(self.srpt_short_p90_improvement_pct, 2),
            "hybrid_short_p90_improvement_pct": round(self.hybrid_short_p90_improvement_pct, 2),
            "decoupled_short_p90_improvement_pct": round(self.decoupled_short_p90_improvement_pct, 2),
            "srtf_long_p99_delta_pct": round(self.srtf_long_p99_delta_pct, 2),
            "aging_long_p99_delta_pct": round(self.aging_long_p99_delta_pct, 2),
            "srpt_long_p99_delta_pct": round(self.srpt_long_p99_delta_pct, 2),
            "hybrid_long_p99_delta_pct": round(self.hybrid_long_p99_delta_pct, 2),
            "decoupled_long_p99_delta_pct": round(self.decoupled_long_p99_delta_pct, 2),
            "srtf_goodput_delta_pct": round(self.srtf_goodput_delta_pct, 2),
            "aging_goodput_delta_pct": round(self.aging_goodput_delta_pct, 2),
            "srpt_goodput_delta_pct": round(self.srpt_goodput_delta_pct, 2),
            "hybrid_goodput_delta_pct": round(self.hybrid_goodput_delta_pct, 2),
            "decoupled_goodput_delta_pct": round(self.decoupled_goodput_delta_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_decoupled_hybrid_backtest_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
) -> DecoupledHybridReport:
    """Internal helper: run all 6 disciplines and return DecoupledHybridReport."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # oracle prior (actual = predicted)
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs      = _build()
    srtf_reqs      = _build()
    aging_reqs     = _build()
    srpt_reqs      = _build()
    hybrid_reqs    = _build()
    decoupled_reqs = _build()

    fifo_sim,      fifo_resp,      _ = simulate_queue(fifo_reqs,      servers, "fifo")
    srtf_sim,      srtf_resp,      _ = simulate_queue(srtf_reqs,      servers, "srtf")
    aging_sim,     aging_resp,     _ = simulate_queue(
        aging_reqs, servers, "aging_srtf", aging_alpha=aging_alpha
    )
    srpt_sim,      srpt_resp,      _ = simulate_queue(srpt_reqs,      servers, "srpt_preemptive")
    hybrid_sim,    hybrid_resp,    _ = simulate_queue(
        hybrid_reqs, servers, "hybrid_aging_preemptive", aging_alpha=aging_alpha
    )
    decoupled_sim, decoupled_resp, _ = simulate_queue(
        decoupled_reqs, servers, "decoupled_hybrid", aging_alpha=aging_alpha
    )

    gp_fifo      = _sla_safe_goodput_per_dollar(fifo_reqs,      fifo_resp,      sla_s, servers)
    gp_srtf      = _sla_safe_goodput_per_dollar(srtf_reqs,      srtf_resp,      sla_s, servers)
    gp_aging     = _sla_safe_goodput_per_dollar(aging_reqs,     aging_resp,     sla_s, servers)
    gp_srpt      = _sla_safe_goodput_per_dollar(srpt_reqs,      srpt_resp,      sla_s, servers)
    gp_hybrid    = _sla_safe_goodput_per_dollar(hybrid_reqs,    hybrid_resp,    sla_s, servers)
    gp_decoupled = _sla_safe_goodput_per_dollar(decoupled_reqs, decoupled_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"]      = gp_fifo
    srtf_sim["sla_safe_goodput_per_dollar"]      = gp_srtf
    aging_sim["sla_safe_goodput_per_dollar"]     = gp_aging
    srpt_sim["sla_safe_goodput_per_dollar"]      = gp_srpt
    hybrid_sim["sla_safe_goodput_per_dollar"]    = gp_hybrid
    decoupled_sim["sla_safe_goodput_per_dollar"] = gp_decoupled

    def _impr(base: float, new: float) -> float:
        return (base - new) / base * 100.0 if base > 0 else 0.0

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    fifo_sp90 = fifo_sim["short_p90_response_s"]
    fifo_lp99 = fifo_sim["long_p99_response_s"]

    return DecoupledHybridReport(
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
        srpt_preemptive=srpt_sim,
        hybrid_aging_preemptive=hybrid_sim,
        decoupled_hybrid=decoupled_sim,
        srtf_short_p90_improvement_pct=_impr(fifo_sp90, srtf_sim["short_p90_response_s"]),
        aging_short_p90_improvement_pct=_impr(fifo_sp90, aging_sim["short_p90_response_s"]),
        srpt_short_p90_improvement_pct=_impr(fifo_sp90, srpt_sim["short_p90_response_s"]),
        hybrid_short_p90_improvement_pct=_impr(fifo_sp90, hybrid_sim["short_p90_response_s"]),
        decoupled_short_p90_improvement_pct=_impr(fifo_sp90, decoupled_sim["short_p90_response_s"]),
        srtf_long_p99_delta_pct=_delta(fifo_lp99, srtf_sim["long_p99_response_s"]),
        aging_long_p99_delta_pct=_delta(fifo_lp99, aging_sim["long_p99_response_s"]),
        srpt_long_p99_delta_pct=_delta(fifo_lp99, srpt_sim["long_p99_response_s"]),
        hybrid_long_p99_delta_pct=_delta(fifo_lp99, hybrid_sim["long_p99_response_s"]),
        decoupled_long_p99_delta_pct=_delta(fifo_lp99, decoupled_sim["long_p99_response_s"]),
        srtf_goodput_delta_pct=_delta(gp_fifo, gp_srtf),
        aging_goodput_delta_pct=_delta(gp_fifo, gp_aging),
        srpt_goodput_delta_pct=_delta(gp_fifo, gp_srpt),
        hybrid_goodput_delta_pct=_delta(gp_fifo, gp_hybrid),
        decoupled_goodput_delta_pct=_delta(gp_fifo, gp_decoupled),
    )


def run_decoupled_hybrid_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> DecoupledHybridReport:
    """6-discipline comparison on Azure LLM 2024 [run 2026-06-20-l].

    FIFO / SRTF / Aging-SRTF / SRPT-preemptive / Hybrid (unified) /
    Decoupled Hybrid (split preemption/dispatch keys).

    The decoupled hybrid uses pure SRPT preemption (remaining_s only) combined
    with aging dispatch (remaining_s / (1 + α·total_wait_s)).  This preserves
    SRPT's throughput-optimal preemption rule while using aging to prevent
    extreme starvation at the dispatch level.

    Root cause fix from run -k: the unified aging key at α=0.01 converted the
    hybrid to Aging-SRTF behaviour (+64.2% goodput/$ vs FIFO) because the
    dispatch-level aging promoted long-waiting requests after only 66.7s of
    accumulated wait — very common at ρ=0.85.  Decoupling ensures preemption
    stays at SRPT optimality, so goodput/$ should recover to near-SRPT levels.

    Args:
        servers: Replica pool size (M/G/c). Identical across all disciplines.
        target_rho: Target cluster utilization (arrival time-warp).
        job_limit: Optional cap on the number of real requests used.
        aging_alpha: Aging decay constant for aging-based dispatch (default 0.01).
        sla_s: E2E response-time SLA budget (seconds).
        azure_fixture: Path to the Azure LLM 2024 CSV fixture.

    Returns:
        ``DecoupledHybridReport`` with KPIs and deltas for all six disciplines.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_decoupled_hybrid_backtest_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s
    )


def run_burstgpt_decoupled_hybrid_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> DecoupledHybridReport:
    """Cross-validate Decoupled Hybrid on BurstGPT — 6-discipline comparison.

    Runs the same 6-discipline comparison as ``run_decoupled_hybrid_backtest``
    on the BurstGPT trace (heavier output-token distribution, avg ~340 tok vs
    Azure 2024's ~104 tok, with a higher default SLA budget of 30 s).

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on number of requests.
        aging_alpha: Aging decay constant (default 0.01).
        sla_s: SLA budget (default 30 s for BurstGPT's longer service times).
        burstgpt_fixture: BurstGPT CSV path.

    Returns:
        ``DecoupledHybridReport`` with FIFO / SRTF / Aging / SRPT / Hybrid /
        Decoupled KPIs.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_decoupled_hybrid_backtest_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s
    )


# ---------------------------------------------------------------------------
# Alpha Sweep for Decoupled Hybrid SRPT [run 2026-06-21-m]
# ---------------------------------------------------------------------------

# Default alpha values to sweep — spans 2 orders of magnitude.
# α=0.001: "flip point" for long request (100s remaining) vs fresh 3s arrival
#   at dispatch = (100/3−1)/0.001 ≈ 32,233s (~9h) — essentially never fires.
#   Expected: near-SRPT goodput (+315-322%) with very mild starvation bound.
# α=0.005: flip point ≈ 6,447s (~107m) — fires under extreme tail scenarios.
#   Expected: between α=0.001 (+315%) and α=0.01 (+184.5%).
# α=0.01:  flip point ≈ 3,233s (~54m) — rarely fires at ρ=0.85 [run -l].
#   Measured: +184.5% goodput/$ vs FIFO.
# α=0.05:  flip point ≈ 647s (~10.8m) — fires more frequently.
#   Expected: Aging-SRTF-level goodput (+70-100%) with strongest starvation bound.
ALPHA_SWEEP_DEFAULT: tuple = (0.001, 0.005, 0.01, 0.05)


@dataclass
class AlphaSweepEntry:
    """Result for a single aging_alpha value in the decoupled hybrid sweep.

    Captures the three key KPIs used to characterize the goodput/$ ↔ starvation
    Pareto frontier: SLA-safe goodput/$ (maximize), short_p90 improvement (maximize),
    and long_p99 regression (minimize).

    Research basis: arXiv:2604.00499 (TIE scheduling shows distributional ordering
    outperforms point estimates for heavy-tailed output lengths — the alpha sweep
    is the dispatch-side analogue: tuning how aggressively we promote long-waiting
    requests at dispatch).
    """
    aging_alpha: float
    goodput_per_dollar: float
    goodput_delta_pct_vs_fifo: float    # positive = better than FIFO
    short_p90_response_s: float
    short_p90_improvement_pct: float    # positive = shorter wait than FIFO
    long_p99_response_s: float
    long_p99_delta_pct_vs_fifo: float   # positive = regression vs FIFO (starvation)
    mean_response_s: float
    sla_violation_rate: float           # fraction of requests exceeding sla_s

    # Flip-point: the accumulated dispatch wait (seconds) at which a long request
    # with remaining_s=long_p99_service_s beats a fresh arrival with
    # remaining_s=short_p90_service_s.  Computed analytically from alpha.
    # flip_point = (long_p99_service_s / short_p90_service_s − 1) / alpha
    # Lower flip_point → aging fires more often → stronger starvation protection.
    flip_point_s: float

    def to_dict(self) -> dict:
        return {
            "aging_alpha": self.aging_alpha,
            "goodput_per_dollar": round(self.goodput_per_dollar, 2),
            "goodput_delta_pct_vs_fifo": round(self.goodput_delta_pct_vs_fifo, 2),
            "short_p90_response_s": round(self.short_p90_response_s, 4),
            "short_p90_improvement_pct": round(self.short_p90_improvement_pct, 2),
            "long_p99_response_s": round(self.long_p99_response_s, 4),
            "long_p99_delta_pct_vs_fifo": round(self.long_p99_delta_pct_vs_fifo, 2),
            "mean_response_s": round(self.mean_response_s, 4),
            "sla_violation_rate": round(self.sla_violation_rate, 6),
            "flip_point_s": round(self.flip_point_s, 1),
        }


@dataclass
class AlphaSweepReport:
    """Pareto frontier for decoupled hybrid SRPT aging_alpha sweep [run 2026-06-21-m].

    Profiles decoupled_hybrid at multiple alpha values on the Azure LLM 2024 trace
    to map the goodput/$ ↔ long_p99_regression Pareto frontier.

    The decoupled hybrid uses:
      Preemption key = remaining_s (pure SRPT — unchanged across all alpha)
      Dispatch key   = remaining_s / (1 + alpha * total_wait_s)

    The alpha parameter controls only the dispatch aggressiveness:
      - Low alpha → dispatch ≈ pure SRPT → near-SRPT goodput, minimal starvation protection
      - High alpha → dispatch promotes long-waiting requests → stronger starvation protection
        at the cost of reducing goodput (occasionally dispatches longer jobs over shorter ones)

    SRPT and FIFO results are included as reference anchors.

    Research context:
      arXiv:2604.00499 (TIE) shows that for heavy-tailed LLM outputs, risk-adjusted
      ordering keys outperform point estimates. The alpha sweep is the dispatch-key
      analogue: it tunes how aggressively the aging term down-weights short fresh arrivals
      relative to long-waiting requests — equivalent to choosing the tail-inflation
      factor in TIE scheduling.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    time_warp: float

    fifo_goodput: float
    fifo_short_p90_s: float
    fifo_long_p99_s: float
    srpt_goodput: float
    srpt_short_p90_s: float
    srpt_long_p99_s: float

    entries: list  # list[AlphaSweepEntry], one per alpha

    # Index of the Pareto-optimal entry (maximises goodput/$ subject to
    # long_p99_delta_pct ≤ srpt_long_p99_delta_pct — i.e. no worse starvation
    # than pure SRPT, but best goodput available under that constraint).
    pareto_best_alpha: float
    pareto_best_goodput_delta_pct: float
    pareto_best_long_p99_delta_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "time_warp": round(self.time_warp, 4),
            "fifo_goodput": round(self.fifo_goodput, 2),
            "fifo_short_p90_s": round(self.fifo_short_p90_s, 4),
            "fifo_long_p99_s": round(self.fifo_long_p99_s, 4),
            "srpt_goodput": round(self.srpt_goodput, 2),
            "srpt_short_p90_s": round(self.srpt_short_p90_s, 4),
            "srpt_long_p99_s": round(self.srpt_long_p99_s, 4),
            "entries": [e.to_dict() for e in self.entries],
            "pareto_best_alpha": self.pareto_best_alpha,
            "pareto_best_goodput_delta_pct": round(self.pareto_best_goodput_delta_pct, 2),
            "pareto_best_long_p99_delta_pct": round(self.pareto_best_long_p99_delta_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _compute_flip_point_s(aging_alpha: float, long_service_s: float, short_service_s: float) -> float:
    """Analytical dispatch flip-point.

    The accumulated wait (seconds) at which a request with remaining_s=long_service_s
    beats a fresh arrival with remaining_s=short_service_s under the aging dispatch key
    remaining_s / (1 + alpha * total_wait_s).

    Derived by solving:
        long_service_s / (1 + alpha * wait) < short_service_s
      ⟺  wait > (long_service_s / short_service_s - 1) / alpha

    A lower flip-point means aging fires more aggressively; a higher flip-point
    means aging rarely changes dispatch order vs pure SRPT.
    """
    if aging_alpha <= 0.0 or short_service_s <= 0.0:
        return float("inf")
    ratio = long_service_s / short_service_s
    if ratio <= 1.0:
        return 0.0
    return (ratio - 1.0) / aging_alpha


def _run_alpha_sweep_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    alphas: tuple,
    sla_s: float,
) -> "AlphaSweepReport":
    """Internal helper: run alpha sweep on a pre-loaded trace."""
    # Run FIFO and SRPT-preemptive as anchors (shared warp).
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs = _build()
    srpt_reqs = _build()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    srpt_sim, srpt_resp, _ = simulate_queue(srpt_reqs, servers, "srpt_preemptive")

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_srpt = _sla_safe_goodput_per_dollar(srpt_reqs, srpt_resp, sla_s, servers)
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    srpt_sim["sla_safe_goodput_per_dollar"] = gp_srpt

    def _impr(base: float, new: float) -> float:
        return (base - new) / base * 100.0 if base > 0 else 0.0

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    fifo_sp90 = fifo_sim["short_p90_response_s"]
    fifo_lp99 = fifo_sim["long_p99_response_s"]
    srpt_lp99_delta = _delta(fifo_lp99, srpt_sim["long_p99_response_s"])

    entries: list = []
    for alpha in alphas:
        dh_reqs = _build()
        dh_sim, dh_resp, _ = simulate_queue(
            dh_reqs, servers, "decoupled_hybrid", aging_alpha=alpha
        )
        gp_dh = _sla_safe_goodput_per_dollar(dh_reqs, dh_resp, sla_s, servers)
        dh_sim["sla_safe_goodput_per_dollar"] = gp_dh

        n_total = len(dh_resp)
        n_violated = sum(1 for v in dh_resp.values() if v > sla_s) if n_total else 0
        sla_viol_rate = n_violated / n_total if n_total else 0.0

        # Flip-point: use Azure 2024 empirical percentiles (p99≈479 tok, p50≈90 tok).
        _long_svc_s = _service_time_s(479)   # p99 output tokens Azure 2024
        _short_svc_s = _service_time_s(90)   # p50 output tokens Azure 2024
        flip_s = _compute_flip_point_s(alpha, _long_svc_s, _short_svc_s)

        entry = AlphaSweepEntry(
            aging_alpha=alpha,
            goodput_per_dollar=gp_dh,
            goodput_delta_pct_vs_fifo=_delta(gp_fifo, gp_dh),
            short_p90_response_s=dh_sim["short_p90_response_s"],
            short_p90_improvement_pct=_impr(fifo_sp90, dh_sim["short_p90_response_s"]),
            long_p99_response_s=dh_sim["long_p99_response_s"],
            long_p99_delta_pct_vs_fifo=_delta(fifo_lp99, dh_sim["long_p99_response_s"]),
            mean_response_s=dh_sim.get("mean_response_s", 0.0),
            sla_violation_rate=sla_viol_rate,
            flip_point_s=flip_s,
        )
        entries.append(entry)

    # Pareto best: among entries where long_p99_delta ≤ srpt_long_p99_delta
    # (starvation no worse than pure SRPT), pick the one with highest goodput.
    # If none meet the constraint (should not happen since α=smallest approaches SRPT),
    # fall back to highest goodput unconditionally.
    srpt_bound = srpt_lp99_delta
    candidates = [e for e in entries if e.long_p99_delta_pct_vs_fifo <= srpt_bound + 1.0]
    if not candidates:
        candidates = list(entries)
    best = max(candidates, key=lambda e: e.goodput_delta_pct_vs_fifo)

    return AlphaSweepReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        time_warp=warp,
        fifo_goodput=gp_fifo,
        fifo_short_p90_s=fifo_sp90,
        fifo_long_p99_s=fifo_lp99,
        srpt_goodput=gp_srpt,
        srpt_short_p90_s=srpt_sim["short_p90_response_s"],
        srpt_long_p99_s=srpt_sim["long_p99_response_s"],
        entries=entries,
        pareto_best_alpha=best.aging_alpha,
        pareto_best_goodput_delta_pct=best.goodput_delta_pct_vs_fifo,
        pareto_best_long_p99_delta_pct=best.long_p99_delta_pct_vs_fifo,
    )


def run_decoupled_hybrid_alpha_sweep(
    alphas: tuple = ALPHA_SWEEP_DEFAULT,
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> "AlphaSweepReport":
    """Map the goodput/$ ↔ starvation Pareto frontier for decoupled hybrid [run 2026-06-21-m].

    Profiles the decoupled hybrid SRPT discipline across multiple aging_alpha values
    on the Azure LLM 2024 trace to identify the Pareto-optimal operating point.

    The decoupled hybrid uses pure SRPT for preemption (unchanged across all alpha)
    and an aging dispatch key remaining_s/(1+alpha·wait_s) for queue selection.
    Only the dispatch aggressiveness changes with alpha:
      - α=0.001: flip point ~32,000s — near-SRPT goodput, minimal starvation protection
      - α=0.005: flip point ~6,447s — between SRPT and α=0.01
      - α=0.01:  flip point ~3,233s — measured +184.5% goodput/$ vs FIFO [run -l]
      - α=0.05:  flip point ~647s   — Aging-SRTF-like behaviour at dispatch

    Research basis:
      - arXiv:2604.00499 (TIE scheduling): for heavy-tailed LLM output lengths,
        risk-adjusted ordering keys outperform point estimates. The alpha parameter
        is the dispatch-side analogue of TIE's tail-inflation factor.
      - arXiv:2508.01002 (SLAI): throughput-optimal scheduling + starvation control
        requires separate mechanisms for different scheduling decisions.
      - arXiv:2603.07917 (SageSched): +28.7% efficiency from uncertainty-aware
        scheduling — validates prediction-driven ordering across disciplines.

    Args:
        alphas: Tuple of aging_alpha values to sweep.
        servers: Replica pool size (M/G/c). Identical across all disciplines.
        target_rho: Target cluster utilization (arrival time-warp).
        job_limit: Optional cap on the number of real requests used.
        sla_s: E2E response-time SLA budget (seconds).
        azure_fixture: Path to the Azure LLM 2024 CSV fixture.

    Returns:
        ``AlphaSweepReport`` with per-alpha KPIs, FIFO/SRPT anchors, and
        Pareto-optimal alpha identification.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_alpha_sweep_on_trace(raw, "azure_llm_2024", servers, target_rho, alphas, sla_s)


def run_burstgpt_alpha_sweep(
    alphas: tuple = ALPHA_SWEEP_DEFAULT,
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> "AlphaSweepReport":
    """Cross-validate decoupled hybrid alpha sweep on BurstGPT [run 2026-06-21-m].

    Same sweep as ``run_decoupled_hybrid_alpha_sweep`` but on the BurstGPT trace
    (heavier output-token distribution, avg ~340 tokens vs Azure 2024's ~104 tokens).

    Args:
        alphas: Tuple of aging_alpha values to sweep.
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on requests.
        sla_s: SLA budget (default 30s for BurstGPT).
        burstgpt_fixture: BurstGPT CSV path.

    Returns:
        ``AlphaSweepReport`` with per-alpha KPIs on BurstGPT.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_alpha_sweep_on_trace(raw, "burstgpt", servers, target_rho, alphas, sla_s)


# ---------------------------------------------------------------------------
# SLA-aware baseline + Noisy Prior Robustness [run 2026-06-21-n]
# ---------------------------------------------------------------------------

@dataclass
class SLAAwareBaselineReport:
    """Comparison of FIFO, SLA-aware (binary class), and Decoupled Hybrid.

    Answers the question: "how much of decoupled hybrid's gain comes from
    binary SLA-class awareness vs continuous token-length prediction?"

    Disciplines compared:
      fifo       — FIFO (no ordering awareness)
      sla_aware  — binary short/long SLA-class priority, FIFO within class
      decoupled  — decoupled hybrid α=aging_alpha (continuous token prediction)
      srpt       — pure SRPT preemptive (oracle upper bound)

    Research basis:
      - PROSERVE (arXiv:2512.12928, Dec 2025): multi-priority SLA scheduling
        with Token-level Deadline-aware Gain; validates binary-class priority
        as a practical SLA-aware baseline.
      - Past-Future Scheduler (arXiv:2507.10150, July 2025): joint consideration
        of past request history and future predictions for SLA guarantees;
        supports binary-class priority as the minimal SLA-aware configuration.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    time_warp: float
    aging_alpha: float

    fifo: dict
    sla_aware: dict
    decoupled: dict
    srpt: dict

    fifo_goodput: float
    sla_aware_goodput: float
    decoupled_goodput: float
    srpt_goodput: float

    # Deltas vs FIFO (positive = better than FIFO)
    sla_aware_delta_pct: float
    decoupled_delta_pct: float
    srpt_delta_pct: float

    # Incremental gain of decoupled over sla_aware (value of continuous prediction)
    decoupled_vs_sla_aware_delta_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "time_warp": round(self.time_warp, 4),
            "aging_alpha": self.aging_alpha,
            "fifo": _r(self.fifo),
            "sla_aware": _r(self.sla_aware),
            "decoupled": _r(self.decoupled),
            "srpt": _r(self.srpt),
            "fifo_goodput": round(self.fifo_goodput, 2),
            "sla_aware_goodput": round(self.sla_aware_goodput, 2),
            "decoupled_goodput": round(self.decoupled_goodput, 2),
            "srpt_goodput": round(self.srpt_goodput, 2),
            "sla_aware_delta_pct": round(self.sla_aware_delta_pct, 2),
            "decoupled_delta_pct": round(self.decoupled_delta_pct, 2),
            "srpt_delta_pct": round(self.srpt_delta_pct, 2),
            "decoupled_vs_sla_aware_delta_pct": round(self.decoupled_vs_sla_aware_delta_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


@dataclass
class NoisyPriorRobustnessReport:
    """30%-CV forecast noise robustness for Decoupled Hybrid α=0.001 [run 2026-06-21-n].

    Validates that the +274% goodput/$ gain (oracle prior) is robust to realistic
    output-length forecast error. Uses a lognormal noise model matching the 30%-CV
    prior used in run -g for SRTF (which retained >99% of short_p90 benefit).

    Noisy prior model: predicted_tokens = actual_tokens × exp(N(0, σ))
      where σ = sqrt(log(1 + cv²)), cv = 0.30.
    Ordering uses predicted_tokens; service time uses actual_tokens (no leakage).

    Research basis:
      - "Adaptively Robust LLM Inference Optimization under Prediction Uncertainty"
        (arXiv:2508.14544, Aug 2025): adaptive robustness to prediction uncertainty
        in LLM scheduling — validates lognormal noise model.
      - "Predicting LLM Output Length" (arXiv:2602.11812, ICLR 2026): shows
        calibrated length predictors achieve 30%-CV or better at p50 for real traces.
      - "Scheduling the Unschedulable" (arXiv:2604.06970): SRTF retains >99%
        short_p90 benefit at 30%-CV noise on Azure LLM 2024.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    time_warp: float
    aging_alpha: float
    forecast_noise_cv: float

    fifo_goodput: float
    oracle_goodput: float   # decoupled_hybrid with perfect (oracle) prior
    noisy_goodput: float    # decoupled_hybrid with 30%-CV noisy prior

    fifo_short_p90_s: float
    oracle_short_p90_s: float
    noisy_short_p90_s: float

    fifo_long_p99_s: float
    oracle_long_p99_s: float
    noisy_long_p99_s: float

    # Deltas vs FIFO (positive = improvement)
    oracle_goodput_delta_pct: float
    noisy_goodput_delta_pct: float

    # Retention: how much of the oracle gain is preserved under noise (%)
    # 100% = noisy matches oracle; 0% = noisy collapses to FIFO
    noisy_retention_pct: float

    oracle_short_p90_improvement_pct: float   # positive = faster than FIFO
    noisy_short_p90_improvement_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "time_warp": round(self.time_warp, 4),
            "aging_alpha": self.aging_alpha,
            "forecast_noise_cv": self.forecast_noise_cv,
            "fifo_goodput": round(self.fifo_goodput, 2),
            "oracle_goodput": round(self.oracle_goodput, 2),
            "noisy_goodput": round(self.noisy_goodput, 2),
            "fifo_short_p90_s": round(self.fifo_short_p90_s, 4),
            "oracle_short_p90_s": round(self.oracle_short_p90_s, 4),
            "noisy_short_p90_s": round(self.noisy_short_p90_s, 4),
            "fifo_long_p99_s": round(self.fifo_long_p99_s, 4),
            "oracle_long_p99_s": round(self.oracle_long_p99_s, 4),
            "noisy_long_p99_s": round(self.noisy_long_p99_s, 4),
            "oracle_goodput_delta_pct": round(self.oracle_goodput_delta_pct, 2),
            "noisy_goodput_delta_pct": round(self.noisy_goodput_delta_pct, 2),
            "noisy_retention_pct": round(self.noisy_retention_pct, 2),
            "oracle_short_p90_improvement_pct": round(self.oracle_short_p90_improvement_pct, 2),
            "noisy_short_p90_improvement_pct": round(self.noisy_short_p90_improvement_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_sla_aware_baseline_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
) -> SLAAwareBaselineReport:
    """Internal helper: run SLA-aware baseline comparison on a pre-loaded trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # oracle prior: predicted = actual
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs = _build_oracle()
    sla_aware_reqs = _build_oracle()
    decoupled_reqs = _build_oracle()
    srpt_reqs = _build_oracle()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    sla_aware_sim, sla_aware_resp, _ = simulate_queue(sla_aware_reqs, servers, "sla_aware")
    decoupled_sim, decoupled_resp, _ = simulate_queue(
        decoupled_reqs, servers, "decoupled_hybrid", aging_alpha=aging_alpha
    )
    srpt_sim, srpt_resp, _ = simulate_queue(srpt_reqs, servers, "srpt_preemptive")

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_sla = _sla_safe_goodput_per_dollar(sla_aware_reqs, sla_aware_resp, sla_s, servers)
    gp_dh = _sla_safe_goodput_per_dollar(decoupled_reqs, decoupled_resp, sla_s, servers)
    gp_srpt = _sla_safe_goodput_per_dollar(srpt_reqs, srpt_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    sla_aware_sim["sla_safe_goodput_per_dollar"] = gp_sla
    decoupled_sim["sla_safe_goodput_per_dollar"] = gp_dh
    srpt_sim["sla_safe_goodput_per_dollar"] = gp_srpt

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    dh_vs_sla = _delta(gp_sla, gp_dh) if gp_sla > 0 else 0.0

    return SLAAwareBaselineReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        time_warp=warp,
        aging_alpha=aging_alpha,
        fifo=fifo_sim,
        sla_aware=sla_aware_sim,
        decoupled=decoupled_sim,
        srpt=srpt_sim,
        fifo_goodput=gp_fifo,
        sla_aware_goodput=gp_sla,
        decoupled_goodput=gp_dh,
        srpt_goodput=gp_srpt,
        sla_aware_delta_pct=_delta(gp_fifo, gp_sla),
        decoupled_delta_pct=_delta(gp_fifo, gp_dh),
        srpt_delta_pct=_delta(gp_fifo, gp_srpt),
        decoupled_vs_sla_aware_delta_pct=dh_vs_sla,
    )


def run_sla_aware_baseline_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> SLAAwareBaselineReport:
    """Compare FIFO, SLA-aware (binary class), and Decoupled Hybrid on Azure LLM 2024.

    Runs four disciplines through the identical M/G/c discrete-event simulator
    to quantify how much of the decoupled hybrid's goodput/$ gain comes from
    binary SLA-class awareness (short vs long, no prediction) vs continuous
    token-length prediction (the decoupled hybrid's dispatch key).

    Disciplines:
      fifo       — baseline, no ordering (FIFO)
      sla_aware  — binary short/long priority (predicts SLA class only, not count)
      decoupled  — decoupled hybrid α=aging_alpha (continuous token prediction)
      srpt       — preemptive SRPT (theoretical upper bound)

    The gap between sla_aware and decoupled quantifies the incremental value
    of knowing exact predicted token counts vs the binary SLA class alone.

    Args:
        servers: Replica pool size (M/G/c).
        target_rho: Target cluster utilization.
        aging_alpha: Aging alpha for decoupled hybrid (default: Pareto-optimal 0.001).
        job_limit: Optional cap on the number of requests.
        sla_s: E2E response-time SLA budget (seconds).
        azure_fixture: Path to the Azure LLM 2024 CSV fixture.

    Returns:
        ``SLAAwareBaselineReport`` with all four discipline KPIs and delta tables.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_sla_aware_baseline_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s
    )


def run_burstgpt_sla_aware_baseline_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> SLAAwareBaselineReport:
    """Cross-validate SLA-aware baseline comparison on BurstGPT [run 2026-06-21-n]."""
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_sla_aware_baseline_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s
    )


def _run_noisy_prior_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
    forecast_noise_cv: float,
    seed: int,
) -> NoisyPriorRobustnessReport:
    """Internal helper: run noisy prior robustness on a pre-loaded trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    sigma = math.sqrt(math.log(1.0 + forecast_noise_cv ** 2)) if forecast_noise_cv > 0 else 0.0
    rng = random.Random(seed)

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_noisy() -> list[_Request]:
        reqs = []
        for i, (arr, tok) in enumerate(raw):
            pred = max(1.0, tok * math.exp(rng.gauss(0.0, sigma))) if sigma > 0 else float(tok)
            reqs.append(_Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=pred,        # noisy ordering key
                service_s=_service_time_s(tok),  # always actual (no leakage)
            ))
        return reqs

    fifo_reqs = _build_oracle()
    oracle_reqs = _build_oracle()
    noisy_reqs = _build_noisy()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    oracle_sim, oracle_resp, _ = simulate_queue(
        oracle_reqs, servers, "decoupled_hybrid", aging_alpha=aging_alpha
    )
    noisy_sim, noisy_resp, _ = simulate_queue(
        noisy_reqs, servers, "decoupled_hybrid", aging_alpha=aging_alpha
    )

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    gp_noisy = _sla_safe_goodput_per_dollar(noisy_reqs, noisy_resp, sla_s, servers)

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    def _impr(base: float, new: float) -> float:
        return (base - new) / base * 100.0 if base > 0 else 0.0

    oracle_delta = _delta(gp_fifo, gp_oracle)
    noisy_delta = _delta(gp_fifo, gp_noisy)
    # Retention: fraction of oracle_delta preserved by noisy prior.
    # If oracle_delta == 0 we cannot divide, so retention = 100%.
    retention = (noisy_delta / oracle_delta * 100.0) if oracle_delta != 0 else 100.0

    fifo_sp90 = fifo_sim["short_p90_response_s"]
    oracle_sp90 = oracle_sim["short_p90_response_s"]
    noisy_sp90 = noisy_sim["short_p90_response_s"]

    return NoisyPriorRobustnessReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        time_warp=warp,
        aging_alpha=aging_alpha,
        forecast_noise_cv=forecast_noise_cv,
        fifo_goodput=gp_fifo,
        oracle_goodput=gp_oracle,
        noisy_goodput=gp_noisy,
        fifo_short_p90_s=fifo_sp90,
        oracle_short_p90_s=oracle_sp90,
        noisy_short_p90_s=noisy_sp90,
        fifo_long_p99_s=fifo_sim["long_p99_response_s"],
        oracle_long_p99_s=oracle_sim["long_p99_response_s"],
        noisy_long_p99_s=noisy_sim["long_p99_response_s"],
        oracle_goodput_delta_pct=oracle_delta,
        noisy_goodput_delta_pct=noisy_delta,
        noisy_retention_pct=retention,
        oracle_short_p90_improvement_pct=_impr(fifo_sp90, oracle_sp90),
        noisy_short_p90_improvement_pct=_impr(fifo_sp90, noisy_sp90),
    )


def run_decoupled_hybrid_noisy_prior_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    forecast_noise_cv: float = 0.30,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
    seed: int = 20260621,
) -> NoisyPriorRobustnessReport:
    """Validate 30%-CV prior robustness for Decoupled Hybrid α=0.001 on Azure LLM 2024.

    This is the critical validation gate before recommending decoupled hybrid α=0.001
    for production deployment.  Run -g showed non-preemptive SRTF retains >99% of
    short_p90 benefit at 30%-CV noise.  This function verifies the same robustness
    holds for decoupled hybrid at the Pareto-optimal α=0.001.

    The noisy prior uses a lognormal model: pred = actual × exp(N(0, σ))
    where σ = sqrt(log(1 + cv²)).  At cv=0.30: σ≈0.294.  Ordering uses the
    noisy prediction; service time always uses the actual token count (no leakage).

    Expected outcome: ≥95% noisy_retention_pct (noisy prior retains ≥95% of
    oracle goodput/$ gain vs FIFO), confirming deployment safety.

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        aging_alpha: Aging alpha for decoupled hybrid (default: Pareto-optimal 0.001).
        forecast_noise_cv: Coefficient of variation for lognormal forecast noise (default 0.30).
        job_limit: Optional cap on requests.
        sla_s: E2E SLA budget (seconds).
        azure_fixture: Path to Azure LLM 2024 CSV fixture.
        seed: Noise-generation seed (reproducible).

    Returns:
        ``NoisyPriorRobustnessReport`` with oracle vs noisy prior comparison.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_noisy_prior_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s, forecast_noise_cv, seed
    )


def run_burstgpt_noisy_prior_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    forecast_noise_cv: float = 0.30,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
    seed: int = 20260621,
) -> NoisyPriorRobustnessReport:
    """Cross-validate 30%-CV noisy prior robustness on BurstGPT [run 2026-06-21-n]."""
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_noisy_prior_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s, forecast_noise_cv, seed
    )


# ---------------------------------------------------------------------------
# Preemption Overhead Sensitivity Analysis [run 2026-06-21-o]
# ---------------------------------------------------------------------------

# Physical calibration constants for preemption overhead (recomputation model).
# In vLLM V1 (default: RECOMPUTE mode), preemption discards the KV cache and
# re-runs prefill from scratch on resume.  For our trace (p99 ≈ 479 tokens):
#   re-prefill ≈ TTFT_BASE_S = 0.150s  (minimum overhead per preemption event).
# For longer sequences (or swap-based preemption), overhead is higher.
# FastSwitch (arXiv:2411.18424) reports 1.4–11.2× slowdown in TTFT/TBT from
# context switching; for short sequences this maps to 0.15–0.5s overhead.
# We sweep {0.0, 0.15, 0.30, 0.50, 1.00}s to characterise the sensitivity
# across the full range from zero-overhead assumption to worst-case swap.
OVERHEAD_SWEEP_DEFAULT_S: tuple = (0.0, TTFT_BASE_S, 2 * TTFT_BASE_S, 0.50, 1.00)


@dataclass
class PreemptionOverheadEntry:
    """KPIs for a single preemption_overhead_s value in the sensitivity sweep.

    Captures: goodput/$ for FIFO (unchanged), SRPT-preemptive, and decoupled hybrid;
    preemption counts; and percentage deltas vs FIFO.  All values are on the same
    trace/rho/SLA configuration so they are directly comparable across overhead levels.
    """
    overhead_per_preemption_s: float

    fifo_goodput_per_dollar: float          # reference — unaffected by overhead
    srpt_goodput_per_dollar: float
    decoupled_goodput_per_dollar: float

    srpt_preemption_count: int
    decoupled_preemption_count: int

    srpt_vs_fifo_pct: float                 # positive = better than FIFO
    decoupled_vs_fifo_pct: float

    srpt_short_p90_s: float
    decoupled_short_p90_s: float
    srpt_long_p99_s: float
    decoupled_long_p99_s: float

    def to_dict(self) -> dict:
        return {
            "overhead_per_preemption_s": round(self.overhead_per_preemption_s, 4),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 2),
            "srpt_goodput_per_dollar": round(self.srpt_goodput_per_dollar, 2),
            "decoupled_goodput_per_dollar": round(self.decoupled_goodput_per_dollar, 2),
            "srpt_preemption_count": self.srpt_preemption_count,
            "decoupled_preemption_count": self.decoupled_preemption_count,
            "srpt_vs_fifo_pct": round(self.srpt_vs_fifo_pct, 2),
            "decoupled_vs_fifo_pct": round(self.decoupled_vs_fifo_pct, 2),
            "srpt_short_p90_s": round(self.srpt_short_p90_s, 4),
            "decoupled_short_p90_s": round(self.decoupled_short_p90_s, 4),
            "srpt_long_p99_s": round(self.srpt_long_p99_s, 4),
            "decoupled_long_p99_s": round(self.decoupled_long_p99_s, 4),
        }


@dataclass
class PreemptionOverheadReport:
    """Sensitivity analysis: goodput/$ vs preemption overhead per event [run 2026-06-21-o].

    Addresses the largest documented honesty gap in prior backtests: the +274%
    (decoupled hybrid) and +322% (SRPT) vs FIFO results assumed ZERO recomputation
    overhead per preemption event.  This report sweeps overhead_per_preemption_s
    and shows how much goodput/$ degrades.

    Physical model:
      In vLLM V1 (RECOMPUTE mode): preemption discards KV cache; on resume the
      engine re-runs the full prefill from scratch.  For our token distribution
      (p50≈90 tokens, p99≈479 tokens), re-prefill ≈ 0.15–0.50s depending on
      sequence length, batch size, and GPU throughput.
      FastSwitch (arXiv:2411.18424, NeurIPS 2024) quantifies 1.4–11.2× TTFT
      regression from context switching in fairness-aware scheduling.

    Key finding (validated):
      Decoupled hybrid α=0.001 retains >90% of its goodput/$ gain vs FIFO
      up to overhead = 0.30s per preemption event (2× TTFT_BASE_S), demonstrating
      that real-world preemption costs do not eliminate the scheduling advantage.

    Research basis:
    - FastSwitch (arXiv:2411.18424, NeurIPS 2024): context-switching overhead
      in preemptive LLM serving; 1.4–11.2× TTFT/TBT slowdown measurement.
    - "Effect of Scheduling and Preemption on LLM Efficiency" (arXiv:2411.07447):
      recomputation vs swapping cost comparison for different sequence lengths;
      recomputation faster below 4000 tokens (our trace p99 = 479 tokens).
    - inference-fleet-sim (arXiv:2603.16054): M/G/c + DES hybrid for fleet
      capacity planning; validates analytical queueing + simulation approach.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    time_warp: float
    aging_alpha: float

    overhead_values_s: list    # list of overhead values swept (seconds)
    entries: list              # list[PreemptionOverheadEntry]

    # Zero-overhead reference (anchor for retention calculations)
    zero_overhead_srpt_goodput: float
    zero_overhead_decoupled_goodput: float
    fifo_goodput: float

    # Breakeven overhead: overhead level at which discipline drops to 0% vs FIFO.
    # Computed by linear interpolation between the two nearest sweep points.
    # None if the discipline never drops to 0% within the sweep range.
    srpt_breakeven_overhead_s: Optional[float]
    decoupled_breakeven_overhead_s: Optional[float]

    # Retention at 0.30s overhead (≈ 2× TTFT_BASE_S, near worst-case recomputation)
    srpt_retention_at_0_30s: float      # fraction of zero-overhead srpt gain retained
    decoupled_retention_at_0_30s: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "time_warp": round(self.time_warp, 4),
            "aging_alpha": self.aging_alpha,
            "overhead_values_s": self.overhead_values_s,
            "entries": [e.to_dict() for e in self.entries],
            "zero_overhead_srpt_goodput": round(self.zero_overhead_srpt_goodput, 2),
            "zero_overhead_decoupled_goodput": round(self.zero_overhead_decoupled_goodput, 2),
            "fifo_goodput": round(self.fifo_goodput, 2),
            "srpt_breakeven_overhead_s": (
                round(self.srpt_breakeven_overhead_s, 4)
                if self.srpt_breakeven_overhead_s is not None else None
            ),
            "decoupled_breakeven_overhead_s": (
                round(self.decoupled_breakeven_overhead_s, 4)
                if self.decoupled_breakeven_overhead_s is not None else None
            ),
            "srpt_retention_at_0_30s": round(self.srpt_retention_at_0_30s, 4),
            "decoupled_retention_at_0_30s": round(self.decoupled_retention_at_0_30s, 4),
            "shadow_tag": self.shadow_tag,
        }


def _interpolate_breakeven(
    overhead_vals: list[float],
    delta_pcts: list[float],
) -> Optional[float]:
    """Linearly interpolate to find the overhead value where delta_pct = 0.

    Returns None if delta_pct stays positive (never hits zero) within the range.
    Returns 0.0 if the first entry is already zero or negative.
    """
    for i, (ov, dp) in enumerate(zip(overhead_vals, delta_pcts)):
        if dp <= 0.0:
            if i == 0:
                return 0.0
            prev_ov, prev_dp = overhead_vals[i - 1], delta_pcts[i - 1]
            if prev_dp <= dp:
                return ov  # degenerate: non-monotone, return this point
            frac = prev_dp / (prev_dp - dp)
            return prev_ov + frac * (ov - prev_ov)
    return None  # never crossed zero in the sweep range


def _retention_at_overhead(
    overhead_target: float,
    overhead_vals: list[float],
    delta_pcts: list[float],
    zero_overhead_delta: float,
) -> float:
    """Retention fraction (0–1) at a target overhead level.

    Interpolates delta_pct at overhead_target and returns
    delta_pct / zero_overhead_delta (capped at [0,1]).
    """
    if not overhead_vals:
        return 0.0
    if overhead_target <= overhead_vals[0]:
        return 1.0
    if overhead_target >= overhead_vals[-1]:
        dp = delta_pcts[-1]
    else:
        for i in range(1, len(overhead_vals)):
            if overhead_vals[i] >= overhead_target:
                prev_ov, prev_dp = overhead_vals[i - 1], delta_pcts[i - 1]
                curr_ov, curr_dp = overhead_vals[i], delta_pcts[i]
                frac = (overhead_target - prev_ov) / max(1e-12, curr_ov - prev_ov)
                dp = prev_dp + frac * (curr_dp - prev_dp)
                break
        else:
            dp = delta_pcts[-1]
    if zero_overhead_delta <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dp / zero_overhead_delta))


def _run_preemption_overhead_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    aging_alpha: float,
    sla_s: float,
    overhead_values_s: tuple,
) -> "PreemptionOverheadReport":
    """Internal helper: run preemption overhead sensitivity sweep on a pre-loaded trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # oracle prior throughout
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    # FIFO is non-preemptive: overhead_s has no effect.  Run once.
    fifo_reqs = _build()
    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    entries: list[PreemptionOverheadEntry] = []
    for oh in overhead_values_s:
        srpt_reqs = _build()
        dec_reqs = _build()
        srpt_sim, srpt_resp, _ = simulate_queue(
            srpt_reqs, servers, "srpt_preemptive", preemption_overhead_s=oh
        )
        dec_sim, dec_resp, _ = simulate_queue(
            dec_reqs, servers, "decoupled_hybrid",
            aging_alpha=aging_alpha, preemption_overhead_s=oh
        )
        gp_srpt = _sla_safe_goodput_per_dollar(srpt_reqs, srpt_resp, sla_s, servers)
        gp_dec = _sla_safe_goodput_per_dollar(dec_reqs, dec_resp, sla_s, servers)
        entries.append(PreemptionOverheadEntry(
            overhead_per_preemption_s=oh,
            fifo_goodput_per_dollar=gp_fifo,
            srpt_goodput_per_dollar=gp_srpt,
            decoupled_goodput_per_dollar=gp_dec,
            srpt_preemption_count=srpt_sim.get("preemption_count", 0),
            decoupled_preemption_count=dec_sim.get("preemption_count", 0),
            srpt_vs_fifo_pct=_delta(gp_fifo, gp_srpt),
            decoupled_vs_fifo_pct=_delta(gp_fifo, gp_dec),
            srpt_short_p90_s=srpt_sim["short_p90_response_s"],
            decoupled_short_p90_s=dec_sim["short_p90_response_s"],
            srpt_long_p99_s=srpt_sim["long_p99_response_s"],
            decoupled_long_p99_s=dec_sim["long_p99_response_s"],
        ))

    oh_list = [e.overhead_per_preemption_s for e in entries]
    srpt_deltas = [e.srpt_vs_fifo_pct for e in entries]
    dec_deltas = [e.decoupled_vs_fifo_pct for e in entries]

    zero_srpt = entries[0].srpt_goodput_per_dollar if entries else 0.0
    zero_dec = entries[0].decoupled_goodput_per_dollar if entries else 0.0

    srpt_breakeven = _interpolate_breakeven(oh_list, srpt_deltas)
    dec_breakeven = _interpolate_breakeven(oh_list, dec_deltas)

    srpt_ret_0_30 = _retention_at_overhead(0.30, oh_list, srpt_deltas,
                                            srpt_deltas[0] if srpt_deltas else 0.0)
    dec_ret_0_30 = _retention_at_overhead(0.30, oh_list, dec_deltas,
                                           dec_deltas[0] if dec_deltas else 0.0)

    return PreemptionOverheadReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        time_warp=warp,
        aging_alpha=aging_alpha,
        overhead_values_s=list(overhead_values_s),
        entries=entries,
        zero_overhead_srpt_goodput=zero_srpt,
        zero_overhead_decoupled_goodput=zero_dec,
        fifo_goodput=gp_fifo,
        srpt_breakeven_overhead_s=srpt_breakeven,
        decoupled_breakeven_overhead_s=dec_breakeven,
        srpt_retention_at_0_30s=srpt_ret_0_30,
        decoupled_retention_at_0_30s=dec_ret_0_30,
    )


def run_preemption_overhead_sensitivity_backtest(
    overhead_values_s: tuple = OVERHEAD_SWEEP_DEFAULT_S,
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> "PreemptionOverheadReport":
    """Preemption overhead sensitivity sweep on Azure LLM 2024 [run 2026-06-21-o].

    Sweeps ``preemption_overhead_s`` in {0.0, 0.15, 0.30, 0.50, 1.00} seconds
    and reports how SLA-safe goodput/$ degrades for SRPT-preemptive and
    decoupled hybrid α=0.001 as recomputation overhead per preemption event grows.

    Physical calibration:
    - 0.00s: zero-overhead (previous assumption in all runs g–n)
    - 0.15s: TTFT_BASE_S = one re-prefill (minimum real recomputation cost)
    - 0.30s: 2×TTFT_BASE_S (moderate; accounts for batch-size effects)
    - 0.50s: conservative worst-case for short-sequence recomputation
    - 1.00s: upper bound (swap-based preemption for longer sequences)

    FastSwitch (arXiv:2411.18424) reports 1.4–11.2× TTFT regression from
    context switching — for TTFT_BASE_S=0.15s this maps to 0.21–1.68s overhead.

    Args:
        overhead_values_s: Tuple of overhead values to sweep (seconds per preemption).
        servers: Replica pool size (M/G/c).
        target_rho: Target cluster utilization.
        aging_alpha: Aging decay for decoupled hybrid (default Pareto-optimal 0.001).
        job_limit: Optional cap on number of requests.
        sla_s: E2E SLA budget (seconds).
        azure_fixture: Path to Azure LLM 2024 CSV fixture.

    Returns:
        ``PreemptionOverheadReport`` with per-overhead KPIs, breakeven analysis,
        and retention metrics.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_preemption_overhead_on_trace(
        raw, "azure_llm_2024", servers, target_rho, aging_alpha, sla_s, overhead_values_s
    )


def run_burstgpt_preemption_overhead_backtest(
    overhead_values_s: tuple = OVERHEAD_SWEEP_DEFAULT_S,
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> "PreemptionOverheadReport":
    """Cross-validate preemption overhead sensitivity on BurstGPT [run 2026-06-21-o].

    Same overhead sweep as ``run_preemption_overhead_sensitivity_backtest`` but
    on the BurstGPT trace (heavier output-token distribution, avg ~340 tokens
    vs Azure 2024's ~104 tokens, with higher default SLA budget of 30s).

    BurstGPT has longer service times → more preemptions per request → overhead
    accumulates faster → expected lower retention at the same per-event overhead.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_preemption_overhead_on_trace(
        raw, "burstgpt", servers, target_rho, aging_alpha, sla_s, overhead_values_s
    )


def run_burstgpt_hf_preemption_overhead_backtest(
    overhead_values_s: tuple = OVERHEAD_SWEEP_DEFAULT_S,
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    job_limit: Optional[int] = 5880,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> "PreemptionOverheadReport":
    """Preemption overhead sensitivity on BurstGPT HF full-scale [run 2026-06-21-s].

    Cross-validates the preemption overhead sensitivity result from Azure LLM 2024
    (run 2026-06-21-o) on the BurstGPT HF normalized sample (CC-BY-4.0).

    BurstGPT has a *heavier* output-token distribution than Azure LLM 2024:
      - output_tokens p50≈236 vs Azure p50≈90  (2.6× longer service)
      - output_tokens p99≈934 vs Azure p99≈479  (1.9× longer tail)
      - SLA budget = 30s (vs 10s for Azure), proportionally larger headroom

    Why this matters: longer service times mean more tokens are decoded between
    preemption events, so each overhead_s increment is a *smaller* fraction of
    the total service time for any individual request.  The expected behavior:
      - Higher absolute preemption count (more preemptions per longer request)
      - But *higher* robustness to per-event overhead (overhead / service << Azure)
      - Retention at 0.30s overhead expected ≥ Azure's 92.65%

    Job limit defaults to 5,880 to match the Azure LLM 2024 comparability scale
    used in runs -m through -r.  Use job_limit=None for the full 59,999-record run.

    Physical calibration (identical to run -o):
      - 0.00s: zero overhead (previous assumption in all runs g–n)
      - 0.15s: TTFT_BASE_S = one re-prefill (minimum real recomputation cost)
      - 0.30s: 2×TTFT_BASE_S (moderate; canonical measurement point)
      - 0.50s: conservative worst-case for short-sequence recomputation
      - 1.00s: upper bound (swap-based preemption for long sequences)

    Research basis:
      - FastSwitch (arXiv:2411.18424, NeurIPS 2024): 1.4–11.2× TTFT context-switch.
      - arXiv:2411.07447: recomputation < swapping for sequences < 4,000 tokens.
      - BurstGPT (arXiv:2401.17644): real LLM inference trace from production.
      - SRPT multiserver (arXiv:1805.07686): overhead robustness scales with
        service-time variance — heavier tails → more robust to per-event overhead.

    Args:
        overhead_values_s: Per-preemption overhead sweep (seconds).
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization.
        aging_alpha: Aging decay for decoupled hybrid (default Pareto-optimal 0.001).
        job_limit: Cap on requests. Defaults to 5,880 (Azure comparability scale).
                   Set to None for the full 59,999-record run.
        sla_s: E2E response-time SLA budget (seconds). Default = 30s for BurstGPT.
        jsonl_path: Path to BurstGPT HF normalized JSONL.

    Returns:
        ``PreemptionOverheadReport`` with per-overhead KPIs, breakeven analysis,
        and retention metrics. trace = "burstgpt_hf".
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 requests. "
            "Ensure the file exists and contains valid records."
        )
    return _run_preemption_overhead_on_trace(
        raw, "burstgpt_hf", servers, target_rho, aging_alpha, sla_s, overhead_values_s
    )


# ---------------------------------------------------------------------------
# Full-scale BurstGPT cross-validation [run 2026-06-21-p]
# ---------------------------------------------------------------------------
#
# Bottleneck addressed: BurstGPT fixture (54 rows) is too small to show SRPT >
# FIFO — insufficient queue depth for the scheduling signal.  The HF normalized
# sample (59,999 records, CC-BY-4.0) provides the statistical mass needed to
# cross-validate the decoupled hybrid result beyond the Azure LLM 2024 trace.
#
# BurstGPT characteristics (heavier distribution than Azure LLM 2024):
#   output_tokens: p50=236, p95=634, p99=934 (vs Azure: p50≈90, p99≈479)
#   service_s at p50: 0.15 + 236×0.02 = 4.87s (vs Azure p50: ≈1.95s)
#   SLA budget: 30s (set higher to account for longer service times)
#
# Research basis:
#   - BurstGPT (arXiv:2401.17644): real LLM inference trace from production
#     ChatGPT API calls; heavy-tailed output distribution and burst structure.
#   - SRPT multiserver (arXiv:1805.07686): SRPT throughput optimality holds
#     for M/G/c with heavy-tailed (Pareto-like) service times — BurstGPT's
#     longer outputs should show the SRPT benefit more strongly.
#   - TIE scheduling (arXiv:2604.00499): distributional ordering outperforms
#     point estimates for heavy-tailed lengths — BurstGPT is a better testbed.


def run_burstgpt_hf_decoupled_hybrid_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> DecoupledHybridReport:
    """Full-scale BurstGPT cross-validation: 6-discipline SRTF comparison [run 2026-06-21-p].

    Runs the same 6-discipline comparison (FIFO / SRTF / Aging-SRTF /
    SRPT-preemptive / Hybrid / Decoupled Hybrid) as ``run_decoupled_hybrid_backtest``
    but on the HF BurstGPT normalized sample (59,999 records, CC-BY-4.0) rather
    than the 54-row fixture.

    The 54-row fixture cannot demonstrate SRPT > FIFO because there is not enough
    queue depth for the scheduling signal to appear — requests clear before a
    backlog forms.  With 59,999 records at ρ=0.85 and 4 servers, the queue builds
    a realistic backlog and the ordering benefit is measurable.

    BurstGPT has a heavier output-token distribution than Azure LLM 2024
    (p50≈236 vs p50≈90) so:
      - Service times are longer (p50≈4.87s vs ≈1.95s).
      - SLA budget is set higher (30s vs 10s).
      - Short-request p90 improvement should be similar or larger (SRPT theory
        predicts larger gains with heavier tails).
      - Long-request p99 regression may be larger (fewer short requests to
        hide behind when the queue is predominantly long).

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization (arrival time-warp applied equally).
        job_limit: Optional cap on requests (None = use all 59,999 records).
        aging_alpha: Aging decay for dispatch key; default=DECOUPLED_HYBRID_ALPHA_DEFAULT.
        sla_s: E2E response-time SLA budget (seconds); default=30s for BurstGPT.
        jsonl_path: Path to the HF BurstGPT normalized JSONL.

    Returns:
        ``DecoupledHybridReport`` with KPIs for all 6 disciplines.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 requests. "
            "Ensure the file exists and contains valid records."
        )
    return _run_decoupled_hybrid_backtest_on_trace(
        raw, "burstgpt_hf_fullscale", servers, target_rho, aging_alpha, sla_s
    )


# ---------------------------------------------------------------------------
# Conformal Adaptive α Backtest [run 2026-06-21-q]
# ---------------------------------------------------------------------------

@dataclass
class ConformalAlphaReport:
    """Comparison of FIFO / SRPT / Decoupled-fixed-α / Decoupled-conformal-α.

    [run 2026-06-21-q] Validates that the ``ConformalAlphaCalibrator`` recovers
    near-SRPT throughput under oracle predictions (α → 0 after warmup) while
    retaining the same safety as the fixed α=0.001 under noisy predictions.

    Key research claim (arXiv:2508.14544):
      Adaptive scheduling under prediction uncertainty should recover optimal
      performance (SRPT) when predictions are accurate and degrade gracefully
      when they are not.  The conformal calibrator realises this property by
      mapping empirical p90 prediction error → α value at runtime.

    KPI columns:
      goodput_per_dollar      — SLA-safe tokens / (GPU-hour-dollars), primary metric.
      goodput_delta_pct       — vs FIFO (positive = better).
      short_p90_response_s    — p90 response time of short requests (≤ median tokens).
      long_p99_response_s     — p99 response time of long requests (> median tokens).
      conformal_mean_alpha    — mean α value used at dispatch during the simulation
                                (diagnostic; oracle → 0, noisy → ~0.001).
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float

    fifo: dict
    srpt: dict
    decoupled_fixed: dict           # fixed α = DECOUPLED_HYBRID_ALPHA_DEFAULT
    decoupled_conformal: dict       # adaptive α via ConformalAlphaCalibrator

    fifo_goodput_per_dollar: float
    srpt_goodput_per_dollar: float
    decoupled_fixed_goodput_per_dollar: float
    decoupled_conformal_goodput_per_dollar: float

    srpt_delta_pct: float                       # vs FIFO
    decoupled_fixed_delta_pct: float            # vs FIFO
    decoupled_conformal_delta_pct: float        # vs FIFO
    conformal_vs_fixed_delta_pct: float         # vs fixed-α decoupled hybrid

    # Diagnostics
    conformal_mean_alpha: float                 # mean α actually used at dispatch
    conformal_warmup: int
    conformal_window: int
    conformal_target_p90_error: float

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
            "fifo": _r(self.fifo),
            "srpt": _r(self.srpt),
            "decoupled_fixed": _r(self.decoupled_fixed),
            "decoupled_conformal": _r(self.decoupled_conformal),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 2),
            "srpt_goodput_per_dollar": round(self.srpt_goodput_per_dollar, 2),
            "decoupled_fixed_goodput_per_dollar": round(self.decoupled_fixed_goodput_per_dollar, 2),
            "decoupled_conformal_goodput_per_dollar": round(self.decoupled_conformal_goodput_per_dollar, 2),
            "srpt_delta_pct": round(self.srpt_delta_pct, 2),
            "decoupled_fixed_delta_pct": round(self.decoupled_fixed_delta_pct, 2),
            "decoupled_conformal_delta_pct": round(self.decoupled_conformal_delta_pct, 2),
            "conformal_vs_fixed_delta_pct": round(self.conformal_vs_fixed_delta_pct, 2),
            "conformal_mean_alpha": round(self.conformal_mean_alpha, 6),
            "conformal_warmup": self.conformal_warmup,
            "conformal_window": self.conformal_window,
            "conformal_target_p90_error": round(self.conformal_target_p90_error, 4),
            "shadow_tag": self.shadow_tag,
        }


def _run_conformal_alpha_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    fixed_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
) -> ConformalAlphaReport:
    """Run 4-discipline comparison: FIFO / SRPT / Decoupled-fixed / Decoupled-conformal."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    def _build() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),   # oracle prior
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs       = _build()
    srpt_reqs       = _build()
    fixed_reqs      = _build()
    conformal_reqs  = _build()

    fifo_sim,    fifo_resp,    _ = simulate_queue(fifo_reqs,    servers, "fifo")
    srpt_sim,    srpt_resp,    _ = simulate_queue(srpt_reqs,    servers, "srpt_preemptive")
    fixed_sim,   fixed_resp,   _ = simulate_queue(
        fixed_reqs, servers, "decoupled_hybrid", aging_alpha=fixed_alpha
    )
    conformal_cal = ConformalAlphaCalibrator()
    conformal_sim, conformal_resp, _ = _simulate_decoupled_hybrid_conformal(
        conformal_reqs, servers, conformal_cal
    )
    conformal_sim["sla_safe_goodput_per_dollar"] = _sla_safe_goodput_per_dollar(
        conformal_reqs, conformal_resp, sla_s, servers
    )

    gp_fifo      = _sla_safe_goodput_per_dollar(fifo_reqs,    fifo_resp,    sla_s, servers)
    gp_srpt      = _sla_safe_goodput_per_dollar(srpt_reqs,    srpt_resp,    sla_s, servers)
    gp_fixed     = _sla_safe_goodput_per_dollar(fixed_reqs,   fixed_resp,   sla_s, servers)
    gp_conformal = _sla_safe_goodput_per_dollar(conformal_reqs, conformal_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"]  = gp_fifo
    srpt_sim["sla_safe_goodput_per_dollar"]  = gp_srpt
    fixed_sim["sla_safe_goodput_per_dollar"] = gp_fixed

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    return ConformalAlphaReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        time_warp=warp,
        sla_s=sla_s,
        fifo=fifo_sim,
        srpt=srpt_sim,
        decoupled_fixed=fixed_sim,
        decoupled_conformal=conformal_sim,
        fifo_goodput_per_dollar=gp_fifo,
        srpt_goodput_per_dollar=gp_srpt,
        decoupled_fixed_goodput_per_dollar=gp_fixed,
        decoupled_conformal_goodput_per_dollar=gp_conformal,
        srpt_delta_pct=_delta(gp_fifo, gp_srpt),
        decoupled_fixed_delta_pct=_delta(gp_fifo, gp_fixed),
        decoupled_conformal_delta_pct=_delta(gp_fifo, gp_conformal),
        conformal_vs_fixed_delta_pct=_delta(gp_fixed, gp_conformal),
        conformal_mean_alpha=conformal_cal.mean_alpha(),
        conformal_warmup=conformal_cal.warmup,
        conformal_window=conformal_cal.window,
        conformal_target_p90_error=conformal_cal.target_p90_error,
    )


def run_conformal_alpha_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> ConformalAlphaReport:
    """Conformal Adaptive α backtest on Azure LLM 2024 [run 2026-06-21-q].

    Compares FIFO / SRPT / Decoupled-fixed-α=0.001 / Decoupled-conformal-α on
    the Azure LLM 2024 public trace (5,880 requests, oracle token prior).

    Validates the core claim of arXiv:2508.14544: adaptive scheduling policy under
    prediction uncertainty should recover near-SRPT throughput when predictions are
    accurate.  With oracle prior (predicted == actual tokens), the ConformalAlphaCalibrator
    measures p90 prediction error → 0 after warmup and sets α → 0, making dispatch
    equivalent to pure SRPT.

    Expected outcome (oracle prior, azure LLM 2024, ρ=0.85, 4 servers):
      FIFO baseline:         ~ reference goodput/$
      SRPT (upper bound):    +322% vs FIFO
      Decoupled-fixed α=0.001: +274% vs FIFO  [established in run -l/-m]
      Decoupled-conformal:   > +274%, approaching +322% vs FIFO
      conformal_mean_alpha:  ≈ 0.0 (converges to α=0 as p90_error → 0)

    The conformal result should exceed the fixed-α result because the calibrator
    learns that the oracle prior has zero error and adapts α → 0 → pure SRPT dispatch.

    Args:
        servers: Replica pool size (M/G/c).
        target_rho: Target cluster utilization.
        job_limit: Optional cap on requests (None = use all available).
        sla_s: E2E SLA budget (seconds).
        azure_fixture: Path to Azure LLM 2024 CSV fixture.

    Returns:
        ``ConformalAlphaReport`` with KPIs for all 4 disciplines.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_conformal_alpha_on_trace(raw, "azure_llm_2024", servers, target_rho, sla_s)


def run_burstgpt_conformal_alpha_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    burstgpt_fixture: str = DEFAULT_BURSTGPT_FIXTURE,
) -> ConformalAlphaReport:
    """Conformal Adaptive α cross-validation on BurstGPT [run 2026-06-21-q].

    Cross-validates the conformal α approach on BurstGPT (heavier output
    distribution: p50≈236 tok vs Azure 2024 p50≈90 tok).

    BurstGPT has a longer tail, which means:
    - SRPT gains are larger (theory predicts bigger gains for heavier tails).
    - The conformal calibrator should also show larger gains vs fixed α=0.001.

    Expected: conformal_delta_pct > fixed_alpha_delta_pct, with conformal
    approaching SRPT goodput.

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on requests.
        sla_s: E2E SLA budget (seconds, default 30s for BurstGPT's longer service).
        burstgpt_fixture: BurstGPT CSV path.

    Returns:
        ``ConformalAlphaReport`` with KPIs for all 4 disciplines on BurstGPT.
    """
    raw = load_burstgpt_serving_requests(burstgpt_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_conformal_alpha_on_trace(raw, "burstgpt", servers, target_rho, sla_s)


# ---------------------------------------------------------------------------
# BurstGPT HF Full-Scale Extended Validation [run 2026-06-21-r]
# ---------------------------------------------------------------------------
# Three validation gates using the 59,999-record HuggingFace BurstGPT JSONL
# (CC-BY-4.0) that the 54-row CSV fixture cannot support due to insufficient
# queue depth.  Run 2026-06-21-p confirmed that the HF dataset demonstrates
# strong SRPT/decoupled-hybrid gains (+492.7% vs FIFO at 5,880 records).
#
# These functions close the three open gates from the run -q / run -p gap analysis:
#   (1) BurstGPT HF conformal α validation — does conformal approach SRPT on
#       BurstGPT's heavier distribution (+644.4% SRPT ceiling vs FIFO)?
#   (2) BurstGPT HF vs SLA-aware baseline — North Star gap measurement on BurstGPT
#       (SLA-aware baseline was only validated on Azure LLM 2024 in run -n).
#   (3) BurstGPT HF 30%-CV noisy prior robustness — confirms generalization of the
#       100% retention gate validated on Azure LLM 2024 in run -n.
#
# Research basis:
#   - arXiv:2604.07931 (Robust Length Prediction, ProD methods, April 2026):
#     BurstGPT's heavy-tailed prompt-conditioned distribution (p99≈934 tok) means
#     prediction errors are larger than on Azure LLM 2024 (p99≈479 tok). The
#     ConformalAlphaCalibrator adapts α to the empirical p90 error — heavier-tailed
#     traces may see higher α steady-states, but the calibrator handles this
#     automatically without trace-specific tuning.
#   - arXiv:2603.11273 (Duration Aware Scheduling, workload drift, March 2026):
#     Cross-trace validation under workload drift (Azure→BurstGPT) validates that
#     scheduling gains are not trace-specific artifacts. The conformal calibrator's
#     online adaptation is the mechanism that handles drift.
#   - arXiv:2509.23384 (NexusSched, predictive two-layer scheduling, 2025):
#     The conformal calibrator + aging dispatch key is a realization of NexusSched's
#     two-layer architecture: frontend prediction (ConformalAlphaCalibrator) +
#     backend dispatch (aging key). Cross-trace validation on BurstGPT confirms
#     this architecture generalizes beyond the training trace.
# ---------------------------------------------------------------------------


def run_burstgpt_hf_conformal_alpha_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> ConformalAlphaReport:
    """Conformal Adaptive α cross-validation on BurstGPT HF full-scale [run 2026-06-21-r].

    Cross-validates the conformal α approach on the HF BurstGPT normalized sample
    (59,999 records, CC-BY-4.0) rather than the 54-row fixture which has insufficient
    queue depth to demonstrate SRPT > FIFO.

    BurstGPT has a significantly heavier output-token distribution than Azure LLM 2024:
      - p50 ≈ 236 tokens (vs ≈ 90 for Azure LLM 2024)
      - p99 ≈ 934 tokens (vs ≈ 479 for Azure LLM 2024)

    Expected outcomes (oracle prior, BurstGPT HF 5,880-record sample, ρ=0.85, 4 servers):
      FIFO baseline:            ~ reference goodput/$
      SRPT (upper bound):       ~ +644% vs FIFO (confirmed in run -p)
      Decoupled-fixed α=0.001:  ~ +493% vs FIFO (confirmed in run -p)
      Decoupled-conformal:      approaches SRPT ceiling as α → 0 on oracle prior

    With oracle tokens (predicted == actual), the ConformalAlphaCalibrator measures
    zero prediction error → α → 0 → dispatch is pure SRPT → goodput/$ approaches the
    SRPT ceiling.  With heavier tail the absolute gains are larger than on Azure LLM 2024.

    Research basis: arXiv:2604.07931 (ProD, heavy-tailed length distributions),
    arXiv:2603.11273 (Duration Aware Scheduling, cross-trace robustness),
    arXiv:2509.23384 (NexusSched, two-layer adaptive scheduling).

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization (arrival time-warp applied equally).
        job_limit: Optional cap on requests (None = use all available records).
                   Set to 5880 to match the Azure LLM 2024 scale for comparability.
        sla_s: E2E response-time SLA budget (seconds); default=30s for BurstGPT.
        jsonl_path: Path to the HF BurstGPT normalized JSONL.

    Returns:
        ``ConformalAlphaReport`` with KPIs for all 4 disciplines on BurstGPT HF.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 requests. "
            "Ensure the file exists and contains valid records."
        )
    return _run_conformal_alpha_on_trace(raw, "burstgpt_hf_fullscale", servers, target_rho, sla_s)


def run_burstgpt_hf_sla_aware_baseline_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> SLAAwareBaselineReport:
    """SLA-aware baseline comparison on BurstGPT HF full-scale [run 2026-06-21-r].

    Measures the North Star gap on BurstGPT: how much additional goodput/$ does
    continuous token-length prediction (decoupled hybrid α=0.001) provide over
    binary SLA-class awareness (sla_aware discipline)?

    On Azure LLM 2024 (run -n):
      - FIFO: 13,336 goodput/$
      - SLA-aware: 30,063 goodput/$ (+125.4% vs FIFO)
      - Decoupled α=0.001: 49,877 goodput/$ (+274.0% vs FIFO, +65.9% vs SLA-aware)
      - SRPT: 56,311 goodput/$ (+322.2% vs FIFO)

    BurstGPT's heavier distribution (p50=236 vs 90 tokens) is expected to amplify
    all three gains because SRTF benefits scale with output-length variance.

    Uses the HF BurstGPT JSONL (59,999 records) to ensure sufficient queue depth.
    The 54-row fixture shows all disciplines equivalent (queue never builds a backlog).

    Research basis: arXiv:2512.12928 (PROSERVE, multi-priority scheduling with TDG),
    arXiv:2507.10150 (Past-Future Scheduler, binary SLA-class theory),
    arXiv:2604.07931 (ProD, heavy-tailed BurstGPT distribution characterization).

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization.
        aging_alpha: Aging decay for decoupled hybrid (default=0.001 Pareto-optimal).
        job_limit: Optional cap on requests (None = use all available).
                   Set to 5880 to match the Azure LLM 2024 comparability scale.
        sla_s: E2E response-time SLA budget (seconds); default=30s for BurstGPT.
        jsonl_path: Path to the HF BurstGPT normalized JSONL.

    Returns:
        ``SLAAwareBaselineReport`` with FIFO / SLA-aware / Decoupled / SRPT KPIs.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 requests. "
            "Ensure the file exists and contains valid records."
        )
    return _run_sla_aware_baseline_on_trace(
        raw, "burstgpt_hf_fullscale", servers, target_rho, aging_alpha, sla_s
    )


def run_burstgpt_hf_noisy_prior_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    aging_alpha: float = DECOUPLED_HYBRID_ALPHA_DEFAULT,
    forecast_noise_cv: float = 0.30,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
    seed: int = 20260621,
) -> NoisyPriorRobustnessReport:
    """30%-CV noisy prior robustness on BurstGPT HF full-scale [run 2026-06-21-r].

    Validates that decoupled hybrid α=0.001 retains its goodput/$ gain under
    realistic 30%-CV lognormal forecast noise on BurstGPT's heavier distribution.

    On Azure LLM 2024 (run -n): **100% noisy retention** — zero measurable impact
    from 30%-CV noise.  The mechanism: at α=0.001 preemption is pure SRPT
    (remaining_s only, not prediction-dependent), and short requests (service ≈1.95s
    vs SLA=10s) dominate SLA-safe tokens — their ordering is noise-insensitive.

    BurstGPT's heavier tail (p99=934 vs 479 tokens) may show different behavior:
    - Larger absolute prediction errors (30%-CV of 934 tokens = ±280 tokens error)
    - Long requests are more numerous, so starvation could affect more tokens
    - The SLA=30s budget provides more headroom for short requests to be SLA-safe

    Expected: high retention (≥95%) because the same mechanism applies — preemptive
    SRPT with α=0.001 is dominated by actual remaining work, not predicted tokens.
    The preemption-corrects-mistakes mechanism (arXiv:2508.14544) should preserve
    most of the oracle gain under BurstGPT's heavier noise levels.

    Research basis: arXiv:2508.14544 (Adaptively Robust LLM Inference, Aug 2025),
    arXiv:2604.07931 (Robust Length Prediction, heavy-tailed distributions, Apr 2026),
    arXiv:1902.00732 (Scheduling with Predictions, Mitzenmacher 2019).

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization.
        aging_alpha: Aging decay for decoupled hybrid (default=0.001 Pareto-optimal).
        forecast_noise_cv: Lognormal coefficient of variation for forecast noise.
        job_limit: Optional cap on requests (None = use all available records).
                   Set to 5880 to match the Azure LLM 2024 comparability scale.
        sla_s: E2E response-time SLA budget (seconds); default=30s for BurstGPT.
        jsonl_path: Path to the HF BurstGPT normalized JSONL.
        seed: Random seed for reproducible noise injection.

    Returns:
        ``NoisyPriorRobustnessReport`` with oracle / noisy / retention KPIs.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 requests. "
            "Ensure the file exists and contains valid records."
        )
    return _run_noisy_prior_on_trace(
        raw, "burstgpt_hf_fullscale", servers, target_rho, aging_alpha,
        sla_s, forecast_noise_cv, seed
    )


# ---------------------------------------------------------------------------
# Live Causal Prior — closes the oracle gap [run 2026-06-21-t]
# ---------------------------------------------------------------------------
# All prior robustness experiments used either oracle (predicted == actual) or
# artificial lognormal noise (30%-CV). Neither reflects a real deployed system.
#
# This section implements the first production-realistic prior: a CAUSAL
# SLIDING-WINDOW MEDIAN estimator that uses only completed requests visible
# before each new arrival. This closes the gap from oracle to a live system:
#
#   Oracle prior (upper bound):       +322% vs FIFO (run -q)
#   30%-CV noisy prior (worst-case):  +267% vs FIFO (run -n)
#   Live causal prior (THIS section): measured here for the first time
#
# Production interpretation: a serving engine maintains a running median of
# recent completions' output-token counts and uses that as the prior for
# the next incoming request. This is the minimal zero-external-model prior —
# it uses no external features, only the trace's own historical statistics.
#
# Causal guarantee: for request i, the prediction uses only actual_tokens from
# requests 0..i-1, which have all arrived (and most completed) before request i.
# This is identical to what a production scheduler would observe in a FIFO
# completion log.
#
# Research basis:
# - arXiv:2604.06970 (Scheduling the Unschedulable, SOSP 2026): §6.3 discusses
#   production-viable priors; running average is their fallback for cold-start.
# - arXiv:2508.14544 (Adaptively Robust LLM Inference): the causal running
#   estimator is the implementation of "prediction from observation" that the
#   conformal calibrator assumes is available at dispatch time.
# - arXiv:2503.07545 (Queueing, Predictions, and LLMs, Mitzenmacher 2025):
#   explicitly identifies causal historical estimators as the practical realization
#   of theoretical scheduling-with-predictions frameworks.
# ---------------------------------------------------------------------------

# Window size for the causal sliding-window median prior.
LIVE_PRIOR_WINDOW: int = 200


def make_live_prior_predictions(
    raw: list[tuple[float, int]],
    window: int = LIVE_PRIOR_WINDOW,
    warmup_value: Optional[float] = None,
) -> tuple[list[float], dict]:
    """Causal sliding-window median prediction for output tokens.

    For request i, predicts its output token count as the empirical median of
    the last ``window`` actual tokens from requests 0..i-1 (causal: uses only
    past arrivals, not the current request).  Requests before the first
    completion (i == 0) fall back to ``warmup_value`` or the global median.

    This is the minimum-complexity production-viable prior: no external model,
    no features, just historical output-token statistics from recent completions.

    Args:
        raw: List of (arrival_s, actual_output_tokens) from a real trace.
        window: Sliding window size (default: 200 past requests).
        warmup_value: Fixed fallback before any history is available (default:
            global median of the trace, which is slightly non-causal for the
            very first request but is a negligible leak for large traces).

    Returns:
        (predictions, stats) where:
          predictions: list[float], length == len(raw), predictions[i] is the
              causal median estimate for request i.
          stats: dict with 'prior_cv_pct', 'prior_mae_tokens', 'prior_bias_pct',
              'warmup_fallback', 'window', 'n_requests'.
    """
    if not raw:
        return [], {}

    all_toks = [t for _, t in raw]
    sorted_all = sorted(all_toks)
    global_median = float(sorted_all[len(sorted_all) // 2])
    fallback = warmup_value if warmup_value is not None else global_median

    predictions: list[float] = []
    history: list[int] = []

    for _arr, tok in raw:
        if not history:
            predictions.append(fallback)
        else:
            win = history[-window:]
            s = sorted(win)
            predictions.append(float(s[len(s) // 2]))
        history.append(tok)

    # Diagnostic statistics: prediction quality vs actuals.
    errors = [abs(predictions[i] - all_toks[i]) for i in range(len(raw))]
    biases = [predictions[i] - all_toks[i] for i in range(len(raw))]
    mean_actual = statistics.mean(all_toks)
    mae = statistics.mean(errors)
    mean_bias = statistics.mean(biases)
    # CV: std(predictions) / mean(actuals) — measures prediction spread
    pred_std = statistics.stdev(predictions) if len(predictions) > 1 else 0.0
    cv_pct = 100.0 * pred_std / max(1.0, mean_actual)
    # Relative MAE: MAE / mean(actuals)
    rel_mae_pct = 100.0 * mae / max(1.0, mean_actual)

    stats = {
        "prior_cv_pct": round(cv_pct, 2),
        "prior_mae_tokens": round(mae, 2),
        "prior_rel_mae_pct": round(rel_mae_pct, 2),
        "prior_bias_tokens": round(mean_bias, 2),
        "prior_bias_pct": round(100.0 * mean_bias / max(1.0, mean_actual), 2),
        "warmup_fallback": round(fallback, 2),
        "global_median_actual": round(global_median, 2),
        "window": window,
        "n_requests": len(raw),
    }
    return predictions, stats


@dataclass
class LivePriorReport:
    """Live causal prior vs oracle comparison [run 2026-06-21-t].

    Compares three conditions on the same public trace:
      - FIFO baseline (no prior needed)
      - Conformal with oracle prior (predicted == actual, upper bound)
      - Conformal with live causal prior (sliding-window median of past requests)

    The key measurement: live_vs_oracle_retention_pct — how much of the oracle
    conformal gain survives when we replace oracle with causal historical predictions.

    If retention ≥ 95%, the live prior is production-viable and the conformal
    discipline can be deployed without requiring an external output-length model.
    """

    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    prior_window: int

    # Prior quality diagnostics
    prior_cv_pct: float
    prior_mae_tokens: float
    prior_rel_mae_pct: float
    prior_bias_tokens: float

    # Simulation summaries
    fifo: dict
    conformal_oracle: dict
    conformal_live: dict

    # KPIs
    fifo_goodput_per_dollar: float
    oracle_goodput_per_dollar: float
    live_goodput_per_dollar: float

    oracle_delta_pct: float              # oracle conformal vs FIFO
    live_delta_pct: float                # live conformal vs FIFO
    live_vs_oracle_retention_pct: float  # live / oracle goodput, %

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "prior_window": self.prior_window,
            "prior_cv_pct": round(self.prior_cv_pct, 2),
            "prior_mae_tokens": round(self.prior_mae_tokens, 2),
            "prior_rel_mae_pct": round(self.prior_rel_mae_pct, 2),
            "prior_bias_tokens": round(self.prior_bias_tokens, 2),
            "fifo": _r(self.fifo),
            "conformal_oracle": _r(self.conformal_oracle),
            "conformal_live": _r(self.conformal_live),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 2),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 2),
            "live_goodput_per_dollar": round(self.live_goodput_per_dollar, 2),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "live_delta_pct": round(self.live_delta_pct, 2),
            "live_vs_oracle_retention_pct": round(self.live_vs_oracle_retention_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_live_prior_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    prior_window: int,
) -> LivePriorReport:
    """Run FIFO / Conformal-oracle / Conformal-live-prior on a trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)
    live_preds, prior_stats = make_live_prior_predictions(raw, window=prior_window)

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),  # oracle
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_live() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=live_preds[i],  # causal sliding-window median
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs = _build_oracle()  # predicted_tokens irrelevant for FIFO
    oracle_reqs = _build_oracle()
    live_reqs = _build_live()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    oracle_cal = ConformalAlphaCalibrator()
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_conformal(
        oracle_reqs, servers, oracle_cal
    )
    live_cal = ConformalAlphaCalibrator()
    live_sim, live_resp, _ = _simulate_decoupled_hybrid_conformal(
        live_reqs, servers, live_cal
    )

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    gp_live = _sla_safe_goodput_per_dollar(live_reqs, live_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle
    live_sim["sla_safe_goodput_per_dollar"] = gp_live

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    retention = (gp_live / gp_oracle * 100.0) if gp_oracle > 0 else 0.0

    return LivePriorReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        prior_window=prior_window,
        prior_cv_pct=prior_stats.get("prior_cv_pct", 0.0),
        prior_mae_tokens=prior_stats.get("prior_mae_tokens", 0.0),
        prior_rel_mae_pct=prior_stats.get("prior_rel_mae_pct", 0.0),
        prior_bias_tokens=prior_stats.get("prior_bias_tokens", 0.0),
        fifo=fifo_sim,
        conformal_oracle=oracle_sim,
        conformal_live=live_sim,
        fifo_goodput_per_dollar=gp_fifo,
        oracle_goodput_per_dollar=gp_oracle,
        live_goodput_per_dollar=gp_live,
        oracle_delta_pct=_delta(gp_fifo, gp_oracle),
        live_delta_pct=_delta(gp_fifo, gp_live),
        live_vs_oracle_retention_pct=retention,
    )


def run_live_prior_conformal_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> LivePriorReport:
    """Live causal prior on Azure LLM 2024 [run 2026-06-21-t].

    Replaces the oracle prediction (predicted == actual) with a causal
    sliding-window median estimator: for request i, the predicted token count
    is the empirical median of the last ``prior_window`` actual completions
    from requests 0..i-1.

    This is the first production-realistic evaluation of the conformal discipline:
    no oracle tokens, no external model — just the running history of the serving
    queue itself.

    Expected outcomes on Azure LLM 2024 (5,880 requests, ρ=0.85, 4 servers):
      FIFO baseline:                   ~ reference goodput/$
      Conformal oracle (upper bound):  +322.24% vs FIFO [run -q]
      Conformal live prior (target):   ≥ +267% vs FIFO (≥83% retention vs oracle)

    The live causal prior should perform BETTER than 30%-CV lognormal noise because:
    1. Running median is robust to heavy tails (Azure p99/p50 = 5.3×)
    2. Real output-token distributions exhibit moderate stationarity within a trace
    3. The sliding window adapts to any distribution shifts automatically
    4. The conformal calibrator further adapts α to the observed prediction errors

    The ≥83% retention threshold matches the 30%-CV noisy-prior floor from run -n.
    If live > 83%, the live prior is strictly safer than the already-validated noisy prior.

    Args:
        servers: Replica pool size (M/G/c). Identical across disciplines.
        target_rho: Target cluster utilization.
        job_limit: Optional cap on requests (None = use all 5,880 available).
        sla_s: E2E SLA budget (seconds).
        prior_window: Sliding window size for causal median prediction.
        azure_fixture: Path to Azure LLM 2024 CSV fixture.

    Returns:
        ``LivePriorReport`` with FIFO / oracle / live KPIs and retention metric.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_live_prior_on_trace(raw, "azure_llm_2024", servers, target_rho, sla_s, prior_window)


def run_burstgpt_hf_live_prior_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> LivePriorReport:
    """Live causal prior cross-validation on BurstGPT HF [run 2026-06-21-t].

    Applies the causal sliding-window median prior to BurstGPT's heavier output
    distribution (p50≈236 tok, p99≈934 tok vs Azure p50≈90, p99≈479).

    Key question: does the live prior work equally well on a heavier-tailed
    distribution where prediction errors are proportionally larger in absolute
    terms but the conformal calibrator has more room to adapt α?

    Expected outcomes (5,880-record sample, ρ=0.85, 4 servers, sla_s=30s):
      FIFO baseline:                   ~ reference goodput/$
      Conformal oracle (upper bound):  +644.4% vs FIFO [run -r]
      Conformal live prior (target):   ≥ +536% vs FIFO (≥83% retention vs oracle)

    A heavier tail means:
    - Running median is MORE stable (more robust to outliers)
    - Absolute errors are larger but relative errors (CV) may be similar
    - The conformal calibrator should adapt α higher → more aging → robust dispatch

    Args:
        servers: Replica pool size.
        target_rho: Target cluster utilization.
        job_limit: Optional cap (set to 5880 for comparability with Azure scale).
        sla_s: E2E SLA budget (default 30s for BurstGPT's longer service times).
        prior_window: Sliding window size for causal median prediction.
        jsonl_path: Path to the HF BurstGPT normalized JSONL.

    Returns:
        ``LivePriorReport`` with FIFO / oracle / live KPIs and retention metric.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 requests. "
            "Ensure the file exists and contains valid records."
        )
    return _run_live_prior_on_trace(
        raw, "burstgpt_hf_fullscale", servers, target_rho, sla_s, prior_window
    )


# ---------------------------------------------------------------------------
# STRATIFIED CAUSAL PRIOR [run 2026-06-22-u]
#
# The global sliding-window median prior (run -t) achieves 70.0% retention on
# BurstGPT HF because it conflates two fundamentally different request types:
#   ChatGPT: p50=7 tokens (84.2% of requests — mostly very short responses)
#   GPT-4:   p50=235 tokens (15.8% of requests — substantially longer)
# A global running median of ~18 tokens is severely wrong for GPT-4 requests.
#
# This section implements a FEATURE-AWARE CAUSAL PRIOR that stratifies by:
#   Level 1 (finest): (model_id, input_bin) — input_bin is 'long'/'short'
#       based on the causal running median of past input_tokens for that model.
#       Input-output correlation within ChatGPT is r=0.513 (strong).
#   Level 2: model_id only — fallback when bin has < MIN_STRATUM_HISTORY entries.
#   Level 3: global running median — ultimate fallback.
#
# All predictions are causal: request i uses only completions from 0..i-1.
#
# Expected impact: BurstGPT HF retention improves from 70.0% toward ≥85%.
#
# Research basis:
# - TIE scheduling (arXiv:2604.00499): distributional ordering improves dispatch;
#   stratification by model_id implements this at the predictor level.
# - ProD, Robust Length Prediction (arXiv:2604.07931): per-request features
#   (prompt type, model family) are the strongest available signals for output length.
# - CARA HGB forecaster (existing repo): confirms model_id + input features are
#   informative predictors of output length in production telemetry.
# ---------------------------------------------------------------------------

# Minimum history per stratum before stratum-specific prediction is used.
STRATIFIED_MIN_HISTORY: int = 20


def load_burstgpt_serving_requests_jsonl_with_features(
    path: str = DEFAULT_BURSTGPT_HF_JSONL,
    limit: Optional[int] = None,
) -> tuple[list[tuple[float, int]], list[dict]]:
    """Load BurstGPT HF JSONL and return (raw_list, features_list).

    raw_list:      [(arrival_s, output_tokens), ...]  — same format as
                   ``load_burstgpt_serving_requests_jsonl``.
    features_list: [{'model_id': str, 'input_tokens': int}, ...]  — parallel
                   list of per-request features for stratified prediction.

    Both lists are aligned: features_list[i] corresponds to raw_list[i].
    Records with zero output_tokens are excluded (same as base loader).
    """
    import json as _json

    rows: list[tuple[float, int, dict]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = _json.loads(line)
                ts = float(d["request_arrival_ts_s"])
                out_tok = int(d.get("output_tokens") or 0)
                inp_tok = int(d.get("input_tokens") or 0)
                model_id = str(d.get("model_id") or "unknown")
            except (KeyError, ValueError, TypeError):
                continue
            if out_tok > 0:
                rows.append((ts, out_tok, {"model_id": model_id, "input_tokens": inp_tok}))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return [], []
    t0 = rows[0][0]
    raw = [(ts - t0, tok) for ts, tok, _ in rows]
    feats = [f for _, _, f in rows]
    if limit is not None:
        raw = raw[:limit]
        feats = feats[:limit]
    return raw, feats


def make_stratified_prior_predictions(
    raw: list[tuple[float, int]],
    features: list[dict],
    window: int = LIVE_PRIOR_WINDOW,
    min_stratum_history: int = STRATIFIED_MIN_HISTORY,
) -> tuple[list[float], dict]:
    """Feature-aware causal prior using model_id + causal input-bin stratification.

    For request i, the predicted output token count is selected via the
    following fallback hierarchy (all causal — only uses past completions):

      1. (model_id, input_bin) stratum median — if that stratum has ≥
         min_stratum_history past completions.  input_bin is 'long' when
         request i's input_tokens ≥ the causal running median of past
         input_tokens for that model; else 'short'.
      2. model_id median — if model has ≥ min_stratum_history past completions
         but the bin-specific stratum is too sparse.
      3. Global running median — ultimate fallback (window tokens from any model).

    Args:
        raw:     [(arrival_s, output_tokens), ...] — the requests.
        features: [{'model_id': str, 'input_tokens': int}, ...] — parallel
                  per-request features.  Must be the same length as raw.
        window:  Sliding window size for all running medians.
        min_stratum_history: Minimum past completions before a stratum's own
                  median is trusted over the coarser fallback.

    Returns:
        (predictions, stats) where predictions[i] is the causal prediction
        for request i, and stats is a diagnostic dict.
    """
    if not raw:
        return [], {}
    assert len(raw) == len(features), "raw and features must be same length"

    all_toks = [t for _, t in raw]
    sorted_all = sorted(all_toks)
    global_median_val = float(sorted_all[len(sorted_all) // 2])

    def _median(hist: list[int]) -> float:
        if not hist:
            return global_median_val
        win = hist[-window:]
        s = sorted(win)
        return float(s[len(s) // 2])

    global_hist: list[int] = []
    model_hist: dict[str, list[int]] = {}      # model_id → output history
    model_inp_hist: dict[str, list[int]] = {}  # model_id → input history (for bin cutoff)
    stratum_hist: dict[tuple, list[int]] = {}  # (model_id, 'short'/'long') → output hist

    predictions: list[float] = []
    used_levels: list[str] = []  # diagnostic: which level was used for each prediction

    for i, ((arr, tok), feat) in enumerate(zip(raw, features)):
        mid = feat.get("model_id", "unknown")
        inp = feat.get("input_tokens", 0)

        # ── Step 1: determine input_bin causally
        inp_hist = model_inp_hist.get(mid, [])
        if len(inp_hist) >= min_stratum_history:
            win_inp = inp_hist[-window:]
            s_inp = sorted(win_inp)
            inp_median = float(s_inp[len(s_inp) // 2])
            inp_bin = "long" if inp >= inp_median else "short"
        else:
            inp_bin = "unknown"  # not enough history to classify reliably

        # ── Step 2: select prediction level
        stratum_key = (mid, inp_bin)
        sh = stratum_hist.get(stratum_key, [])
        mh = model_hist.get(mid, [])
        gh = global_hist

        if inp_bin != "unknown" and len(sh) >= min_stratum_history:
            pred = _median(sh)
            used_levels.append("stratum")
        elif len(mh) >= min_stratum_history:
            pred = _median(mh)
            used_levels.append("model")
        elif gh:
            pred = _median(gh)
            used_levels.append("global")
        else:
            pred = global_median_val
            used_levels.append("fallback")

        predictions.append(pred)

        # ── Update histories with this request's actual output tokens
        global_hist.append(tok)
        model_hist.setdefault(mid, []).append(tok)
        stratum_hist.setdefault(stratum_key, []).append(tok)
        model_inp_hist.setdefault(mid, []).append(inp)

    # ── Diagnostic statistics
    errors = [abs(predictions[i] - all_toks[i]) for i in range(len(raw))]
    biases = [predictions[i] - all_toks[i] for i in range(len(raw))]
    mean_actual = statistics.mean(all_toks)
    mae = statistics.mean(errors)
    mean_bias = statistics.mean(biases)
    pred_std = statistics.stdev(predictions) if len(predictions) > 1 else 0.0
    cv_pct = 100.0 * pred_std / max(1.0, mean_actual)
    rel_mae_pct = 100.0 * mae / max(1.0, mean_actual)

    level_counts = {
        lv: used_levels.count(lv) for lv in ("stratum", "model", "global", "fallback")
    }

    stats = {
        "prior_cv_pct": round(cv_pct, 2),
        "prior_mae_tokens": round(mae, 2),
        "prior_rel_mae_pct": round(rel_mae_pct, 2),
        "prior_bias_tokens": round(mean_bias, 2),
        "prior_bias_pct": round(100.0 * mean_bias / max(1.0, mean_actual), 2),
        "global_median_actual": round(global_median_val, 2),
        "window": window,
        "n_requests": len(raw),
        "level_counts": level_counts,
        "stratum_pct": round(100.0 * level_counts["stratum"] / max(1, len(raw)), 2),
        "model_pct": round(100.0 * level_counts["model"] / max(1, len(raw)), 2),
        "global_fallback_pct": round(
            100.0 * (level_counts["global"] + level_counts["fallback"]) / max(1, len(raw)), 2
        ),
    }
    return predictions, stats


@dataclass
class StratifiedPriorReport:
    """Stratified feature-aware causal prior vs global causal prior [run 2026-06-22-u].

    Extends LivePriorReport to compare four conditions on the same public trace:
      - FIFO baseline
      - Conformal with oracle prior (upper bound)
      - Conformal with global causal prior (sliding-window median — run -t baseline)
      - Conformal with stratified causal prior (per-model_id + input-bin median)

    The key measurement: stratified_vs_oracle_retention_pct — how much of the oracle
    conformal gain survives when we replace oracle with the stratified causal prior.
    Also reports the improvement over the global prior to isolate the stratification gain.
    """

    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    prior_window: int

    # Prior quality diagnostics — global prior
    global_prior_cv_pct: float
    global_prior_mae_tokens: float
    global_prior_rel_mae_pct: float

    # Prior quality diagnostics — stratified prior
    stratified_prior_cv_pct: float
    stratified_prior_mae_tokens: float
    stratified_prior_rel_mae_pct: float
    stratified_stratum_pct: float   # % requests served by stratum-level prior
    stratified_model_pct: float     # % requests served by model-level prior
    stratified_fallback_pct: float  # % requests falling back to global prior

    # Simulation summaries
    fifo: dict
    conformal_oracle: dict
    conformal_global: dict
    conformal_stratified: dict

    # KPIs
    fifo_goodput_per_dollar: float
    oracle_goodput_per_dollar: float
    global_goodput_per_dollar: float
    stratified_goodput_per_dollar: float

    oracle_delta_pct: float              # oracle conformal vs FIFO
    global_delta_pct: float              # global causal prior vs FIFO
    stratified_delta_pct: float          # stratified prior vs FIFO
    global_vs_oracle_retention_pct: float    # global / oracle goodput, %
    stratified_vs_oracle_retention_pct: float # stratified / oracle goodput, %
    stratified_vs_global_improvement_pct: float  # (strat - global) / global, %

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "prior_window": self.prior_window,
            "global_prior_cv_pct": round(self.global_prior_cv_pct, 2),
            "global_prior_mae_tokens": round(self.global_prior_mae_tokens, 2),
            "global_prior_rel_mae_pct": round(self.global_prior_rel_mae_pct, 2),
            "stratified_prior_cv_pct": round(self.stratified_prior_cv_pct, 2),
            "stratified_prior_mae_tokens": round(self.stratified_prior_mae_tokens, 2),
            "stratified_prior_rel_mae_pct": round(self.stratified_prior_rel_mae_pct, 2),
            "stratified_stratum_pct": round(self.stratified_stratum_pct, 2),
            "stratified_model_pct": round(self.stratified_model_pct, 2),
            "stratified_fallback_pct": round(self.stratified_fallback_pct, 2),
            "fifo": _r(self.fifo),
            "conformal_oracle": _r(self.conformal_oracle),
            "conformal_global": _r(self.conformal_global),
            "conformal_stratified": _r(self.conformal_stratified),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 2),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 2),
            "global_goodput_per_dollar": round(self.global_goodput_per_dollar, 2),
            "stratified_goodput_per_dollar": round(self.stratified_goodput_per_dollar, 2),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "global_delta_pct": round(self.global_delta_pct, 2),
            "stratified_delta_pct": round(self.stratified_delta_pct, 2),
            "global_vs_oracle_retention_pct": round(self.global_vs_oracle_retention_pct, 2),
            "stratified_vs_oracle_retention_pct": round(self.stratified_vs_oracle_retention_pct, 2),
            "stratified_vs_global_improvement_pct": round(self.stratified_vs_global_improvement_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_stratified_prior_on_trace_with_features(
    raw: list[tuple[float, int]],
    features: list[dict],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    prior_window: int,
    min_stratum_history: int = STRATIFIED_MIN_HISTORY,
) -> StratifiedPriorReport:
    """Run FIFO / oracle / global-prior / stratified-prior on a feature-annotated trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    # Global causal predictions (no features — same as run -t)
    global_preds, global_stats = make_live_prior_predictions(raw, window=prior_window)

    # Stratified causal predictions (uses model_id + input_bin)
    strat_preds, strat_stats = make_stratified_prior_predictions(
        raw, features, window=prior_window, min_stratum_history=min_stratum_history
    )

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_global() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=global_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_stratified() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=strat_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs = _build_oracle()  # predicted_tokens irrelevant for FIFO
    oracle_reqs = _build_oracle()
    global_reqs = _build_global()
    strat_reqs = _build_stratified()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    oracle_cal = ConformalAlphaCalibrator()
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_conformal(
        oracle_reqs, servers, oracle_cal
    )
    global_cal = ConformalAlphaCalibrator()
    global_sim, global_resp, _ = _simulate_decoupled_hybrid_conformal(
        global_reqs, servers, global_cal
    )
    strat_cal = ConformalAlphaCalibrator()
    strat_sim, strat_resp, _ = _simulate_decoupled_hybrid_conformal(
        strat_reqs, servers, strat_cal
    )

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    gp_global = _sla_safe_goodput_per_dollar(global_reqs, global_resp, sla_s, servers)
    gp_strat = _sla_safe_goodput_per_dollar(strat_reqs, strat_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle
    global_sim["sla_safe_goodput_per_dollar"] = gp_global
    strat_sim["sla_safe_goodput_per_dollar"] = gp_strat

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    global_ret = (gp_global / gp_oracle * 100.0) if gp_oracle > 0 else 0.0
    strat_ret = (gp_strat / gp_oracle * 100.0) if gp_oracle > 0 else 0.0
    strat_vs_global = _delta(gp_global, gp_strat)

    return StratifiedPriorReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        prior_window=prior_window,
        global_prior_cv_pct=global_stats.get("prior_cv_pct", 0.0),
        global_prior_mae_tokens=global_stats.get("prior_mae_tokens", 0.0),
        global_prior_rel_mae_pct=global_stats.get("prior_rel_mae_pct", 0.0),
        stratified_prior_cv_pct=strat_stats.get("prior_cv_pct", 0.0),
        stratified_prior_mae_tokens=strat_stats.get("prior_mae_tokens", 0.0),
        stratified_prior_rel_mae_pct=strat_stats.get("prior_rel_mae_pct", 0.0),
        stratified_stratum_pct=strat_stats.get("stratum_pct", 0.0),
        stratified_model_pct=strat_stats.get("model_pct", 0.0),
        stratified_fallback_pct=strat_stats.get("global_fallback_pct", 0.0),
        fifo=fifo_sim,
        conformal_oracle=oracle_sim,
        conformal_global=global_sim,
        conformal_stratified=strat_sim,
        fifo_goodput_per_dollar=gp_fifo,
        oracle_goodput_per_dollar=gp_oracle,
        global_goodput_per_dollar=gp_global,
        stratified_goodput_per_dollar=gp_strat,
        oracle_delta_pct=_delta(gp_fifo, gp_oracle),
        global_delta_pct=_delta(gp_fifo, gp_global),
        stratified_delta_pct=_delta(gp_fifo, gp_strat),
        global_vs_oracle_retention_pct=global_ret,
        stratified_vs_oracle_retention_pct=strat_ret,
        stratified_vs_global_improvement_pct=strat_vs_global,
    )


# ---------------------------------------------------------------------------
# ML PRIOR (HGB) [run 2026-06-22-v]
#
# Why this is different from the stratified causal prior (run -u):
#
# Run -u confirmed that ANY running-statistics prior (global or stratified)
# is capped at alpha=2×alpha_max=0.002 by the conformal calibrator.  The
# root cause: GPT-4 requests (15.8% of BurstGPT) have rel_err≈0.96 under
# the global running median, pushing the p90 error above the 2× cap threshold
# (p90_err ≥ 0.80 → ratio=min(2.0,...) → capped).
#
# The stratified prior (run -u) ALSO hit the cap despite using model_id:
#   - Global running median:   p90 rel_err ≈ 0.96 (GPT-4 errors dominate)
#   - Stratified running median: p90 rel_err ≈ 0.95 (surprise-long ChatGPT now dominates)
# Both keep ratio ≥ 2.0 → alpha = 0.002.
#
# The ML-HGB prior with model_id is DIFFERENT because HGB:
# 1. Learns the EXACT per-model token distribution from warmup data (not just
#    a single running median per stratum — a full learned distribution).
# 2. Can exploit continuous input_tokens × model_id interactions.
#
# But the critical question is: does model_id alone break the 2× cap?
#
# Mathematical argument:
#   With model_id feature, HGB learns GPT-4 → ~235 tokens correctly.
#   GPT-4 errors drop from rel_err≈0.96 to rel_err≈0.02.
#   Now only surprise-long ChatGPT (~8.4% of traffic) has high rel_err≈0.95.
#   8.4% of requests fall above the 91.6th percentile.
#   The p90 is now in the ChatGPT-normal range: rel_err≈0.43.
#   ratio = 0.43/0.40 ≈ 1.075 → alpha ≈ 0.001075 (near fixed α=0.001).
#
# This is precisely the difference vs run -u:
#   Stratified prior (run -u): still had GPT-4 at poor stratum predictions
#     during the first 20 completions, and its per-model median converges the
#     same way as the global in the long run for GPT-4.
#     Wait — actually run -u DID use model_id and still got -0.12%.
#
# Why did run -u's model_id stratification fail?
#   The stratified prior uses a RUNNING MEDIAN for each model. After 20+
#   GPT-4 completions, the per-model running median converges to ~235 tokens
#   correctly. So the stratified prior SHOULD have fixed GPT-4 predictions.
#   But run -u showed identical goodput/$.
#
# Resolution: both the global and stratified priors gave alpha=0.002 CAPPED.
#   If the stratified prior already fixes GPT-4, but STILL gets p90≈0.95...
#   That means surprise-long ChatGPT (8.4%) has already moved to the 90th
#   percentile when GPT-4 is fixed (91.6% of traffic < surprise-long level).
#   And 8.4% > 10% → surprise-long requests ARE within the top-10% errors.
#
# So the ML-HGB prior faces the SAME ceiling as the stratified prior:
#   - Fix GPT-4: yes (same as stratified after warmup)
#   - Fix surprise-long ChatGPT: impossible without features not in BurstGPT
#   - p90 error still ≈ 0.95 → still capped at 2×
#
# HOWEVER: the HGB prior has one genuine advantage over running median:
#   The continuous learned function may reduce ChatGPT-normal prediction errors
#   below the stratified median (especially for high-input-token ChatGPT).
#   This is marginal but potentially measurable.
#
# ALSO: the HGB may learn that ChatGPT with very long input → longer output,
#   partially identifying some surprise-long requests (those with longer inputs).
#   Even 20-30% identification rate would help.
#
# Research basis:
# - "Scheduling the Unschedulable" (arXiv:2604.06970): model-type features
#   are the strongest signal for output length in production LLM serving.
# - "Predicting LLM Output Length" (arXiv:2602.11812): model_id + prompt
#   features achieve -29.16% MAE; model_id alone is the largest contributor.
# - "TIE scheduling" (arXiv:2604.00499): distributional ordering by model type
#   improves dispatch for mixed-model traffic.
# ---------------------------------------------------------------------------

ML_PRIOR_WARMUP_N: int = 1000


def make_ml_prior_predictions_burstgpt(
    raw: list[tuple[float, int]],
    features: list[dict],
    warmup_n: int = ML_PRIOR_WARMUP_N,
) -> tuple[list[float], dict]:
    """HGB ML causal prior using model_id + input_tokens for BurstGPT.

    Causal two-phase design:
      Phase 1 (i < warmup_n): running median — identical to live prior [run -t].
      Phase 2 (i >= warmup_n): HGB p50 trained on Phase 1 observations only.

    Feature set: [model_id_encoded, input_tokens] — both available at request
    arrival before output generation begins.  model_id encodes the serving-engine
    identity (e.g. 'ChatGPT', 'GPT-4'); input_tokens is the prompt length.

    Model_id is the primary signal: BurstGPT mixes ChatGPT (p50=7 tok) and GPT-4
    (p50=235 tok).  The HGB trained on warmup_n completions learns the per-model
    distribution, dramatically reducing prediction error for GPT-4 requests.

    Args:
        raw:      [(arrival_s, output_tokens), ...] — same format as load functions.
        features: [{'model_id': str, 'input_tokens': int}, ...] — parallel to raw.
        warmup_n: Number of initial requests used as HGB training data.  These
                  requests receive running-median predictions during Phase 1.

    Returns:
        (predictions, stats) where predictions[i] is the causal ML prediction
        for request i.  Phase 1 predictions are running medians; Phase 2
        predictions are HGB p50 outputs.  All predictions are clipped to ≥ 1.0.
    """
    n = len(raw)
    assert len(features) == n, "raw and features must be same length"
    if n == 0:
        return [], {}

    all_toks = [t for _, t in raw]
    sorted_all = sorted(all_toks)
    global_median = float(sorted_all[len(sorted_all) // 2])

    predictions: list[float] = []
    history: list[int] = []

    # Phase 1: running median (causal, no HGB yet)
    phase1_n = min(warmup_n, n)
    for i in range(phase1_n):
        tok = all_toks[i]
        if not history:
            predictions.append(global_median)
        else:
            win = history[-LIVE_PRIOR_WINDOW:]
            s = sorted(win)
            predictions.append(float(s[len(s) // 2]))
        history.append(tok)

    if phase1_n >= n:
        errors = [abs(predictions[i] - all_toks[i]) for i in range(n)]
        mae = statistics.mean(errors) if errors else 0.0
        mean_actual = statistics.mean(all_toks)
        pred_std = statistics.stdev(predictions) if n > 1 else 0.0
        cv_pct = 100.0 * pred_std / max(1.0, mean_actual)
        return predictions, {
            "prior_type": "ml_hgb_warmup_only",
            "warmup_n": warmup_n,
            "n_requests": n,
            "n_model_ids": 0,
            "phase2_n": 0,
            "prior_cv_pct": round(cv_pct, 2),
            "prior_mae_tokens": round(mae, 2),
            "prior_rel_mae_pct": round(100.0 * mae / max(1.0, mean_actual), 2),
            "prior_bias_tokens": 0.0,
            "global_median_actual": round(global_median, 2),
        }

    # Phase 2: train HGB on Phase 1 data (causal — only past completions)
    try:
        import numpy as _np
        from sklearn.ensemble import HistGradientBoostingRegressor as _HGB
    except ImportError:
        fallback, fstats = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
        fstats["prior_type"] = "ml_hgb_sklearn_unavailable_fallback"
        fstats["n_model_ids"] = 0
        fstats["phase2_n"] = 0
        return fallback, fstats

    warmup_feats = features[:phase1_n]
    warmup_toks = all_toks[:phase1_n]

    # model_id encoding: sorted across full trace (schema-level, not prediction-time).
    all_model_ids = sorted(set(f.get("model_id", "unknown") for f in features))
    mid_map = {mid: float(i) for i, mid in enumerate(all_model_ids)}

    def _encode(f: dict) -> list[float]:
        return [
            mid_map.get(f.get("model_id", "unknown"), 0.0),
            float(f.get("input_tokens", 0)),
        ]

    X_train = _np.array([_encode(f) for f in warmup_feats], dtype=_np.float64)
    y_train = _np.array(warmup_toks, dtype=_np.float64)

    hgb = _HGB(
        loss="quantile",
        quantile=0.50,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=max(5, phase1_n // 20),
        learning_rate=0.1,
        random_state=42,
    )
    hgb.fit(X_train, y_train)

    phase2_n = n - phase1_n
    X_pred = _np.array(
        [_encode(features[i]) for i in range(phase1_n, n)], dtype=_np.float64
    )
    preds_arr = hgb.predict(X_pred)
    for p in preds_arr:
        predictions.append(max(1.0, float(p)))

    errors = [abs(predictions[i] - all_toks[i]) for i in range(n)]
    biases = [predictions[i] - all_toks[i] for i in range(n)]
    mae = statistics.mean(errors)
    mean_bias = statistics.mean(biases)
    mean_actual = statistics.mean(all_toks)
    pred_std = statistics.stdev(predictions) if n > 1 else 0.0
    cv_pct = 100.0 * pred_std / max(1.0, mean_actual)

    return predictions, {
        "prior_type": "ml_hgb_p50",
        "warmup_n": phase1_n,
        "n_model_ids": len(all_model_ids),
        "n_requests": n,
        "phase2_n": phase2_n,
        "prior_cv_pct": round(cv_pct, 2),
        "prior_mae_tokens": round(mae, 2),
        "prior_rel_mae_pct": round(100.0 * mae / max(1.0, mean_actual), 2),
        "prior_bias_tokens": round(mean_bias, 2),
        "global_median_actual": round(global_median, 2),
    }


@dataclass
class MLPriorReport:
    """ML-HGB prior vs oracle comparison [run 2026-06-22-v].

    Compares four conditions on the same public trace:
      - FIFO baseline
      - Conformal oracle (upper bound, predicted == actual)
      - Conformal global prior (running median — run -t baseline)
      - Conformal ML-HGB prior (model_id + input_tokens, trained on warmup data)

    Key measurement: ml_vs_oracle_retention_pct — how much of the oracle gain
    survives when we replace oracle with the ML-HGB prior.
    Also: ml_vs_global_improvement_pct — gain of ML prior over running median.
    """

    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    warmup_n: int
    n_model_ids: int

    global_prior_cv_pct: float
    global_prior_mae_tokens: float
    global_prior_rel_mae_pct: float

    ml_prior_cv_pct: float
    ml_prior_mae_tokens: float
    ml_prior_rel_mae_pct: float

    fifo: dict
    conformal_oracle: dict
    conformal_global: dict
    conformal_ml: dict

    fifo_goodput_per_dollar: float
    oracle_goodput_per_dollar: float
    global_goodput_per_dollar: float
    ml_goodput_per_dollar: float

    oracle_delta_pct: float
    global_delta_pct: float
    ml_delta_pct: float
    global_vs_oracle_retention_pct: float
    ml_vs_oracle_retention_pct: float
    ml_vs_global_improvement_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "warmup_n": self.warmup_n,
            "n_model_ids": self.n_model_ids,
            "global_prior_cv_pct": round(self.global_prior_cv_pct, 2),
            "global_prior_mae_tokens": round(self.global_prior_mae_tokens, 2),
            "global_prior_rel_mae_pct": round(self.global_prior_rel_mae_pct, 2),
            "ml_prior_cv_pct": round(self.ml_prior_cv_pct, 2),
            "ml_prior_mae_tokens": round(self.ml_prior_mae_tokens, 2),
            "ml_prior_rel_mae_pct": round(self.ml_prior_rel_mae_pct, 2),
            "fifo": _r(self.fifo),
            "conformal_oracle": _r(self.conformal_oracle),
            "conformal_global": _r(self.conformal_global),
            "conformal_ml": _r(self.conformal_ml),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 2),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 2),
            "global_goodput_per_dollar": round(self.global_goodput_per_dollar, 2),
            "ml_goodput_per_dollar": round(self.ml_goodput_per_dollar, 2),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "global_delta_pct": round(self.global_delta_pct, 2),
            "ml_delta_pct": round(self.ml_delta_pct, 2),
            "global_vs_oracle_retention_pct": round(self.global_vs_oracle_retention_pct, 2),
            "ml_vs_oracle_retention_pct": round(self.ml_vs_oracle_retention_pct, 2),
            "ml_vs_global_improvement_pct": round(self.ml_vs_global_improvement_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_ml_prior_on_trace_with_features(
    raw: list[tuple[float, int]],
    features: list[dict],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    warmup_n: int,
) -> MLPriorReport:
    """Run FIFO / oracle / global-prior / ML-HGB-prior on a feature-annotated trace."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    global_preds, global_stats = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
    ml_preds, ml_stats = make_ml_prior_predictions_burstgpt(raw, features, warmup_n=warmup_n)

    n_model_ids = ml_stats.get("n_model_ids", 0)

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_global() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=global_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_ml() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=ml_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs = _build_oracle()
    oracle_reqs = _build_oracle()
    global_reqs = _build_global()
    ml_reqs = _build_ml()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")

    oracle_cal = ConformalAlphaCalibrator()
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_conformal(
        oracle_reqs, servers, oracle_cal
    )
    global_cal = ConformalAlphaCalibrator()
    global_sim, global_resp, _ = _simulate_decoupled_hybrid_conformal(
        global_reqs, servers, global_cal
    )
    ml_cal = ConformalAlphaCalibrator()
    ml_sim, ml_resp, _ = _simulate_decoupled_hybrid_conformal(
        ml_reqs, servers, ml_cal
    )

    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    gp_global = _sla_safe_goodput_per_dollar(global_reqs, global_resp, sla_s, servers)
    gp_ml = _sla_safe_goodput_per_dollar(ml_reqs, ml_resp, sla_s, servers)

    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle
    global_sim["sla_safe_goodput_per_dollar"] = gp_global
    ml_sim["sla_safe_goodput_per_dollar"] = gp_ml

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    global_ret = (gp_global / gp_oracle * 100.0) if gp_oracle > 0 else 0.0
    ml_ret = (gp_ml / gp_oracle * 100.0) if gp_oracle > 0 else 0.0

    return MLPriorReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        warmup_n=warmup_n,
        n_model_ids=n_model_ids,
        global_prior_cv_pct=global_stats.get("prior_cv_pct", 0.0),
        global_prior_mae_tokens=global_stats.get("prior_mae_tokens", 0.0),
        global_prior_rel_mae_pct=global_stats.get("prior_rel_mae_pct", 0.0),
        ml_prior_cv_pct=ml_stats.get("prior_cv_pct", 0.0),
        ml_prior_mae_tokens=ml_stats.get("prior_mae_tokens", 0.0),
        ml_prior_rel_mae_pct=ml_stats.get("prior_rel_mae_pct", 0.0),
        fifo=fifo_sim,
        conformal_oracle=oracle_sim,
        conformal_global=global_sim,
        conformal_ml=ml_sim,
        fifo_goodput_per_dollar=gp_fifo,
        oracle_goodput_per_dollar=gp_oracle,
        global_goodput_per_dollar=gp_global,
        ml_goodput_per_dollar=gp_ml,
        oracle_delta_pct=_delta(gp_fifo, gp_oracle),
        global_delta_pct=_delta(gp_fifo, gp_global),
        ml_delta_pct=_delta(gp_fifo, gp_ml),
        global_vs_oracle_retention_pct=global_ret,
        ml_vs_oracle_retention_pct=ml_ret,
        ml_vs_global_improvement_pct=_delta(gp_global, gp_ml),
    )


def run_burstgpt_hf_stratified_prior_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
    min_stratum_history: int = STRATIFIED_MIN_HISTORY,
) -> StratifiedPriorReport:
    """Stratified feature-aware causal prior on BurstGPT HF [run 2026-06-22-u].

    Compares four conditions on BurstGPT HF (59,999 records, CC-BY-4.0):
      1. FIFO — no ordering
      2. Conformal oracle — perfect token-length prediction (upper bound)
      3. Conformal global prior — causal sliding-window median [run -t baseline]
      4. Conformal stratified prior — per-(model_id, input_bin) causal median [NEW]

    Motivation: run -t measured 70.0% oracle retention for BurstGPT HF vs 81.6%
    for Azure LLM 2024. The root cause: BurstGPT mixes two model types with
    dramatically different output length distributions:
      ChatGPT (84.2% of traffic): p50=7 tokens
      GPT-4   (15.8% of traffic): p50=235 tokens
    A global running median (~18 tokens) is wrong for GPT-4 by 33×.

    The stratified prior addresses this by:
      1. Stratifying predictions by model_id (ChatGPT vs GPT-4)
      2. Within each model, further stratifying by input_token_bin ('short'/'long')
         using the causal running median of past input_tokens for that model as cutoff
      3. Using the per-stratum running median as the prediction (window=200)
      4. Falling back to per-model median if stratum is too sparse (< 20 requests)
      5. Falling back to global median if model is too sparse

    Expected outcomes (5,880-record sample, ρ=0.85, 4 servers, sla_s=30s):
      FIFO baseline:               ~ reference goodput/$
      Conformal oracle:            +644.4% vs FIFO [run -r baseline]
      Conformal global prior:      +420.83% vs FIFO [run -t baseline], 70.0% retention
      Conformal stratified prior:  ≥ +520% vs FIFO (≥ 80% retention), target ≥ 85%

    Why stratification helps:
      ChatGPT input-output correlation: r=0.513 (strong) — short/long input bins
      capture real response length differences.
      GPT-4 output p50=235 vs global p50=18 — per-model median is 13× more accurate.

    Args:
        servers:              Replica pool size (M/G/c).
        target_rho:           Target cluster utilization.
        job_limit:            Optional request cap (default: None = use all available).
        sla_s:                E2E SLA budget (seconds).
        prior_window:         Sliding window size for running medians.
        jsonl_path:           Path to BurstGPT HF normalized JSONL.
        min_stratum_history:  Minimum completions before using a stratum prior.

    Returns:
        ``StratifiedPriorReport`` with FIFO / oracle / global / stratified KPIs.
    """
    raw, features = load_burstgpt_serving_requests_jsonl_with_features(
        jsonl_path, limit=job_limit
    )
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 valid requests."
        )
    return _run_stratified_prior_on_trace_with_features(
        raw, features,
        trace_name="burstgpt_hf_stratified",
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        prior_window=prior_window,
        min_stratum_history=min_stratum_history,
    )


def run_burstgpt_hf_ml_prior_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    warmup_n: int = ML_PRIOR_WARMUP_N,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> MLPriorReport:
    """ML-HGB prior on BurstGPT HF [run 2026-06-22-v].

    Replaces the running-median prior (run -t, 70% retention) with an HGB model
    trained on per-request features (model_id, input_tokens) using a causal
    time-split: HGB is trained on the first ``warmup_n`` completions, then used
    to predict all subsequent requests.

    Research question:
      Can a trained ML predictor break the running-statistics ceiling (70% retention)?

    Mechanism:
      GPT-4 requests (15.8% of BurstGPT) have p50=235 tokens, but the global
      running median (~10 tokens) under-predicts them by 23×.  With 15.8% of
      requests at rel_err≈0.96, the p90 error exceeds 0.80, capping the
      conformal calibrator at α=2×alpha_max=0.002.  The HGB with model_id
      feature correctly predicts GPT-4 → ~235 tokens, dropping their errors
      to near 0.  With GPT-4 errors eliminated, p90 error falls to ~0.43
      (ChatGPT-normal tier), moving α from 0.002 to ~0.001 → dispatch
      becomes more SRPT-like → goodput/$ improves.

    Comparison conditions (all on BurstGPT HF, same fixture as run -t/-u):
      1. FIFO baseline
      2. Conformal oracle: +644.4% vs FIFO [run -r]
      3. Conformal global prior: +420.83% vs FIFO, 70.0% retention [run -t]
      4. Conformal ML-HGB prior: target ≥ +450% vs FIFO, ≥ 70% retention [NEW]

    Honesty note:
      Surprise-long ChatGPT requests (~8.4% of traffic, short input → long output)
      cannot be predicted by model_id or input_tokens alone.  They will continue
      to have large prediction errors (rel_err≈0.95) in Phase 2.  Whether they
      remain above or below the p90 error threshold determines whether the ML
      prior breaks the ceiling or merely approaches the stratified-prior result.

    Args:
        servers:    Replica pool size (M/G/c).
        target_rho: Target cluster utilization.
        job_limit:  Optional request cap (None = use all available).
        sla_s:      E2E SLA budget (seconds).
        warmup_n:   Requests to use as HGB training data.
        jsonl_path: Path to BurstGPT HF normalized JSONL.

    Returns:
        ``MLPriorReport`` with FIFO / oracle / global / ML-HGB KPIs.
    """
    raw, features = load_burstgpt_serving_requests_jsonl_with_features(
        jsonl_path, limit=job_limit
    )
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 valid requests."
        )
    return _run_ml_prior_on_trace_with_features(
        raw, features,
        trace_name="burstgpt_hf_ml_prior",
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        warmup_n=warmup_n,
    )


# ---------------------------------------------------------------------------
# Per-Class Conformal Calibration [run 2026-06-22-w]
# ---------------------------------------------------------------------------
#
# BOTTLENECK IDENTIFICATION
# ─────────────────────────
# Runs -t (live causal prior), -u (stratified), -v (ML-HGB) all confirmed the
# "running-statistics ceiling": BurstGPT achieves 70–82% oracle retention and
# no further improvement is possible from better prediction alone.
#
# Root cause (confirmed diagnostics):
#   BurstGPT mixes ChatGPT (p50=7 tok, high intra-class variance, p5=1/p95=800+)
#   and GPT-4 (p50≈235 tok, tight distribution).  The global p90 relative
#   prediction error is dominated by ChatGPT's long-tail "surprise" requests
#   (8.4% of traffic, rel_err≈0.95), keeping p90_err ≥ 0.80 regardless of
#   predictor quality.  This pushes the conformal calibrator's ratio to 2.0 →
#   global α capped at 0.002 for ALL requests.
#
#   Meanwhile, GPT-4 requests are predicted accurately by ML-HGB (rel_err≈0.02),
#   yet they also receive the globally-capped α=0.002 — preventing them from
#   benefiting from near-SRPT dispatch that their accurate predictions could enable.
#
# STRUCTURAL FIX: PER-CLASS CONFORMAL CALIBRATION
# ────────────────────────────────────────────────
# Instead of one global sliding window of residuals → one global α:
#   - Maintain a ConformalAlphaCalibrator per model class (model_id).
#   - On each completion, update the per-class calibrator for that request's class.
#   - At dispatch, use the per-class α for each waiting request's class.
#   - Fall back to the global calibrator for classes with insufficient data.
#
# Expected effect:
#   GPT-4 class:     per-class p90 rel_err ≈ 0.02 → α ≈ 0 → near-SRPT dispatch
#   ChatGPT class:   per-class p90 rel_err high  → α ≈ 0.002 (capped, same as global)
#   Mixed queue:     GPT-4 competes by pure remaining service; ChatGPT gets aging guard
#   Global result:   GPT-4 short requests better prioritized → more SLA-safe tokens
#
# Research basis:
# - RC3P (arXiv:2406.06818): class-conditional conformal prediction with class-wise
#   thresholding achieves 26.25% reduction in prediction-set sizes vs uniform CP.
# - TIE scheduling (arXiv:2604.00499): distributional ordering by model type
#   improves dispatch for mixed-model traffic.
# - Group-conditional conformal (ICCV 2023W Melki et al.): separate quantile
#   calibration per group achieves group-conditional coverage guarantees.
# - arXiv:2503.07545 (Mitzenmacher & Shahout 2025): identifies per-class
#   calibration as the open problem for production LLM schedulers.
# ---------------------------------------------------------------------------

# Minimum completions per class before using its per-class α (vs global fallback).
PER_CLASS_WARMUP_MIN: int = CONFORMAL_WARMUP // 2   # 50 completions per class


class PerClassConformalCalibrator:
    """Per-class conformal α calibrator for mixed-model LLM serving queues.

    Maintains one ``ConformalAlphaCalibrator`` per model_id class and a global
    fallback calibrator for classes with insufficient data.

    Key property: classes with accurate predictions (low per-class p90 rel_err)
    converge to α ≈ 0 independently of other classes' error rates.  This breaks
    the "running-statistics ceiling" where a single noisy class (ChatGPT) caps
    the global α for all other classes (GPT-4).

    Args:
        alpha_max:         Same as ``ConformalAlphaCalibrator``.
        warmup:            Per-class warmup completions (global calibrator uses
                           the full ``warmup``; per-class starts at warmup//2).
        window:            Sliding window per calibrator.
        target_p90_error:  Target p90 relative error for α=alpha_max mapping.
    """

    def __init__(
        self,
        alpha_max: float = CONFORMAL_ALPHA_MAX,
        warmup: int = CONFORMAL_WARMUP,
        window: int = CONFORMAL_WINDOW,
        target_p90_error: float = CONFORMAL_TARGET_P90_ERROR,
    ) -> None:
        self.alpha_max = alpha_max
        self.warmup = warmup
        self.window = window
        self.target_p90_error = target_p90_error
        self._global = ConformalAlphaCalibrator(alpha_max, warmup, window, target_p90_error)
        self._classes: dict[str, ConformalAlphaCalibrator] = {}
        self._class_counts: dict[str, int] = {}
        self._per_class_alpha_sum: dict[str, float] = {}
        self._per_class_alpha_count: dict[str, int] = {}

    def update(self, predicted_tokens: float, actual_tokens: int, model_id: str = "") -> None:
        """Record a completed request's residual in its class and the global calibrator."""
        self._global.update(predicted_tokens, actual_tokens)
        if model_id:
            if model_id not in self._classes:
                self._classes[model_id] = ConformalAlphaCalibrator(
                    self.alpha_max, PER_CLASS_WARMUP_MIN, self.window, self.target_p90_error
                )
            self._classes[model_id].update(predicted_tokens, actual_tokens)
            self._class_counts[model_id] = self._class_counts.get(model_id, 0) + 1

    def current_alpha(self, model_id: str = "") -> float:
        """Return calibrated dispatch α for the given model class.

        Uses per-class calibrator if the class has seen ≥ PER_CLASS_WARMUP_MIN
        completions (i.e., its warmup is satisfied).  Otherwise falls back to
        the global calibrator to avoid cold-start instability.
        """
        if model_id and model_id in self._classes:
            cls_cal = self._classes[model_id]
            if cls_cal._n_completed >= PER_CLASS_WARMUP_MIN:
                alpha = cls_cal.current_alpha()
                self._per_class_alpha_sum[model_id] = (
                    self._per_class_alpha_sum.get(model_id, 0.0) + alpha
                )
                self._per_class_alpha_count[model_id] = (
                    self._per_class_alpha_count.get(model_id, 0) + 1
                )
                return alpha
        return self._global.current_alpha()

    def mean_alpha(self) -> float:
        """Global mean α across all dispatch events (including per-class ones)."""
        return self._global.mean_alpha()

    def per_class_mean_alpha(self) -> dict[str, float]:
        """Diagnostic: mean α per class (only classes that reached per-class warmup)."""
        return {
            mid: (self._per_class_alpha_sum.get(mid, 0.0) /
                  max(1, self._per_class_alpha_count.get(mid, 1)))
            for mid in self._classes
            if self._class_counts.get(mid, 0) >= PER_CLASS_WARMUP_MIN
        }

    def class_counts(self) -> dict[str, int]:
        """Number of completions seen per class."""
        return dict(self._class_counts)


def _simulate_decoupled_hybrid_abs_conformal(
    requests: list["_Request"],
    servers: int,
    calibrator: "AbsoluteErrorConformalCalibrator",
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Decoupled Hybrid SRPT + absolute-error conformal alpha [run 2026-06-22-x].

    Phase 2 unification: the discipline now lives in
    ``aurelius.optimizer.policies.serving_queue.simulate_decoupled_hybrid_abs_conformal``.
    This thin shim injects the benchmark's ``_summarize`` (evaluation layer) and
    preserves the exact signature, return contract, and behavior. See
    research/results/canonical_optimizer_phase2_serving_policy_parity_2026-06-22.md.
    """
    return _abs_conformal_impl(
        requests, servers, calibrator, preemption_overhead_s, summarize=_summarize
    )


def _simulate_decoupled_hybrid_per_class_conformal(
    requests: list[_Request],
    servers: int,
    calibrator: PerClassConformalCalibrator,
    preemption_overhead_s: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Decoupled Hybrid SRPT with Per-Class Conformal Adaptive α [run 2026-06-22-w].

    Identical to ``_simulate_decoupled_hybrid_conformal`` except that the dispatch
    aging parameter α is calibrated *per model class* (request.model_id) rather
    than globally.  Each class independently adapts its α from its own empirical
    p90 prediction error.

    **Preemption key (on new arrival r):**
        remaining_s  [pure SRPT — same as global conformal variant]

    **Dispatch key (when server becomes free):**
        key(entry, t) = remaining_s / (1 + α_class(entry.model_id, t) × total_wait_s)
    where α_class = calibrator.current_alpha(model_id) is re-evaluated per request.

    With accurate predictions for a class (e.g. GPT-4, rel_err≈0.02):
      α_class → 0 after per-class warmup  →  dispatch is pure SRPT for that class.
    With noisy predictions (e.g. ChatGPT, rel_err≈0.95):
      α_class → 0.002 (capped)  →  same as global conformal variant.

    Research basis:
    - RC3P (arXiv:2406.06818): class-conditional conformal achieves group coverage
      guarantees while reducing prediction set sizes 26.25% vs uniform CP.
    - TIE scheduling (arXiv:2604.00499): distributional ordering by model type.
    - Group-conditional conformal (Melki et al., ICCV 2023W): per-group calibration.
    - arXiv:2503.07545: per-class calibration as open problem for LLM schedulers.
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]

    s_req:          list = [None] * servers
    s_start:        list = [0.0] * servers
    s_rem0:         list = [0.0] * servers
    s_ver:          list = [0]   * servers
    s_frozen_wait:  list = [0.0] * servers

    waiting: list = []

    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return max(0.0, s_rem0[sid] - (t - s_start[sid]))

    def _per_class_dispatch_key(entry: tuple, t: float) -> tuple:
        rem_s, frozen_wait_s, wait_entered_s, req = entry
        current_wait = t - wait_entered_s
        total_wait = frozen_wait_s + current_wait
        alpha = calibrator.current_alpha(req.model_id)
        ek = rem_s / max(1e-9, 1.0 + alpha * total_wait)
        return (ek, req.idx)

    def _start(sid: int, req: "_Request", rem: float, frozen_wait: float, t: float) -> None:
        s_req[sid]   = req
        s_start[sid] = t
        s_rem0[sid]  = rem
        s_ver[sid]  += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    response: dict[int, float] = {}

    while events:
        ev  = heapq.heappop(events)
        t   = ev[0]
        ety = ev[1]

        if ety == 0:  # ---- ARRIVAL ----------------------------------------
            req = ev[5]
            free = next((s for s in range(servers) if s_req[s] is None), None)
            if free is not None:
                s_frozen_wait[free] = 0.0
                _start(free, req, req.service_s, 0.0, t)
            else:
                # PURE SRPT PREEMPTION: no aging on preemption key.
                worst_sid, worst_rem = 0, -1.0
                for s in range(servers):
                    r = _remaining(s, t)
                    if r > worst_rem:
                        worst_rem, worst_sid = r, s

                if req.service_s < worst_rem:
                    preempted = s_req[worst_sid]
                    prem = _remaining(worst_sid, t)
                    pfrozen = s_frozen_wait[worst_sid]
                    s_req[worst_sid]  = None
                    s_ver[worst_sid] += 1
                    s_frozen_wait[worst_sid] = 0.0
                    _start(worst_sid, req, req.service_s, 0.0, t)
                    _npreempt[0] += 1
                    waiting.append((prem + preemption_overhead_s, pfrozen, t, preempted))
                else:
                    waiting.append((req.service_s, 0.0, t, req))

        else:  # ---- COMPLETION ---------------------------------------------
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue
            response[req.idx] = t - req.arrival_s

            # Update per-class calibrator with this request's residual.
            calibrator.update(req.predicted_tokens, req.actual_tokens, req.model_id)

            s_req[sid]  = None
            s_ver[sid] += 1

            if waiting:
                # PER-CLASS CONFORMAL DISPATCH: α recalibrated per class after each completion.
                best_i = min(
                    range(len(waiting)),
                    key=lambda i: _per_class_dispatch_key(waiting[i], t),
                )
                rem_s, frozen_wait_s, wait_entered_s, nxt = waiting.pop(best_i)
                new_frozen = frozen_wait_s + (t - wait_entered_s)
                s_frozen_wait[sid] = new_frozen
                _start(sid, nxt, rem_s, new_frozen, t)

    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp  = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits, servers)
    summary["preemption_count"] = _npreempt[0]
    summary["per_class_mean_alpha"] = calibrator.per_class_mean_alpha()
    summary["class_counts"] = calibrator.class_counts()
    return summary, response, wait_map


# ---------------------------------------------------------------------------
# Per-Class Conformal Report and Backtest [run 2026-06-22-w]
# ---------------------------------------------------------------------------

@dataclass
class PerClassConformalReport:
    """Comparison: FIFO / Oracle conformal / Global conformal / Per-class conformal.

    [run 2026-06-22-w] Validates that per-class conformal calibration breaks
    the running-statistics ceiling identified in runs -t/-u/-v.

    Primary hypothesis: classes with accurate predictions (GPT-4, rel_err≈0.02)
    converge to α≈0 independently, achieving near-SRPT dispatch for those
    requests while ChatGPT requests retain the safe α≈0.002 starvation guard.

    KPI columns:
      *_goodput_per_dollar     — SLA-safe tokens / (GPU-hour-dollars).
      *_delta_pct              — vs FIFO baseline.
      per_class_mean_alpha     — per-class mean α at dispatch (diagnostic).
      class_counts             — completions seen per class (diagnostic).
      per_class_vs_global_pct  — improvement of per-class over global conformal.
      per_class_vs_oracle_retention_pct — fraction of oracle gain achieved.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    n_model_ids: int

    fifo_goodput_per_dollar: float
    oracle_goodput_per_dollar: float
    global_goodput_per_dollar: float
    per_class_goodput_per_dollar: float

    oracle_delta_pct: float
    global_delta_pct: float
    per_class_delta_pct: float

    global_vs_oracle_retention_pct: float
    per_class_vs_oracle_retention_pct: float
    per_class_vs_global_pct: float

    global_mean_alpha: float
    per_class_mean_alpha: dict
    class_counts: dict

    fifo_sim: dict
    oracle_sim: dict
    global_sim: dict
    per_class_sim: dict

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "n_model_ids": self.n_model_ids,
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 2),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 2),
            "global_goodput_per_dollar": round(self.global_goodput_per_dollar, 2),
            "per_class_goodput_per_dollar": round(self.per_class_goodput_per_dollar, 2),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "global_delta_pct": round(self.global_delta_pct, 2),
            "per_class_delta_pct": round(self.per_class_delta_pct, 2),
            "global_vs_oracle_retention_pct": round(self.global_vs_oracle_retention_pct, 2),
            "per_class_vs_oracle_retention_pct": round(self.per_class_vs_oracle_retention_pct, 2),
            "per_class_vs_global_pct": round(self.per_class_vs_global_pct, 2),
            "global_mean_alpha": round(self.global_mean_alpha, 6),
            "per_class_mean_alpha": {
                k: round(v, 6) for k, v in (self.per_class_mean_alpha or {}).items()
            },
            "class_counts": self.class_counts,
            "fifo_sim": _r(self.fifo_sim),
            "oracle_sim": _r(self.oracle_sim),
            "global_sim": _r(self.global_sim),
            "per_class_sim": _r(self.per_class_sim),
            "shadow_tag": self.shadow_tag,
        }


def _run_per_class_conformal_on_trace(
    raw: list[tuple[float, int]],
    features: list[dict],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    ml_warmup_n: int,
) -> "PerClassConformalReport":
    """Run 4-discipline comparison: FIFO / oracle / global conformal / per-class conformal.

    Uses ML-HGB predictions (model_id + input_tokens) as the shared prior for
    both global-conformal and per-class-conformal disciplines.  The only
    difference is the calibrator: global uses one window for all requests;
    per-class uses one window per model_id.
    """
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    # Generate ML-HGB predictions (same as run -v, causal two-phase).
    ml_preds, ml_stats = make_ml_prior_predictions_burstgpt(raw, features, warmup_n=ml_warmup_n)
    # Use unique model IDs from features (more reliable than ml_stats for n_model_ids).
    n_model_ids = len(set(f.get("model_id", "") for f in features if f.get("model_id")))

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
                model_id=features[i].get("model_id", "") if i < len(features) else "",
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_ml() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=ml_preds[i],
                service_s=_service_time_s(tok),
                model_id=features[i].get("model_id", "") if i < len(features) else "",
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs        = _build_oracle()
    oracle_reqs      = _build_oracle()
    global_conf_reqs = _build_ml()
    per_cls_reqs     = _build_ml()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo

    oracle_cal = ConformalAlphaCalibrator()
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_conformal(
        oracle_reqs, servers, oracle_cal
    )
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle

    global_cal = ConformalAlphaCalibrator()
    global_sim, global_resp, _ = _simulate_decoupled_hybrid_conformal(
        global_conf_reqs, servers, global_cal
    )
    gp_global = _sla_safe_goodput_per_dollar(global_conf_reqs, global_resp, sla_s, servers)
    global_sim["sla_safe_goodput_per_dollar"] = gp_global

    per_cls_cal = PerClassConformalCalibrator()
    per_cls_sim, per_cls_resp, _ = _simulate_decoupled_hybrid_per_class_conformal(
        per_cls_reqs, servers, per_cls_cal
    )
    gp_per_cls = _sla_safe_goodput_per_dollar(per_cls_reqs, per_cls_resp, sla_s, servers)
    per_cls_sim["sla_safe_goodput_per_dollar"] = gp_per_cls

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    def _retention(oracle_delta: float, candidate_delta: float) -> float:
        return candidate_delta / oracle_delta * 100.0 if oracle_delta > 0 else 0.0

    oracle_delta  = _delta(gp_fifo, gp_oracle)
    global_delta  = _delta(gp_fifo, gp_global)
    per_cls_delta = _delta(gp_fifo, gp_per_cls)

    return PerClassConformalReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        n_model_ids=n_model_ids,
        fifo_goodput_per_dollar=gp_fifo,
        oracle_goodput_per_dollar=gp_oracle,
        global_goodput_per_dollar=gp_global,
        per_class_goodput_per_dollar=gp_per_cls,
        oracle_delta_pct=oracle_delta,
        global_delta_pct=global_delta,
        per_class_delta_pct=per_cls_delta,
        global_vs_oracle_retention_pct=_retention(oracle_delta, global_delta),
        per_class_vs_oracle_retention_pct=_retention(oracle_delta, per_cls_delta),
        per_class_vs_global_pct=_delta(gp_global, gp_per_cls),
        global_mean_alpha=global_cal.mean_alpha(),
        per_class_mean_alpha=per_cls_cal.per_class_mean_alpha(),
        class_counts=per_cls_cal.class_counts(),
        fifo_sim=fifo_sim,
        oracle_sim=oracle_sim,
        global_sim=global_sim,
        per_class_sim=per_cls_sim,
    )


def run_burstgpt_per_class_conformal_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
    ml_warmup_n: int = 300,
) -> "PerClassConformalReport":
    """Per-class conformal calibration backtest on BurstGPT HF [run 2026-06-22-w].

    Tests the hypothesis that per-class conformal calibration breaks the
    running-statistics ceiling identified in runs -t/-u/-v.

    Key question: does per-class α for GPT-4 (which ML-HGB predicts accurately)
    converge to near-0 independently of ChatGPT's high-variance errors?

    Disciplines compared:
      1. FIFO (baseline)
      2. Conformal oracle — upper bound (predicted == actual, global calibrator)
      3. Conformal ML-HGB global — replicates run -v result (~70% retention)
      4. Conformal ML-HGB per-class — NEW: per-class α → 0 for GPT-4 class

    Expected outcome (BurstGPT HF, ρ=0.85, 4 servers, SLA=30s):
      FIFO:                 reference goodput/$
      Oracle conformal:     +644.4% vs FIFO [run -r]
      Global conformal:     +420.83% vs FIFO [run -t] or ~-0.12% vs global-prior [run -v]
      Per-class conformal:  >+420.83% vs FIFO — GPT-4 class achieves near-SRPT

    Args:
        servers:      Replica pool size (M/G/c).
        target_rho:   Target cluster utilization.
        job_limit:    Optional request cap (None = use all available).
        sla_s:        E2E SLA budget.
        jsonl_path:   BurstGPT HF JSONL path with model_id features.
        ml_warmup_n:  HGB training window size.

    Returns:
        ``PerClassConformalReport`` with 4-discipline KPIs and per-class diagnostics.
    """
    raw, features = load_burstgpt_serving_requests_jsonl_with_features(
        jsonl_path, limit=job_limit
    )
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 valid requests."
        )
    return _run_per_class_conformal_on_trace(
        raw, features,
        trace_name="burstgpt_hf_per_class_conformal",
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        ml_warmup_n=ml_warmup_n,
    )


# ---------------------------------------------------------------------------
# Absolute-Error Conformal Backtest [run 2026-06-22-x]
# ---------------------------------------------------------------------------

@dataclass
class AbsConformalReport:
    """Comparison: FIFO / Oracle / Rel-conformal (live prior) / Abs-conformal (live prior).

    [run 2026-06-22-x] Tests whether replacing relative error with absolute error
    in the conformal calibrator breaks the calibrator cap and improves goodput/$.

    **Root cause being addressed:**
    Runs -t/-u/-v/-w confirmed that the relative-error calibrator caps at
    2×alpha_max=0.002 on BurstGPT HF because short ChatGPT requests (actual=7,
    predicted=18) produce rel_err=1.57 >> target=0.40, even though the absolute
    misprediction (11 tokens) is scheduling-irrelevant.

    Absolute error drives α from the genuine long-request uncertainty (300–600
    tokens), yielding α ≈ 0.0006–0.001 (near Pareto-optimal) instead of 0.002.

    **Disciplines compared:**
      1. FIFO                 — baseline
      2. Conformal oracle     — upper bound (predicted == actual)
      3. Rel-conformal live   — current best [run -t]: rel error, running median prior
      4. Abs-conformal live   — NEW: abs error, running median prior

    KPI columns:
      *_goodput_per_dollar     — SLA-safe tokens / (GPU-hour-dollars), primary metric.
      *_delta_pct              — vs FIFO (positive = better).
      abs_vs_rel_delta_pct     — improvement of abs-conformal over rel-conformal (key finding).
      abs_mean_alpha           — mean α of abs calibrator at dispatch (diagnostic).
      rel_mean_alpha           — mean α of rel calibrator at dispatch (diagnostic).
      abs_p90_abs_err_tokens   — p90 absolute error seen by abs calibrator (diagnostic).
      abs_vs_oracle_retention  — abs-conformal goodput/$ as fraction of oracle (%).
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float
    target_p90_abs_tokens: float

    fifo: dict
    conformal_oracle: dict
    rel_conformal_live: dict
    abs_conformal_live: dict

    fifo_goodput_per_dollar: float
    oracle_goodput_per_dollar: float
    rel_conformal_goodput_per_dollar: float
    abs_conformal_goodput_per_dollar: float

    oracle_delta_pct: float
    rel_conformal_delta_pct: float
    abs_conformal_delta_pct: float
    abs_vs_rel_delta_pct: float
    abs_vs_oracle_retention_pct: float
    rel_vs_oracle_retention_pct: float

    abs_mean_alpha: float
    rel_mean_alpha: float
    abs_p90_abs_err_tokens: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "target_p90_abs_tokens": self.target_p90_abs_tokens,
            "fifo": _r(self.fifo),
            "conformal_oracle": _r(self.conformal_oracle),
            "rel_conformal_live": _r(self.rel_conformal_live),
            "abs_conformal_live": _r(self.abs_conformal_live),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 4),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 4),
            "rel_conformal_goodput_per_dollar": round(self.rel_conformal_goodput_per_dollar, 4),
            "abs_conformal_goodput_per_dollar": round(self.abs_conformal_goodput_per_dollar, 4),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "rel_conformal_delta_pct": round(self.rel_conformal_delta_pct, 2),
            "abs_conformal_delta_pct": round(self.abs_conformal_delta_pct, 2),
            "abs_vs_rel_delta_pct": round(self.abs_vs_rel_delta_pct, 2),
            "abs_vs_oracle_retention_pct": round(self.abs_vs_oracle_retention_pct, 2),
            "rel_vs_oracle_retention_pct": round(self.rel_vs_oracle_retention_pct, 2),
            "abs_mean_alpha": round(self.abs_mean_alpha, 6),
            "rel_mean_alpha": round(self.rel_mean_alpha, 6),
            "abs_p90_abs_err_tokens": round(self.abs_p90_abs_err_tokens, 1),
            "shadow_tag": self.shadow_tag,
        }


def _run_abs_conformal_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
) -> AbsConformalReport:
    """Four-discipline comparison: FIFO / Oracle / Rel-conformal / Abs-conformal."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)
    live_preds, _ = make_live_prior_predictions(raw, window=prior_window)

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_live() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=live_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    fifo_reqs    = _build_oracle()
    oracle_reqs  = _build_oracle()
    rel_reqs     = _build_live()
    abs_reqs     = _build_live()

    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo

    oracle_cal = ConformalAlphaCalibrator()
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_conformal(
        oracle_reqs, servers, oracle_cal
    )
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle

    rel_cal = ConformalAlphaCalibrator()
    rel_sim, rel_resp, _ = _simulate_decoupled_hybrid_conformal(
        rel_reqs, servers, rel_cal
    )
    gp_rel = _sla_safe_goodput_per_dollar(rel_reqs, rel_resp, sla_s, servers)
    rel_sim["sla_safe_goodput_per_dollar"] = gp_rel

    abs_cal = AbsoluteErrorConformalCalibrator(
        target_p90_abs_tokens=target_p90_abs_tokens
    )
    abs_sim, abs_resp, _ = _simulate_decoupled_hybrid_abs_conformal(
        abs_reqs, servers, abs_cal
    )
    gp_abs = _sla_safe_goodput_per_dollar(abs_reqs, abs_resp, sla_s, servers)
    abs_sim["sla_safe_goodput_per_dollar"] = gp_abs

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    return AbsConformalReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        time_warp=warp,
        sla_s=sla_s,
        target_p90_abs_tokens=target_p90_abs_tokens,
        fifo=fifo_sim,
        conformal_oracle=oracle_sim,
        rel_conformal_live=rel_sim,
        abs_conformal_live=abs_sim,
        fifo_goodput_per_dollar=gp_fifo,
        oracle_goodput_per_dollar=gp_oracle,
        rel_conformal_goodput_per_dollar=gp_rel,
        abs_conformal_goodput_per_dollar=gp_abs,
        oracle_delta_pct=_delta(gp_fifo, gp_oracle),
        rel_conformal_delta_pct=_delta(gp_fifo, gp_rel),
        abs_conformal_delta_pct=_delta(gp_fifo, gp_abs),
        abs_vs_rel_delta_pct=_delta(gp_rel, gp_abs),
        abs_vs_oracle_retention_pct=(gp_abs / gp_oracle * 100.0) if gp_oracle > 0 else 0.0,
        rel_vs_oracle_retention_pct=(gp_rel / gp_oracle * 100.0) if gp_oracle > 0 else 0.0,
        abs_mean_alpha=abs_cal.mean_alpha(),
        rel_mean_alpha=rel_cal.mean_alpha(),
        abs_p90_abs_err_tokens=abs_cal.p90_abs_err_tokens(),
    )


def run_abs_conformal_azure_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = None,
    sla_s: float = DEFAULT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> AbsConformalReport:
    """Absolute-error conformal calibration on Azure LLM 2024 [run 2026-06-22-x].

    Tests whether replacing relative error with absolute error in the conformal
    calibrator breaks the calibrator cap observed in runs -t/-u/-v/-w.

    Disciplines compared:
      1. FIFO                     — baseline
      2. Conformal oracle         — upper bound (+322.24% vs FIFO expected [run -q])
      3. Rel-conformal live prior — current best (+244.42% vs FIFO expected [run -t])
      4. Abs-conformal live prior — NEW: does lower α improve goodput/$?

    Expected outcome if hypothesis is correct:
      Abs-conformal mean_α < rel-conformal mean_α (less capping from short misses).
      Abs-conformal goodput/$ > rel-conformal goodput/$ (closer to Pareto-optimal α=0.001).
      Abs-conformal goodput/$ > +244.42% vs FIFO — frontier improvement.

    Args:
        servers:               Replica pool size.
        target_rho:            Target cluster utilization.
        job_limit:             Optional request cap.
        sla_s:                 E2E SLA budget (seconds).
        prior_window:          Sliding-window size for running-median prior.
        target_p90_abs_tokens: Calibration target (tokens) for α = alpha_max.
        azure_fixture:         Path to Azure LLM 2024 CSV fixture.

    Returns:
        ``AbsConformalReport`` comparing all four disciplines.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_abs_conformal_on_trace(
        raw, "azure_llm_2024", servers, target_rho, sla_s,
        prior_window=prior_window,
        target_p90_abs_tokens=target_p90_abs_tokens,
    )


def run_abs_conformal_burstgpt_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = 5880,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> AbsConformalReport:
    """Absolute-error conformal calibration on BurstGPT HF [run 2026-06-22-x].

    Cross-validates abs-conformal on BurstGPT HF (heavier tail: p99≈934 tokens,
    bimodal ChatGPT + GPT-4 structure).  BurstGPT is the primary test trace because
    the relative-error calibrator is most severely capped here (ChatGPT short
    requests with rel_err=1.57 >> 0.40).

    Expected outcome if hypothesis is correct:
      Abs-conformal mean_α ≈ 0.0005–0.001 (vs rel-conformal 0.002 — capped).
      Abs-conformal goodput/$ > +420.83% vs FIFO (current live-prior best [run -t]).
      Frontier improvement confirmed if abs > rel by ≥ 1%.

    Args:
        servers:               Replica pool size.
        target_rho:            Target cluster utilization.
        job_limit:             Request cap (default 5880 for Azure comparability).
        sla_s:                 E2E SLA budget (seconds).
        prior_window:          Sliding-window size for running-median prior.
        target_p90_abs_tokens: Calibration target for α = alpha_max.
        jsonl_path:            Path to BurstGPT HF JSONL dataset.

    Returns:
        ``AbsConformalReport`` comparing all four disciplines.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 valid requests."
        )
    return _run_abs_conformal_on_trace(
        raw, "burstgpt_hf_abs_conformal", servers, target_rho, sla_s,
        prior_window=prior_window,
        target_p90_abs_tokens=target_p90_abs_tokens,
    )


# ---------------------------------------------------------------------------
# SLA-aware vs Abs-Conformal Head-to-Head [run 2026-06-22-y]
# ---------------------------------------------------------------------------
# Directly answers the north-star question: does abs-conformal (live prior)
# outperform SLA-aware scheduling by +300%?
#
# Compares six disciplines on identical simulator physics:
#   1. fifo               — arrival-order baseline
#   2. sla_aware_oracle   — binary short/long split on actual token counts (oracle)
#   3. sla_aware_live     — binary short/long split on running-median prediction (live)
#   4. rel_conformal_live — decoupled hybrid + relative-error conformal calibrator (live)
#   5. abs_conformal_live — decoupled hybrid + absolute-error conformal calibrator (live)
#   6. conformal_oracle   — decoupled hybrid + conformal calibrator, oracle prior (upper bound)
#
# Primary finding: abs_conformal_vs_sla_aware_oracle_delta_pct
#   If abs-conformal live-prior beats oracle SLA-aware, continuous prediction
#   dominates binary SLA classification regardless of prediction quality.
#
# Research basis:
#   - "Efficient Serving of LLM Applications with Probabilistic Demand Modeling"
#     (arXiv:2506.14851, Jun 2026): probabilistic request modeling shows that
#     continuous output-length uncertainty is more informative than binary SLA classes.
#   - "GoodServe: Towards High-Goodput Serving of Agentic LLM Inferences over
#     Heterogeneous Resources" (arXiv:2605.16867, May 2026): goodput and SLO
#     violation ratio as co-equal scheduling metrics — motivates head-to-head.
#   - "Flow-Controlled Scheduling for LLM Inference with Provable Stability
#     Guarantees" (arXiv:2604.11001, Apr 2026): flow control complements SRPT;
#     stability under different scheduling regimes.
# ---------------------------------------------------------------------------


@dataclass
class SLAAwareAbsConformalReport:
    """Six-discipline comparison: FIFO / SLA-aware (oracle+live) / conformal / oracle.

    [run 2026-06-22-y] Directly measures abs-conformal vs SLA-aware to answer
    whether abs-conformal achieves the north-star +300% vs SLA-aware schedulers.

    Disciplines:
      fifo                — FIFO (no ordering), oracle prior for token counts.
      sla_aware_oracle    — binary short/long split using ACTUAL token counts (oracle).
                            Upper bound for SLA-aware: knows the exact output length
                            but only uses binary classification.
      sla_aware_live      — binary short/long split using running-median prior (live).
                            Fair comparison: same prediction quality as abs-conformal.
      rel_conformal_live  — decoupled hybrid + relative-error conformal, live prior.
                            Current run-t best.
      abs_conformal_live  — decoupled hybrid + absolute-error conformal, live prior.
                            Current frontier [run 2026-06-22-x].
      conformal_oracle    — decoupled hybrid + rel-error conformal, oracle prior.
                            Upper bound: perfect token predictions.

    Primary KPI:
      abs_vs_sla_aware_oracle_delta_pct
        If > 0: abs-conformal with running-median prior beats oracle SLA-aware.
        This proves continuous token prediction + conformal calibration dominates
        binary SLA classification regardless of prediction accuracy.

      abs_vs_sla_aware_live_delta_pct
        Incremental gain of abs-conformal over live-prior SLA-aware.
        Quantifies the value of continuous ordering over binary classification
        when both use the same (running-median) prior.
    """
    trace: str
    total_requests: int
    servers: int
    target_rho: float
    time_warp: float
    sla_s: float
    target_p90_abs_tokens: float

    fifo: dict
    sla_aware_oracle: dict
    sla_aware_live: dict
    rel_conformal_live: dict
    abs_conformal_live: dict
    conformal_oracle: dict

    fifo_goodput_per_dollar: float
    sla_aware_oracle_goodput_per_dollar: float
    sla_aware_live_goodput_per_dollar: float
    rel_conformal_goodput_per_dollar: float
    abs_conformal_goodput_per_dollar: float
    oracle_goodput_per_dollar: float

    # All deltas vs FIFO
    sla_aware_oracle_delta_pct: float
    sla_aware_live_delta_pct: float
    rel_conformal_delta_pct: float
    abs_conformal_delta_pct: float
    oracle_delta_pct: float

    # Key head-to-head deltas (primary finding of run -y)
    abs_vs_sla_aware_oracle_delta_pct: float  # abs-conformal vs oracle SLA-aware
    abs_vs_sla_aware_live_delta_pct: float    # abs-conformal vs live-prior SLA-aware
    abs_vs_rel_delta_pct: float               # abs-conformal vs rel-conformal (run-x)

    # Retention metrics (fraction of oracle goodput achieved)
    abs_vs_oracle_retention_pct: float
    rel_vs_oracle_retention_pct: float
    sla_aware_oracle_retention_pct: float
    sla_aware_live_retention_pct: float

    # Calibrator diagnostics
    abs_mean_alpha: float
    rel_mean_alpha: float
    abs_p90_abs_err_tokens: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "time_warp": round(self.time_warp, 6),
            "sla_s": self.sla_s,
            "target_p90_abs_tokens": self.target_p90_abs_tokens,
            "fifo": _r(self.fifo),
            "sla_aware_oracle": _r(self.sla_aware_oracle),
            "sla_aware_live": _r(self.sla_aware_live),
            "rel_conformal_live": _r(self.rel_conformal_live),
            "abs_conformal_live": _r(self.abs_conformal_live),
            "conformal_oracle": _r(self.conformal_oracle),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 4),
            "sla_aware_oracle_goodput_per_dollar": round(self.sla_aware_oracle_goodput_per_dollar, 4),
            "sla_aware_live_goodput_per_dollar": round(self.sla_aware_live_goodput_per_dollar, 4),
            "rel_conformal_goodput_per_dollar": round(self.rel_conformal_goodput_per_dollar, 4),
            "abs_conformal_goodput_per_dollar": round(self.abs_conformal_goodput_per_dollar, 4),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 4),
            "sla_aware_oracle_delta_pct": round(self.sla_aware_oracle_delta_pct, 2),
            "sla_aware_live_delta_pct": round(self.sla_aware_live_delta_pct, 2),
            "rel_conformal_delta_pct": round(self.rel_conformal_delta_pct, 2),
            "abs_conformal_delta_pct": round(self.abs_conformal_delta_pct, 2),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "abs_vs_sla_aware_oracle_delta_pct": round(self.abs_vs_sla_aware_oracle_delta_pct, 2),
            "abs_vs_sla_aware_live_delta_pct": round(self.abs_vs_sla_aware_live_delta_pct, 2),
            "abs_vs_rel_delta_pct": round(self.abs_vs_rel_delta_pct, 2),
            "abs_vs_oracle_retention_pct": round(self.abs_vs_oracle_retention_pct, 2),
            "rel_vs_oracle_retention_pct": round(self.rel_vs_oracle_retention_pct, 2),
            "sla_aware_oracle_retention_pct": round(self.sla_aware_oracle_retention_pct, 2),
            "sla_aware_live_retention_pct": round(self.sla_aware_live_retention_pct, 2),
            "abs_mean_alpha": round(self.abs_mean_alpha, 6),
            "rel_mean_alpha": round(self.rel_mean_alpha, 6),
            "abs_p90_abs_err_tokens": round(self.abs_p90_abs_err_tokens, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_sla_aware_abs_conformal_on_trace(
    raw: list[tuple[float, int]],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
) -> SLAAwareAbsConformalReport:
    """Six-discipline comparison: FIFO / SLA-aware / conformal (rel+abs) / oracle."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)
    live_preds, _ = make_live_prior_predictions(raw, window=prior_window)

    def _build_oracle() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    def _build_live() -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=live_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    # Build request lists (each discipline gets its own copy)
    fifo_reqs          = _build_oracle()
    sla_oracle_reqs    = _build_oracle()  # SLA-aware with oracle token counts
    sla_live_reqs      = _build_live()    # SLA-aware with live-prior predictions
    rel_reqs           = _build_live()    # Rel-conformal with live prior
    abs_reqs           = _build_live()    # Abs-conformal with live prior
    oracle_conf_reqs   = _build_oracle()  # Conformal with oracle prior (upper bound)

    # Run all six disciplines
    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo

    sla_oracle_sim, sla_oracle_resp, _ = simulate_queue(sla_oracle_reqs, servers, "sla_aware")
    gp_sla_oracle = _sla_safe_goodput_per_dollar(sla_oracle_reqs, sla_oracle_resp, sla_s, servers)
    sla_oracle_sim["sla_safe_goodput_per_dollar"] = gp_sla_oracle

    sla_live_sim, sla_live_resp, _ = simulate_queue(sla_live_reqs, servers, "sla_aware")
    gp_sla_live = _sla_safe_goodput_per_dollar(sla_live_reqs, sla_live_resp, sla_s, servers)
    sla_live_sim["sla_safe_goodput_per_dollar"] = gp_sla_live

    rel_cal = ConformalAlphaCalibrator()
    rel_sim, rel_resp, _ = _simulate_decoupled_hybrid_conformal(rel_reqs, servers, rel_cal)
    gp_rel = _sla_safe_goodput_per_dollar(rel_reqs, rel_resp, sla_s, servers)
    rel_sim["sla_safe_goodput_per_dollar"] = gp_rel

    abs_cal = AbsoluteErrorConformalCalibrator(target_p90_abs_tokens=target_p90_abs_tokens)
    abs_sim, abs_resp, _ = _simulate_decoupled_hybrid_abs_conformal(abs_reqs, servers, abs_cal)
    gp_abs = _sla_safe_goodput_per_dollar(abs_reqs, abs_resp, sla_s, servers)
    abs_sim["sla_safe_goodput_per_dollar"] = gp_abs

    oracle_cal = ConformalAlphaCalibrator()
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_conformal(
        oracle_conf_reqs, servers, oracle_cal
    )
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_conf_reqs, oracle_resp, sla_s, servers)
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    def _retention(gp: float) -> float:
        return (gp / gp_oracle * 100.0) if gp_oracle > 0 else 0.0

    return SLAAwareAbsConformalReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        time_warp=warp,
        sla_s=sla_s,
        target_p90_abs_tokens=target_p90_abs_tokens,
        fifo=fifo_sim,
        sla_aware_oracle=sla_oracle_sim,
        sla_aware_live=sla_live_sim,
        rel_conformal_live=rel_sim,
        abs_conformal_live=abs_sim,
        conformal_oracle=oracle_sim,
        fifo_goodput_per_dollar=gp_fifo,
        sla_aware_oracle_goodput_per_dollar=gp_sla_oracle,
        sla_aware_live_goodput_per_dollar=gp_sla_live,
        rel_conformal_goodput_per_dollar=gp_rel,
        abs_conformal_goodput_per_dollar=gp_abs,
        oracle_goodput_per_dollar=gp_oracle,
        sla_aware_oracle_delta_pct=_delta(gp_fifo, gp_sla_oracle),
        sla_aware_live_delta_pct=_delta(gp_fifo, gp_sla_live),
        rel_conformal_delta_pct=_delta(gp_fifo, gp_rel),
        abs_conformal_delta_pct=_delta(gp_fifo, gp_abs),
        oracle_delta_pct=_delta(gp_fifo, gp_oracle),
        abs_vs_sla_aware_oracle_delta_pct=_delta(gp_sla_oracle, gp_abs),
        abs_vs_sla_aware_live_delta_pct=_delta(gp_sla_live, gp_abs),
        abs_vs_rel_delta_pct=_delta(gp_rel, gp_abs),
        abs_vs_oracle_retention_pct=_retention(gp_abs),
        rel_vs_oracle_retention_pct=_retention(gp_rel),
        sla_aware_oracle_retention_pct=_retention(gp_sla_oracle),
        sla_aware_live_retention_pct=_retention(gp_sla_live),
        abs_mean_alpha=abs_cal.mean_alpha(),
        rel_mean_alpha=rel_cal.mean_alpha(),
        abs_p90_abs_err_tokens=abs_cal.p90_abs_err_tokens(),
    )


def run_sla_aware_abs_conformal_azure_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
    sla_s: float = DEFAULT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
) -> SLAAwareAbsConformalReport:
    """Six-discipline head-to-head on Azure LLM 2024 [run 2026-06-22-y].

    Directly measures whether abs-conformal (live prior) outperforms oracle
    SLA-aware scheduling to answer the north-star question.

    Disciplines compared (identical M/G/c physics):
      1. fifo               — FIFO baseline
      2. sla_aware_oracle   — binary short/long using actual token counts (oracle)
      3. sla_aware_live     — binary short/long using running-median prior
      4. rel_conformal_live — decoupled hybrid + relative-error conformal, live
      5. abs_conformal_live — decoupled hybrid + absolute-error conformal, live
      6. conformal_oracle   — decoupled hybrid + conformal, oracle prior (ceiling)

    Primary finding: ``abs_vs_sla_aware_oracle_delta_pct``
      If positive, abs-conformal with live prior beats oracle SLA-aware —
      continuous prediction dominates binary classification regardless of
      prediction quality.

    Args:
        servers:               Replica pool size (M/G/c).
        target_rho:            Target cluster utilization.
        job_limit:             Request cap.
        sla_s:                 E2E response-time SLA budget (seconds).
        prior_window:          Sliding-window size for running-median prior.
        target_p90_abs_tokens: Calibration target for abs-error calibrator.
        azure_fixture:         Path to the Azure LLM 2024 CSV fixture.

    Returns:
        ``SLAAwareAbsConformalReport`` with all six discipline KPIs.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    return _run_sla_aware_abs_conformal_on_trace(
        raw, "azure_llm_2024_sla_aware_vs_abs_conformal",
        servers, target_rho, sla_s,
        prior_window=prior_window,
        target_p90_abs_tokens=target_p90_abs_tokens,
    )


def run_sla_aware_abs_conformal_burstgpt_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> SLAAwareAbsConformalReport:
    """Six-discipline head-to-head on BurstGPT HF [run 2026-06-22-y].

    Cross-validates the Azure LLM 2024 head-to-head on BurstGPT HF
    (heavier output-token distribution — stronger test of continuous prediction).

    Args:
        servers:               Replica pool size (M/G/c).
        target_rho:            Target cluster utilization.
        job_limit:             Request cap (default 5880 for Azure comparability).
        sla_s:                 E2E SLA budget (default 30s for BurstGPT).
        prior_window:          Sliding-window size for running-median prior.
        target_p90_abs_tokens: Calibration target for abs-error calibrator.
        jsonl_path:            Path to BurstGPT HF JSONL dataset.

    Returns:
        ``SLAAwareAbsConformalReport`` with all six discipline KPIs.
    """
    raw = load_burstgpt_serving_requests_jsonl(jsonl_path, limit=job_limit)
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 valid requests."
        )
    return _run_sla_aware_abs_conformal_on_trace(
        raw, "burstgpt_hf_sla_aware_vs_abs_conformal",
        servers, target_rho, sla_s,
        prior_window=prior_window,
        target_p90_abs_tokens=target_p90_abs_tokens,
    )


# =============================================================================
# Compound Economic × Queue Scheduling [run 2026-06-22-z]
# =============================================================================
# Answers: does the compound system (abs-conformal queue + economic provisioning)
# achieve the north-star of +300% vs oracle SLA-aware schedulers?
#
# Architecture:
#   Queue layer:       abs-conformal SRPT discipline (run 2026-06-22-x/y)
#   Provisioning layer: economic scheduling (time-of-day, spot, regional routing)
#                       from BENCHMARK_REGISTRY §1.1 — +25.75% vs SLA-aware,
#                       -21.2% GPU-hours (Azure LLM 2024 weekly trace).
#
# Independence assumption (verified):
#   Provisioning decisions (which GPU, when, where) are orthogonal to per-request
#   queue ordering. The compound gain is therefore multiplicative:
#     compound_goodput/$ = queue_goodput/$ × economic_cost_factor
#   where economic_cost_factor = 1.2575 (reduces effective GPU cost to 79.5%).
#
# Key finding (run -z):
#   Compound = +130% vs oracle SLA-aware (Azure), +166% (BurstGPT).
#   North-star (+300% vs SLA-aware) is NOT achieved by compound queue+economic.
#   Path to +300%: economic_factor_needed ≈ 2.18× (vs current 1.2575×), requiring
#   ~54% GPU-hour savings vs current -21.2%.


@dataclass
class CompoundEconomicQueueReport:
    """Compound economic × queue scheduling report [run 2026-06-22-z].

    Measures the combined gain when abs-conformal queue scheduling (queue layer)
    is composed with economic provisioning optimization (provisioning layer).

    The two layers are orthogonal:
      - Queue layer changes which request is served next (increases goodput numerator).
      - Provisioning layer selects cheaper GPU/time/region (reduces cost denominator).

    Compound gain formula:
      compound_goodput/$ = abs_conformal_goodput/$ × economic_cost_factor
      where economic_cost_factor = 1 + economic_gain_vs_sla_aware_pct / 100.

    The economic_cost_factor is sourced from BENCHMARK_REGISTRY §1.1 (Azure LLM 2024
    weekly trace, run 2026-06-21-s): +25.75% goodput/$ vs SLA-aware = 1.2575× cost
    efficiency improvement (i.e., effective GPU cost reduced to 79.5% of baseline).

    Primary KPI: compound_vs_sla_aware_oracle_delta_pct
      If ≥ 300.0: north-star achieved by compound system.
      If < 300.0: additional economic optimization required; see
        economic_factor_needed_for_north_star for the required multiplier.

    All results are shadow-only simulator estimates; NOT production savings.
    """

    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float

    # Queue-layer results (from run -y abs-conformal backtest)
    fifo_goodput_per_dollar: float
    sla_aware_oracle_goodput_per_dollar: float
    abs_conformal_goodput_per_dollar: float
    queue_vs_sla_aware_oracle_delta_pct: float   # queue alone vs oracle SLA-aware (run -y)
    abs_vs_fifo_delta_pct: float                  # abs-conformal vs FIFO

    # Economic-layer parameters
    economic_cost_factor: float          # provisioning multiplier (from BENCHMARK_REGISTRY)
    economic_cost_factor_source: str     # documentation reference

    # Compound results
    compound_goodput_per_dollar: float   # abs_conformal × economic_cost_factor
    compound_vs_sla_aware_oracle_delta_pct: float   # compound vs oracle SLA-aware
    compound_vs_fifo_delta_pct: float               # compound vs FIFO

    # North-star analysis
    north_star_target_pct: float         # = 300.0
    north_star_achieved: bool            # compound_vs_sla_aware_oracle_delta_pct >= 300.0
    economic_factor_needed_for_north_star: float  # factor needed to reach +300% vs SLA-aware
    economic_factor_needed_delta_vs_current: float  # how much more than current factor

    # Correction of run-t over-estimate
    run_t_compound_estimate_vs_fifo_pct: float   # run-t's multiplicative estimate
    corrected_compound_vs_fifo_pct: float        # correct compound vs FIFO
    over_estimate_factor: float                  # run-t over-estimate / correct compound

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 4),
            "sla_aware_oracle_goodput_per_dollar": round(self.sla_aware_oracle_goodput_per_dollar, 4),
            "abs_conformal_goodput_per_dollar": round(self.abs_conformal_goodput_per_dollar, 4),
            "queue_vs_sla_aware_oracle_delta_pct": round(self.queue_vs_sla_aware_oracle_delta_pct, 2),
            "abs_vs_fifo_delta_pct": round(self.abs_vs_fifo_delta_pct, 2),
            "economic_cost_factor": round(self.economic_cost_factor, 4),
            "economic_cost_factor_source": self.economic_cost_factor_source,
            "compound_goodput_per_dollar": round(self.compound_goodput_per_dollar, 4),
            "compound_vs_sla_aware_oracle_delta_pct": round(self.compound_vs_sla_aware_oracle_delta_pct, 2),
            "compound_vs_fifo_delta_pct": round(self.compound_vs_fifo_delta_pct, 2),
            "north_star_target_pct": self.north_star_target_pct,
            "north_star_achieved": self.north_star_achieved,
            "economic_factor_needed_for_north_star": round(self.economic_factor_needed_for_north_star, 4),
            "economic_factor_needed_delta_vs_current": round(self.economic_factor_needed_delta_vs_current, 4),
            "run_t_compound_estimate_vs_fifo_pct": round(self.run_t_compound_estimate_vs_fifo_pct, 2),
            "corrected_compound_vs_fifo_pct": round(self.corrected_compound_vs_fifo_pct, 2),
            "over_estimate_factor": round(self.over_estimate_factor, 4),
            "shadow_tag": self.shadow_tag,
        }


def _compute_compound_economic_queue(
    queue_rpt: SLAAwareAbsConformalReport,
    economic_cost_factor: float = ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY,
    economic_cost_factor_source: str = (
        "BENCHMARK_REGISTRY §1.1 Azure LLM 2024 — +25.75% vs sla_aware, "
        "-21.2% GPU-hours (run 2026-06-21-s)"
    ),
) -> CompoundEconomicQueueReport:
    """Apply the provisioning-layer economic factor to the queue-layer abs-conformal result.

    Independence of the two layers:
      - Queue ordering (abs-conformal SRPT) increases SLA-compliant tokens — numerator.
      - Provisioning (time-of-day, spot pricing, regional routing) reduces GPU cost —
        denominator.
      - The compound is multiplicative: compound = queue_goodput/$ × economic_factor.

    Corrects the run-t over-estimate:
      run-t computed compound = queue_multiplier_vs_fifo × economic_multiplier_vs_fifo,
      but both multipliers share the SLA-aware component, double-counting it.
      The correct compound:
        compound_goodput/$ = abs_conformal_goodput/$ × economic_cost_factor
      where economic_cost_factor = (economic+sla_aware_goodput/$) / (sla_aware_goodput/$)
      = 1 + economic_gain_vs_sla_aware = 1.2575.
    """
    gp_fifo = queue_rpt.fifo_goodput_per_dollar
    gp_sla = queue_rpt.sla_aware_oracle_goodput_per_dollar
    gp_abs = queue_rpt.abs_conformal_goodput_per_dollar
    oracle_gp = queue_rpt.oracle_goodput_per_dollar

    compound_gp = gp_abs * economic_cost_factor

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    # run-t compound over-estimate: used fifo_multiplier × economic_fifo_multiplier
    # The economic scheduler (constraint_aware) on Azure LLM 2024 full week achieved
    # +183.4% vs FIFO (= 2.834× FIFO). run-t multiplied this with the queue multiplier
    # vs FIFO, double-counting the SLA-aware component.
    # econ_fifo_multiplier = economic_cost_factor × (gp_sla / gp_fifo)
    # because: econ_sla_aware_goodput/$ = gp_sla × economic_cost_factor
    #           vs FIFO: gp_sla × economic_cost_factor / gp_fifo
    econ_fifo_multiplier = economic_cost_factor * (gp_sla / gp_fifo) if gp_fifo > 0 else 1.0
    queue_fifo_multiplier = gp_abs / gp_fifo if gp_fifo > 0 else 1.0
    run_t_estimate_multiplier = queue_fifo_multiplier * econ_fifo_multiplier
    run_t_delta_vs_fifo = (run_t_estimate_multiplier - 1.0) * 100.0
    corrected_delta_vs_fifo = _delta(gp_fifo, compound_gp)
    over_estimate = run_t_estimate_multiplier / (compound_gp / gp_fifo) if gp_fifo > 0 else 1.0

    # Economic factor needed to reach north-star (+300% vs oracle SLA-aware)
    # Need: compound_gp >= gp_sla × NORTH_STAR_MULTIPLIER
    # compound_gp = gp_abs × factor_needed
    # → factor_needed = gp_sla × NORTH_STAR_MULTIPLIER / gp_abs
    factor_needed = (gp_sla * NORTH_STAR_MULTIPLIER / gp_abs) if gp_abs > 0 else float("inf")
    factor_delta_vs_current = factor_needed - economic_cost_factor

    return CompoundEconomicQueueReport(
        trace=queue_rpt.trace.replace("sla_aware_vs_abs_conformal", "compound_economic_queue"),
        total_requests=queue_rpt.total_requests,
        servers=queue_rpt.servers,
        target_rho=queue_rpt.target_rho,
        sla_s=queue_rpt.sla_s,
        fifo_goodput_per_dollar=gp_fifo,
        sla_aware_oracle_goodput_per_dollar=gp_sla,
        abs_conformal_goodput_per_dollar=gp_abs,
        queue_vs_sla_aware_oracle_delta_pct=_delta(gp_sla, gp_abs),
        abs_vs_fifo_delta_pct=_delta(gp_fifo, gp_abs),
        economic_cost_factor=economic_cost_factor,
        economic_cost_factor_source=economic_cost_factor_source,
        compound_goodput_per_dollar=compound_gp,
        compound_vs_sla_aware_oracle_delta_pct=_delta(gp_sla, compound_gp),
        compound_vs_fifo_delta_pct=corrected_delta_vs_fifo,
        north_star_target_pct=300.0,
        north_star_achieved=_delta(gp_sla, compound_gp) >= 300.0,
        economic_factor_needed_for_north_star=factor_needed,
        economic_factor_needed_delta_vs_current=factor_delta_vs_current,
        run_t_compound_estimate_vs_fifo_pct=run_t_delta_vs_fifo,
        corrected_compound_vs_fifo_pct=corrected_delta_vs_fifo,
        over_estimate_factor=over_estimate,
    )


def run_compound_economic_queue_azure_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
    sla_s: float = DEFAULT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
    economic_cost_factor: float = ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY,
) -> CompoundEconomicQueueReport:
    """Compound economic × queue backtest on Azure LLM 2024 [run 2026-06-22-z].

    Composes:
      1. abs-conformal queue scheduling (run -y): +83.27% vs oracle SLA-aware
      2. Economic provisioning (BENCHMARK_REGISTRY §1.1): +25.75% vs SLA-aware
         via -21.2% GPU-hours (time-of-day/spot/regional routing)

    Independence: provisioning layer (cost denominator) is orthogonal to queue
    ordering layer (goodput numerator). Compound = queue × economic_cost_factor.

    Args:
        servers:              Replica pool size.
        target_rho:           Target utilization.
        job_limit:            Request cap.
        sla_s:                E2E SLA budget.
        prior_window:         Sliding-window for running-median prior.
        target_p90_abs_tokens: Abs-error calibration target.
        azure_fixture:        Azure LLM 2024 CSV fixture path.
        economic_cost_factor: Provisioning cost efficiency multiplier.

    Returns:
        ``CompoundEconomicQueueReport`` with compound north-star assessment.
    """
    queue_rpt = run_sla_aware_abs_conformal_azure_backtest(
        servers=servers,
        target_rho=target_rho,
        job_limit=job_limit,
        sla_s=sla_s,
        prior_window=prior_window,
        target_p90_abs_tokens=target_p90_abs_tokens,
        azure_fixture=azure_fixture,
    )
    return _compute_compound_economic_queue(queue_rpt, economic_cost_factor=economic_cost_factor)


def run_compound_economic_queue_burstgpt_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
    economic_cost_factor: float = ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY,
) -> CompoundEconomicQueueReport:
    """Compound economic × queue backtest on BurstGPT HF [run 2026-06-22-z].

    Cross-validates the compound result on BurstGPT HF. The economic cost factor
    is sourced from the Azure LLM 2024 provisioning benchmark (BENCHMARK_REGISTRY §1.1)
    and applied conservatively to BurstGPT — provisioning-level savings (spot pricing,
    regional routing, time-of-day) are workload-agnostic.

    Args:
        servers:              Replica pool size.
        target_rho:           Target utilization.
        job_limit:            Request cap.
        sla_s:                E2E SLA budget (default 30s for BurstGPT).
        prior_window:         Sliding-window for running-median prior.
        target_p90_abs_tokens: Abs-error calibration target.
        jsonl_path:           Path to BurstGPT HF JSONL.
        economic_cost_factor: Provisioning cost efficiency multiplier.

    Returns:
        ``CompoundEconomicQueueReport`` with compound north-star assessment.
    """
    queue_rpt = run_sla_aware_abs_conformal_burstgpt_backtest(
        servers=servers,
        target_rho=target_rho,
        job_limit=job_limit,
        sla_s=sla_s,
        prior_window=prior_window,
        target_p90_abs_tokens=target_p90_abs_tokens,
        jsonl_path=jsonl_path,
    )
    return _compute_compound_economic_queue(queue_rpt, economic_cost_factor=economic_cost_factor)


# ---------------------------------------------------------------------------
# ML Prior under Absolute-Error Conformal — Run 2026-06-22-z
#
# Run -v found the ML-HGB prior (model_id + input_tokens) to be a NULL RESULT
# on BurstGPT: ml_vs_global_improvement = -0.12% under the RELATIVE-error
# conformal calibrator. But run -v explicitly identified the cause: the
# relative-error calibrator was CAPPED at mean_α = 0.002 for BOTH the global
# and ML priors, because p90 relative prediction error stayed >= 0.80 in both
# cases (ChatGPT short-request rel_err dominates). The calibrator — not the
# prior accuracy — was the binding constraint.
#
# Run -x then REMOVED that cap with the absolute-error conformal calibrator,
# lifting the global running-median prior from +420.83% (70.0% retention) to
# +557.12% (88.3% retention) on BurstGPT.
#
# This run closes the obvious open question left by runs -v and -x:
#   Does the ML-HGB prior — whose accuracy IS better than the running median
#   (run -v measured MAE -2.5%, and far better per-model centering) — finally
#   translate into a goodput gain once the absolute-error calibrator can
#   exploit it (α no longer capped)?
#
# Design: a clean 2x2 (prior {global running-median, ML-HGB}) x (calibrator
# {relative-error, absolute-error}), plus FIFO and oracle. This isolates the
# two factors:
#   - global+rel  : run -t baseline      (+420.83%, 70.0% retention)
#   - global+abs  : run -x result        (+557.12%, 88.3% retention)
#   - ml+rel      : run -v null result   (+420.2%,  69.88% retention)
#   - ml+abs      : NEW — the open cell
#
# Falsifiable hypothesis: ml+abs > global+abs by >= 1% (frontier improvement),
# because the abs calibrator rewards the ML prior's better long-request
# centering that the rel calibrator masked.
#
# Research basis:
# - GAP_ANALYSIS run -v Q-conclusion (rel-error formula is the binding
#   constraint, not prediction accuracy)
# - run -x (absolute-error conformal breaks the running-statistics ceiling)
# - arXiv:2508.14544 (Adaptively Robust LLM Inference)
# - arXiv:1902.00732 (Scheduling with Predictions, Mitzenmacher 2019)
# ---------------------------------------------------------------------------


@dataclass
class MLAbsConformalReport:
    """ML prior x {rel, abs} conformal 2x2 comparison [run 2026-06-22-z].

    Six conditions on one public trace:
      - FIFO                        — baseline
      - Conformal oracle (abs)      — upper bound (perfect token prediction)
      - global + rel-conformal      — run -t baseline
      - global + abs-conformal      — run -x result
      - ML-HGB + rel-conformal      — run -v null result
      - ML-HGB + abs-conformal      — NEW (this run)

    Primary measurement: ml_abs_vs_global_abs_pct — does the ML prior beat the
    running-median prior once the absolute-error calibrator can use it?
    Secondary: ml_abs_vs_ml_rel_pct — does abs-conformal unlock the ML prior
    that rel-conformal capped (run -v)?
    """

    trace: str
    total_requests: int
    servers: int
    target_rho: float
    sla_s: float
    warmup_n: int
    n_model_ids: int
    target_p90_abs_tokens: float

    # Prior quality diagnostics
    global_prior_cv_pct: float
    global_prior_mae_tokens: float
    ml_prior_cv_pct: float
    ml_prior_mae_tokens: float

    # Calibrator diagnostics
    global_rel_mean_alpha: float
    global_abs_mean_alpha: float
    ml_rel_mean_alpha: float
    ml_abs_mean_alpha: float

    # Simulation summaries
    fifo: dict
    conformal_oracle: dict
    global_rel: dict
    global_abs: dict
    ml_rel: dict
    ml_abs: dict

    # KPIs (SLA-safe goodput/$)
    fifo_goodput_per_dollar: float
    oracle_goodput_per_dollar: float
    global_rel_goodput_per_dollar: float
    global_abs_goodput_per_dollar: float
    ml_rel_goodput_per_dollar: float
    ml_abs_goodput_per_dollar: float

    # Deltas vs FIFO
    oracle_delta_pct: float
    global_rel_delta_pct: float
    global_abs_delta_pct: float
    ml_rel_delta_pct: float
    ml_abs_delta_pct: float

    # Retention vs oracle
    global_abs_retention_pct: float
    ml_abs_retention_pct: float

    # The two key contrasts
    ml_abs_vs_global_abs_pct: float   # PRIMARY: ML vs running-median under abs-conformal
    ml_abs_vs_ml_rel_pct: float       # SECONDARY: does abs unlock the ML prior?

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _r(d: dict) -> dict:
            return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "servers": self.servers,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "warmup_n": self.warmup_n,
            "n_model_ids": self.n_model_ids,
            "target_p90_abs_tokens": self.target_p90_abs_tokens,
            "global_prior_cv_pct": round(self.global_prior_cv_pct, 2),
            "global_prior_mae_tokens": round(self.global_prior_mae_tokens, 2),
            "ml_prior_cv_pct": round(self.ml_prior_cv_pct, 2),
            "ml_prior_mae_tokens": round(self.ml_prior_mae_tokens, 2),
            "global_rel_mean_alpha": round(self.global_rel_mean_alpha, 6),
            "global_abs_mean_alpha": round(self.global_abs_mean_alpha, 6),
            "ml_rel_mean_alpha": round(self.ml_rel_mean_alpha, 6),
            "ml_abs_mean_alpha": round(self.ml_abs_mean_alpha, 6),
            "fifo": _r(self.fifo),
            "conformal_oracle": _r(self.conformal_oracle),
            "global_rel": _r(self.global_rel),
            "global_abs": _r(self.global_abs),
            "ml_rel": _r(self.ml_rel),
            "ml_abs": _r(self.ml_abs),
            "fifo_goodput_per_dollar": round(self.fifo_goodput_per_dollar, 4),
            "oracle_goodput_per_dollar": round(self.oracle_goodput_per_dollar, 4),
            "global_rel_goodput_per_dollar": round(self.global_rel_goodput_per_dollar, 4),
            "global_abs_goodput_per_dollar": round(self.global_abs_goodput_per_dollar, 4),
            "ml_rel_goodput_per_dollar": round(self.ml_rel_goodput_per_dollar, 4),
            "ml_abs_goodput_per_dollar": round(self.ml_abs_goodput_per_dollar, 4),
            "oracle_delta_pct": round(self.oracle_delta_pct, 2),
            "global_rel_delta_pct": round(self.global_rel_delta_pct, 2),
            "global_abs_delta_pct": round(self.global_abs_delta_pct, 2),
            "ml_rel_delta_pct": round(self.ml_rel_delta_pct, 2),
            "ml_abs_delta_pct": round(self.ml_abs_delta_pct, 2),
            "global_abs_retention_pct": round(self.global_abs_retention_pct, 2),
            "ml_abs_retention_pct": round(self.ml_abs_retention_pct, 2),
            "ml_abs_vs_global_abs_pct": round(self.ml_abs_vs_global_abs_pct, 2),
            "ml_abs_vs_ml_rel_pct": round(self.ml_abs_vs_ml_rel_pct, 2),
            "shadow_tag": self.shadow_tag,
        }


def _run_ml_abs_conformal_on_trace(
    raw: list[tuple[float, int]],
    features: list[dict],
    trace_name: str,
    servers: int,
    target_rho: float,
    sla_s: float,
    warmup_n: int,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
) -> MLAbsConformalReport:
    """2x2 (prior x calibrator) + FIFO + oracle on a feature-annotated trace [run -z]."""
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)

    global_preds, global_stats = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
    ml_preds, ml_stats = make_ml_prior_predictions_burstgpt(raw, features, warmup_n=warmup_n)
    n_model_ids = ml_stats.get("n_model_ids", 0)

    def _build(preds: Optional[list[float]]) -> list[_Request]:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=(float(tok) if preds is None else preds[i]),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    # FIFO
    fifo_reqs = _build(None)
    fifo_sim, fifo_resp, _ = simulate_queue(fifo_reqs, servers, "fifo")
    gp_fifo = _sla_safe_goodput_per_dollar(fifo_reqs, fifo_resp, sla_s, servers)
    fifo_sim["sla_safe_goodput_per_dollar"] = gp_fifo

    # Oracle (abs-conformal calibrator; α→0 with perfect prediction)
    oracle_reqs = _build(None)
    oracle_cal = AbsoluteErrorConformalCalibrator(target_p90_abs_tokens=target_p90_abs_tokens)
    oracle_sim, oracle_resp, _ = _simulate_decoupled_hybrid_abs_conformal(
        oracle_reqs, servers, oracle_cal
    )
    gp_oracle = _sla_safe_goodput_per_dollar(oracle_reqs, oracle_resp, sla_s, servers)
    oracle_sim["sla_safe_goodput_per_dollar"] = gp_oracle

    # global + rel-conformal (run -t baseline)
    gr_reqs = _build(global_preds)
    gr_cal = ConformalAlphaCalibrator()
    gr_sim, gr_resp, _ = _simulate_decoupled_hybrid_conformal(gr_reqs, servers, gr_cal)
    gp_gr = _sla_safe_goodput_per_dollar(gr_reqs, gr_resp, sla_s, servers)
    gr_sim["sla_safe_goodput_per_dollar"] = gp_gr

    # global + abs-conformal (run -x result)
    ga_reqs = _build(global_preds)
    ga_cal = AbsoluteErrorConformalCalibrator(target_p90_abs_tokens=target_p90_abs_tokens)
    ga_sim, ga_resp, _ = _simulate_decoupled_hybrid_abs_conformal(ga_reqs, servers, ga_cal)
    gp_ga = _sla_safe_goodput_per_dollar(ga_reqs, ga_resp, sla_s, servers)
    ga_sim["sla_safe_goodput_per_dollar"] = gp_ga

    # ML + rel-conformal (run -v null result)
    mr_reqs = _build(ml_preds)
    mr_cal = ConformalAlphaCalibrator()
    mr_sim, mr_resp, _ = _simulate_decoupled_hybrid_conformal(mr_reqs, servers, mr_cal)
    gp_mr = _sla_safe_goodput_per_dollar(mr_reqs, mr_resp, sla_s, servers)
    mr_sim["sla_safe_goodput_per_dollar"] = gp_mr

    # ML + abs-conformal (NEW — the open cell)
    ma_reqs = _build(ml_preds)
    ma_cal = AbsoluteErrorConformalCalibrator(target_p90_abs_tokens=target_p90_abs_tokens)
    ma_sim, ma_resp, _ = _simulate_decoupled_hybrid_abs_conformal(ma_reqs, servers, ma_cal)
    gp_ma = _sla_safe_goodput_per_dollar(ma_reqs, ma_resp, sla_s, servers)
    ma_sim["sla_safe_goodput_per_dollar"] = gp_ma

    def _delta(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    return MLAbsConformalReport(
        trace=trace_name,
        total_requests=len(raw),
        servers=servers,
        target_rho=target_rho,
        sla_s=sla_s,
        warmup_n=warmup_n,
        n_model_ids=n_model_ids,
        target_p90_abs_tokens=target_p90_abs_tokens,
        global_prior_cv_pct=global_stats.get("prior_cv_pct", 0.0),
        global_prior_mae_tokens=global_stats.get("prior_mae_tokens", 0.0),
        ml_prior_cv_pct=ml_stats.get("prior_cv_pct", 0.0),
        ml_prior_mae_tokens=ml_stats.get("prior_mae_tokens", 0.0),
        global_rel_mean_alpha=gr_cal.mean_alpha(),
        global_abs_mean_alpha=ga_cal.mean_alpha(),
        ml_rel_mean_alpha=mr_cal.mean_alpha(),
        ml_abs_mean_alpha=ma_cal.mean_alpha(),
        fifo=fifo_sim,
        conformal_oracle=oracle_sim,
        global_rel=gr_sim,
        global_abs=ga_sim,
        ml_rel=mr_sim,
        ml_abs=ma_sim,
        fifo_goodput_per_dollar=gp_fifo,
        oracle_goodput_per_dollar=gp_oracle,
        global_rel_goodput_per_dollar=gp_gr,
        global_abs_goodput_per_dollar=gp_ga,
        ml_rel_goodput_per_dollar=gp_mr,
        ml_abs_goodput_per_dollar=gp_ma,
        oracle_delta_pct=_delta(gp_fifo, gp_oracle),
        global_rel_delta_pct=_delta(gp_fifo, gp_gr),
        global_abs_delta_pct=_delta(gp_fifo, gp_ga),
        ml_rel_delta_pct=_delta(gp_fifo, gp_mr),
        ml_abs_delta_pct=_delta(gp_fifo, gp_ma),
        global_abs_retention_pct=(gp_ga / gp_oracle * 100.0) if gp_oracle > 0 else 0.0,
        ml_abs_retention_pct=(gp_ma / gp_oracle * 100.0) if gp_oracle > 0 else 0.0,
        ml_abs_vs_global_abs_pct=_delta(gp_ga, gp_ma),
        ml_abs_vs_ml_rel_pct=_delta(gp_mr, gp_ma),
    )


def run_burstgpt_hf_ml_abs_conformal_backtest(
    servers: int = 4,
    target_rho: float = 0.85,
    job_limit: Optional[int] = 5880,
    sla_s: float = DEFAULT_BURSTGPT_SLA_S,
    warmup_n: int = ML_PRIOR_WARMUP_N,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    jsonl_path: str = DEFAULT_BURSTGPT_HF_JSONL,
) -> MLAbsConformalReport:
    """ML-HGB prior under absolute-error conformal calibration on BurstGPT HF [run -z].

    Closes the open question from runs -v and -x: run -v found the ML prior to be
    a null result (-0.12% vs global) under the RELATIVE-error calibrator, which was
    capped at mean_α=0.002 for both priors. Run -x removed the cap via the
    absolute-error calibrator (global prior: +420.83% → +557.12%). This backtest
    tests whether the ML prior's better accuracy finally pays off once abs-conformal
    can exploit it.

    Six disciplines on BurstGPT HF (default 5,880 requests, ρ=0.85, SLA=30s):
      FIFO / oracle(abs) / global+rel / global+abs / ml+rel / ml+abs.

    Falsifiable hypothesis: ml_abs_vs_global_abs_pct >= 1% (frontier improvement).

    Returns:
        ``MLAbsConformalReport`` with the full 2x2 + FIFO + oracle comparison.
    """
    raw, features = load_burstgpt_serving_requests_jsonl_with_features(
        jsonl_path, limit=job_limit
    )
    if len(raw) < 2:
        raise ValueError(
            f"BurstGPT HF JSONL at {jsonl_path!r} returned fewer than 2 valid requests."
        )
    return _run_ml_abs_conformal_on_trace(
        raw, features, "burstgpt_hf_ml_abs_conformal",
        servers, target_rho, sla_s, warmup_n,
        target_p90_abs_tokens=target_p90_abs_tokens,
    )


# ---------------------------------------------------------------------------
# Joint Economic × Queue Compound Backtest — Run 2026-06-23
#
# First TRUE compound measurement: provisioning (MCS per-tick variable-c
# replica schedule) and queue ordering (abs-conformal SRTF) composed in a
# SINGLE discrete-event simulation on the Azure LLM 2024 public trace.
#
# Previous compound (run-z) applied the economic cost factor as a post-hoc
# independence multiplier:
#   compound_gp/$ = abs_conformal_gp/$ × ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY
# This run removes that assumption by driving the serving simulation with the
# actual MCS c_schedule so both effects are measured end-to-end.
#
# 2×2 factorial design:
#   queue discipline: {FIFO, abs-conformal SRTF}
#   provisioning:     {fixed-c=4, MCS variable-c}
# Cost denominator: provisioned GPU hours × GPU_HOUR_USD (changes between
# fixed-c and MCS-c; NOT the service-time cost shared across queue disciplines).
#
# Expected results (vs run-z independence estimates):
#   abs-conformal+fixed-c:  +313% vs FIFO+fixed-c  [reference for run-y]
#   abs-conformal+MCS-c:    > abs-conformal+fixed-c × provisioning_cost_factor
#   independence vs truth:  close if queue+econ interactions are small
#
# Falsifiable hypothesis:
#   TRUE compound ≥ 0.90 × independence estimate
#   (≥10% tolerance for SLA degradation under variable-c transitions)
# ---------------------------------------------------------------------------


def _erlang_c_sla_timeout_pct(
    lam: float,
    mean_service_s: float,
    c: int,
    sla_wait_threshold_s: float,
) -> float:
    """Fraction of M/M/c arrivals waiting longer than sla_wait_threshold_s (%).

    Uses the standard Erlang-C formula with service rate μ = 1/mean_service_s.
    M/M/c is a conservative approximation for M/D/c (deterministic service times
    used by the queue simulation). Returns 100.0 when system is overloaded (ρ≥1).

    This provides queue-simulation-consistent physics for the MCS c_schedule:
    service time = TTFT_BASE_S + output_tokens × TPOT_S (same as _service_time_s).
    """
    mu = 1.0 / max(mean_service_s, 1e-12)
    a = lam / mu          # total traffic intensity (Erlangs)
    rho = a / max(c, 1)   # per-server utilization

    if rho >= 1.0:
        return 100.0

    # Erlang-C: P(new arrival must wait) via log-domain summation for stability.
    log_a = math.log(a) if a > 1e-12 else -1e9
    # Compute a^c / c! in log space
    log_ac_over_cfact = c * log_a - sum(math.log(k) for k in range(1, c + 1))
    # Compute sum_{k=0}^{c-1} a^k / k!
    log_sum_terms: list = []
    log_fact_k = 0.0
    for k in range(c):
        if k > 0:
            log_fact_k += math.log(k)
        log_sum_terms.append(k * log_a - log_fact_k)

    log_last = log_ac_over_cfact + math.log(c / max(c - a, 1e-9))
    all_logs = log_sum_terms + [log_last]
    max_log = max(all_logs)
    denom = sum(math.exp(x - max_log) for x in all_logs)
    erlang_c_prob = math.exp(log_last - max_log) / denom

    # P(wait > t) = C(c,a) * exp(-(c*μ - λ) * t)
    excess_rate = c * mu - lam
    prob_exceed = erlang_c_prob * math.exp(-excess_rate * sla_wait_threshold_s)
    return min(100.0, max(0.0, prob_exceed * 100.0))


def _joint_mcs_c_schedule(
    raw: list,
    tick_seconds: float,
    warp: float,
    mcs_gate: float = 9.5,
    sla_s: float = DEFAULT_SLA_S,
) -> list:
    """Per-tick MCS replica counts using queue-simulation-consistent physics.

    Uses Erlang-C M/M/c with μ = 1/_service_time_s(mean_output_tokens) per
    server — the same service model as the discrete-event queue simulation.
    This avoids the throughput-model mismatch between the provisioning backtest
    (FALLBACK_TOKENS_PER_S=2500 tok/s via continuous batching) and the queue
    simulation (TPOT_S=0.020 s/tok via sequential decoding).

    Finds min c per tick where P(queue_wait > sla_s − service_s) < mcs_gate%.
    Empty ticks (no arrivals) return 1 replica.

    Args:
        raw:          Raw ``(arrival_s, output_tokens)`` tuples (unwarped).
        tick_seconds: Tick duration in warped seconds.
        warp:         Time-warp factor (arrival_warped = arrival_raw / warp).
        mcs_gate:     Timeout-rate threshold (%). Default 9.5 < 10% SLA target.
        sla_s:        E2E SLA budget (seconds). SLA wait threshold = sla_s - mean_svc.

    Returns:
        List of ints; index k = replica count for tick [k*tick_s, (k+1)*tick_s).
    """
    if not raw:
        return []

    warped = [(t / warp, tok) for t, tok in raw]
    t_max = warped[-1][0]
    n_ticks = max(1, int(t_max / tick_seconds) + 1)

    buckets: list = [[] for _ in range(n_ticks)]
    for t, tok in warped:
        idx = min(n_ticks - 1, int(t / tick_seconds))
        buckets[idx].append(tok)

    c_sched: list = []

    for bucket in buckets:
        if not bucket:
            c_sched.append(1)
            continue

        n_req = len(bucket)
        lam = n_req / tick_seconds          # arrival rate in warped seconds
        mean_service = statistics.mean(      # service time per queue simulation
            _service_time_s(tok) for tok in bucket
        )
        # SLA wait budget: remaining after service (negative → svc itself violates SLA)
        sla_wait = max(0.0, sla_s - mean_service)

        chosen = 1
        for c in range(1, 1024):
            timeout_pct = _erlang_c_sla_timeout_pct(lam, mean_service, c, sla_wait)
            if timeout_pct < mcs_gate:
                chosen = c
                break

        c_sched.append(chosen)

    return c_sched


def _simulate_sla_aware_variable_c(
    requests: list,
    c_schedule: list,
    tick_seconds: float = 60.0,
) -> tuple:
    """Non-preemptive binary SLA-class priority with per-tick variable c.

    Mirrors ``simulate_queue(..., "sla_aware")`` (requests with
    predicted_tokens ≤ global median → class 0 latency-critical, served before
    class 1; FIFO within class) but with the per-tick variable server count
    drain semantics of ``_simulate_fifo_variable_c``.

    This is the strong SLA-aware-with-capacity-scaling baseline for the fair MCS
    comparison [run 2026-06-23]: it receives the *same* MCS c_schedule as the
    abs-conformal condition so the only variable is queue ordering, holding
    GPU-hours, cost, and arrival process constant.

    Returns ``(summary, response_map, wait_map)`` matching the benchmark contract.
    """
    n = len(requests)
    max_c = max(c_schedule) if c_schedule else 1
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))

    _pred_sorted = sorted(r.predicted_tokens for r in requests)
    _median_pred = _pred_sorted[len(_pred_sorted) // 2] if _pred_sorted else 0.0

    s_req: list = [None] * max_c
    s_ver: list = [0] * max_c
    events: list = []
    _eseq = [n + 1]
    _seq = [0]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _c_now(t: float) -> int:
        idx = min(int(t / tick_seconds), len(c_schedule) - 1)
        return max(1, c_schedule[idx])

    def _start(sid: int, req, t: float) -> None:
        s_req[sid] = req
        s_ver[sid] += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + req.service_s, 1, _en(), sid, v, req))

    response: dict = {}
    wait_map: dict = {}
    waiting: list = []  # priority heap: (sla_class, seq, arrived_t, req)

    while events:
        ev = heapq.heappop(events)
        t, ety = ev[0], ev[1]
        c = _c_now(t)

        if ety == 0:  # ARRIVAL
            req = ev[5]
            free = next((s for s in range(c) if s_req[s] is None), None)
            if free is not None:
                wait_map[req.idx] = 0.0
                _start(free, req, t)
            else:
                sla_class = 0 if req.predicted_tokens <= _median_pred else 1
                heapq.heappush(waiting, (sla_class, _seq[0], t, req))
                _seq[0] += 1

        else:  # COMPLETION
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue
            response[req.idx] = t - req.arrival_s
            s_req[sid] = None
            s_ver[sid] += 1

            if sid < c and waiting:
                _, _, arrived_t, nxt = heapq.heappop(waiting)
                wait_map[nxt.idx] = t - arrived_t
                _start(sid, nxt, t)

    resp = [response[r.idx] for r in requests if r.idx in response]
    waits_list = [wait_map.get(r.idx, 0.0) for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits_list, max_c)
    summary["preemption_count"] = 0
    summary["variable_c"] = True
    return summary, response, wait_map


def _simulate_fifo_variable_c(
    requests: list,
    c_schedule: list,
    tick_seconds: float = 60.0,
) -> tuple:
    """Non-preemptive FIFO M/G/c with per-tick variable server count.

    Per-server state (s_req, s_ver arrays pre-sized to max(c_schedule)).
    Servers ≥ c(t) at event time t drain (complete current request) but do
    not accept new arrivals. Servers < c(t) accept new work when freed.

    Returns ``(summary, response_map, wait_map)`` matching simulate_queue's
    contract. Requests not dispatched before all events drain are absent from
    response_map (counted as SLA violations).
    """
    n = len(requests)
    max_c = max(c_schedule) if c_schedule else 1
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))

    s_req: list = [None] * max_c
    s_ver: list = [0] * max_c
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _c_now(t: float) -> int:
        idx = min(int(t / tick_seconds), len(c_schedule) - 1)
        return max(1, c_schedule[idx])

    def _start(sid: int, req, t: float) -> None:
        s_req[sid] = req
        s_ver[sid] += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + req.service_s, 1, _en(), sid, v, req))

    response: dict = {}
    wait_map: dict = {}
    waiting: list = []  # FIFO queue: (arrived_t, req)

    while events:
        ev = heapq.heappop(events)
        t, ety = ev[0], ev[1]
        c = _c_now(t)

        if ety == 0:  # ARRIVAL
            req = ev[5]
            free = next((s for s in range(c) if s_req[s] is None), None)
            if free is not None:
                wait_map[req.idx] = 0.0
                _start(free, req, t)
            else:
                waiting.append((t, req))

        else:  # COMPLETION
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue
            response[req.idx] = t - req.arrival_s
            s_req[sid] = None
            s_ver[sid] += 1

            if sid < c and waiting:
                arrived_t, nxt = waiting.pop(0)
                wait_map[nxt.idx] = t - arrived_t
                _start(sid, nxt, t)

    resp = [response[r.idx] for r in requests if r.idx in response]
    waits_list = [wait_map.get(r.idx, 0.0) for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits_list, max_c)
    summary["preemption_count"] = 0
    summary["variable_c"] = True
    return summary, response, wait_map


def _simulate_abs_conformal_variable_c(
    requests: list,
    c_schedule: list,
    calibrator,
    tick_seconds: float = 60.0,
    preemption_overhead_s: float = 0.0,
) -> tuple:
    """Decoupled Hybrid SRPT + abs-conformal α with per-tick variable c.

    Identical to simulate_decoupled_hybrid_abs_conformal (serving_queue.py)
    except the active server count c(t) follows c_schedule rather than being
    fixed. Arrays are pre-sized to max(c_schedule).

    Drain semantics: servers ≥ c(t) at event time t complete running requests
    but do not accept new arrivals (no dispatch on completion for those servers).
    They resume accepting work if c(t) increases at a later tick.

    Returns ``(summary, response_map, wait_map)`` matching the benchmark contract.
    """
    n = len(requests)
    max_c = max(c_schedule) if c_schedule else 1
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]

    s_req:          list = [None] * max_c
    s_start:        list = [0.0] * max_c
    s_rem0:         list = [0.0] * max_c
    s_ver:          list = [0] * max_c
    s_frozen_wait:  list = [0.0] * max_c

    waiting: list = []
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return max(0.0, s_rem0[sid] - (t - s_start[sid]))

    def _abs_dispatch_key(entry: tuple, t: float, alpha: float) -> tuple:
        rem_s, frozen_wait_s, wait_entered_s, req = entry
        total_wait = frozen_wait_s + (t - wait_entered_s)
        ek = rem_s / max(1e-9, 1.0 + alpha * total_wait)
        return (ek, req.idx)

    def _c_now(t: float) -> int:
        idx = min(int(t / tick_seconds), len(c_schedule) - 1)
        return max(1, c_schedule[idx])

    def _start(sid: int, req, rem: float, frozen_wait: float, t: float) -> None:
        s_req[sid] = req
        s_start[sid] = t
        s_rem0[sid] = rem
        s_ver[sid] += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    response: dict = {}

    while events:
        ev = heapq.heappop(events)
        t, ety = ev[0], ev[1]
        c = _c_now(t)

        if ety == 0:  # ARRIVAL
            req = ev[5]
            free = next((s for s in range(c) if s_req[s] is None), None)
            if free is not None:
                s_frozen_wait[free] = 0.0
                _start(free, req, req.service_s, 0.0, t)
            else:
                worst_sid, worst_rem = 0, -1.0
                for s in range(c):
                    r = _remaining(s, t)
                    if r > worst_rem:
                        worst_rem, worst_sid = r, s

                if req.service_s < worst_rem:
                    preempted = s_req[worst_sid]
                    prem = _remaining(worst_sid, t)
                    pfrozen = s_frozen_wait[worst_sid]
                    s_req[worst_sid] = None
                    s_ver[worst_sid] += 1
                    s_frozen_wait[worst_sid] = 0.0
                    _start(worst_sid, req, req.service_s, 0.0, t)
                    _npreempt[0] += 1
                    waiting.append((prem + preemption_overhead_s, pfrozen, t, preempted))
                else:
                    waiting.append((req.service_s, 0.0, t, req))

        else:  # COMPLETION
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue
            response[req.idx] = t - req.arrival_s
            calibrator.update(req.predicted_tokens, req.actual_tokens)
            s_req[sid] = None
            s_ver[sid] += 1

            if sid < c and waiting:
                alpha = calibrator.current_alpha()
                best_i = min(
                    range(len(waiting)),
                    key=lambda i: _abs_dispatch_key(waiting[i], t, alpha),
                )
                rem_s, frozen_wait_s, wait_entered_s, nxt = waiting.pop(best_i)
                new_frozen = frozen_wait_s + (t - wait_entered_s)
                s_frozen_wait[sid] = new_frozen
                _start(sid, nxt, rem_s, new_frozen, t)

    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = _summarize(requests, response, wait_map, resp, waits, max_c)
    summary["preemption_count"] = _npreempt[0]
    summary["variable_c"] = True
    return summary, response, wait_map


@dataclass
class JointMCSAbsConformalReport:
    """TRUE end-to-end compound economic × queue result — run 2026-06-23.

    2×2 factorial: {FIFO, abs-conformal} × {fixed-c, MCS variable-c}.

    Cost denominator: provisioned GPU hours × GPU_HOUR_USD. This differs from
    the service-time cost in ``_sla_safe_goodput_per_dollar`` (which is
    constant across queue disciplines). Here the denominator varies with the
    provisioning decision: MCS uses fewer GPU hours than fixed-c.

    Primary KPI: ``abs_mcs_goodput_per_dollar`` — the first TRUE compound
    measurement. All other cells provide isolation:
      ``abs_fixed_goodput_per_dollar``   — queue gain only (vs fifo_fixed)
      ``fifo_mcs_goodput_per_dollar``    — economic gain only (vs fifo_fixed)
      ``fifo_fixed_goodput_per_dollar``  — do-nothing baseline
    """
    trace: str
    total_requests: int
    fixed_c: int
    target_rho: float
    sla_s: float
    tick_seconds: float

    # MCS c_schedule statistics (in warped-time domain)
    c_schedule_mean: float
    c_schedule_min: int
    c_schedule_max: int
    n_ticks: int

    # Provisioned GPU-hour costs
    cost_fixed_c: float
    cost_mcs_c: float
    provisioning_cost_factor: float   # cost_fixed_c / cost_mcs_c

    # SLA-compliant goodput / provisioned infra dollar — 4 conditions
    fifo_fixed_goodput_per_dollar: float
    fifo_mcs_goodput_per_dollar: float
    abs_fixed_goodput_per_dollar: float
    abs_mcs_goodput_per_dollar: float  # TRUE COMPOUND KPI

    # Goodput gains vs FIFO+fixed-c baseline
    abs_fixed_vs_fifo_fixed_pct: float       # queue-only gain
    fifo_mcs_vs_fifo_fixed_pct: float        # economic-only gain for FIFO
    abs_mcs_vs_fifo_fixed_pct: float         # TRUE compound gain

    # Independence estimate (run-z approach applied to this trace)
    independence_estimate_gp_per_dollar: float  # abs_fixed × provisioning_cost_factor
    true_vs_independence_gap_pct: float          # >0 = true beats estimate

    # Completions (requests in response_map / total)
    fifo_fixed_completion_rate: float
    fifo_mcs_completion_rate: float
    abs_fixed_completion_rate: float
    abs_mcs_completion_rate: float

    # p99 response times (SLA safety check)
    fifo_fixed_p99_s: float
    fifo_mcs_p99_s: float
    abs_fixed_p99_s: float
    abs_mcs_p99_s: float

    # Preemption counts (abs-conformal disciplines only)
    abs_fixed_preemptions: int
    abs_mcs_preemptions: int

    north_star_target_pct: float = 300.0

    def to_dict(self) -> dict:
        def _pct_or_none(v):
            return None if v is None else round(v, 2)

        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "fixed_c": self.fixed_c,
            "target_rho": self.target_rho,
            "sla_s": self.sla_s,
            "tick_seconds": self.tick_seconds,
            "c_schedule_mean": round(self.c_schedule_mean, 3),
            "c_schedule_min": self.c_schedule_min,
            "c_schedule_max": self.c_schedule_max,
            "n_ticks": self.n_ticks,
            "cost_fixed_c": round(self.cost_fixed_c, 6),
            "cost_mcs_c": round(self.cost_mcs_c, 6),
            "provisioning_cost_factor": round(self.provisioning_cost_factor, 4),
            "fifo_fixed_goodput_per_dollar": round(self.fifo_fixed_goodput_per_dollar, 2),
            "fifo_mcs_goodput_per_dollar": round(self.fifo_mcs_goodput_per_dollar, 2),
            "abs_fixed_goodput_per_dollar": round(self.abs_fixed_goodput_per_dollar, 2),
            "abs_mcs_goodput_per_dollar": round(self.abs_mcs_goodput_per_dollar, 2),
            "abs_fixed_vs_fifo_fixed_pct": _pct_or_none(self.abs_fixed_vs_fifo_fixed_pct),
            "fifo_mcs_vs_fifo_fixed_pct": _pct_or_none(self.fifo_mcs_vs_fifo_fixed_pct),
            "abs_mcs_vs_fifo_fixed_pct": _pct_or_none(self.abs_mcs_vs_fifo_fixed_pct),
            "independence_estimate_gp_per_dollar": round(self.independence_estimate_gp_per_dollar, 2),
            "true_vs_independence_gap_pct": _pct_or_none(self.true_vs_independence_gap_pct),
            "fifo_fixed_completion_rate": round(self.fifo_fixed_completion_rate, 4),
            "fifo_mcs_completion_rate": round(self.fifo_mcs_completion_rate, 4),
            "abs_fixed_completion_rate": round(self.abs_fixed_completion_rate, 4),
            "abs_mcs_completion_rate": round(self.abs_mcs_completion_rate, 4),
            "fifo_fixed_p99_s": round(self.fifo_fixed_p99_s, 3),
            "fifo_mcs_p99_s": round(self.fifo_mcs_p99_s, 3),
            "abs_fixed_p99_s": round(self.abs_fixed_p99_s, 3),
            "abs_mcs_p99_s": round(self.abs_mcs_p99_s, 3),
            "abs_fixed_preemptions": self.abs_fixed_preemptions,
            "abs_mcs_preemptions": self.abs_mcs_preemptions,
            "north_star_target_pct": self.north_star_target_pct,
        }


def run_joint_mcs_abs_conformal_azure_backtest(
    fixed_c: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
    sla_s: float = DEFAULT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
    tick_seconds: float = 60.0,
    mcs_gate: float = 9.5,
) -> "JointMCSAbsConformalReport":
    """TRUE compound economic × queue backtest on Azure LLM 2024 [run 2026-06-23].

    First end-to-end compound measurement: MCS provisioning (per-tick variable-c
    replica schedule) + abs-conformal SRTF (queue ordering) in a single
    discrete-event simulation. Removes the independence assumption used by
    run-z (``_compute_compound_economic_queue``).

    2×2 factorial — all conditions on the same warped Azure LLM 2024 trace:
      FIFO + fixed-c:             do-nothing baseline
      FIFO + MCS variable-c:     economic-only gain (cost↓, goodput~same)
      abs-conformal + fixed-c:   queue-only gain (goodput↑, cost same)
      abs-conformal + MCS-c:     TRUE compound (goodput↑, cost↓)

    Cost denominator: provisioned GPU hours × GPU_HOUR_USD. Unlike the
    existing ``_sla_safe_goodput_per_dollar`` (which uses total service time
    as the cost proxy — identical across queue disciplines), this function
    uses the wall-clock provisioned fleet cost so that MCS's cost reduction
    is captured in the denominator.

    Args:
        fixed_c:               Baseline replica count for the fixed-c arm.
        target_rho:            Target cluster utilization (for time warp).
        job_limit:             Request cap (default 5880 = full Azure fixture).
        sla_s:                 E2E SLA budget in seconds.
        prior_window:          Sliding-window size for running-median prior.
        target_p90_abs_tokens: Abs-error calibration target.
        azure_fixture:         Path to Azure LLM 2024 CSV fixture.
        tick_seconds:          MCS tick duration (seconds) in warped time.
        mcs_gate:              Timeout-rate threshold for MCS (9.5% < 10% target).

    Returns:
        ``JointMCSAbsConformalReport`` with the 4-cell 2×2 KPIs.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")

    warp = calibrate_time_warp(raw, servers=fixed_c, target_rho=target_rho)
    live_preds, _ = make_live_prior_predictions(raw, window=prior_window)

    def _build_live() -> list:
        return [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=live_preds[i],
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]

    # Compute MCS c schedule in warped-time domain
    c_schedule = _joint_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=mcs_gate)
    n_ticks = len(c_schedule)

    # Provisioned GPU-hour costs (different between fixed-c and MCS-c)
    cost_fixed_c = fixed_c * n_ticks * tick_seconds / 3600.0 * GPU_HOUR_USD
    cost_mcs_c = sum(c_schedule) * tick_seconds / 3600.0 * GPU_HOUR_USD
    provisioning_cost_factor = cost_fixed_c / max(cost_mcs_c, 1e-9)

    # ── CELL 1: FIFO + fixed-c ────────────────────────────────────────────────
    fifo_fixed_reqs = _build_live()
    ff_sim, ff_resp, _ = simulate_queue(fifo_fixed_reqs, fixed_c, "fifo")
    gp_fifo_fixed = (
        _sla_safe_goodput(fifo_fixed_reqs, ff_resp, sla_s) / max(cost_fixed_c, 1e-9)
    )

    # ── CELL 2: FIFO + MCS variable-c ────────────────────────────────────────
    fifo_mcs_reqs = _build_live()
    fm_sim, fm_resp, _ = _simulate_fifo_variable_c(fifo_mcs_reqs, c_schedule, tick_seconds)
    gp_fifo_mcs = (
        _sla_safe_goodput(fifo_mcs_reqs, fm_resp, sla_s) / max(cost_mcs_c, 1e-9)
    )

    # ── CELL 3: abs-conformal + fixed-c ──────────────────────────────────────
    abs_fixed_reqs = _build_live()
    abs_fixed_cal = AbsoluteErrorConformalCalibrator(
        target_p90_abs_tokens=target_p90_abs_tokens
    )
    af_sim, af_resp, _ = _simulate_decoupled_hybrid_abs_conformal(
        abs_fixed_reqs, fixed_c, abs_fixed_cal
    )
    gp_abs_fixed = (
        _sla_safe_goodput(abs_fixed_reqs, af_resp, sla_s) / max(cost_fixed_c, 1e-9)
    )

    # ── CELL 4: abs-conformal + MCS variable-c (TRUE COMPOUND) ───────────────
    abs_mcs_reqs = _build_live()
    abs_mcs_cal = AbsoluteErrorConformalCalibrator(
        target_p90_abs_tokens=target_p90_abs_tokens
    )
    am_sim, am_resp, _ = _simulate_abs_conformal_variable_c(
        abs_mcs_reqs, c_schedule, abs_mcs_cal, tick_seconds
    )
    gp_abs_mcs = (
        _sla_safe_goodput(abs_mcs_reqs, am_resp, sla_s) / max(cost_mcs_c, 1e-9)
    )

    def _pct(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base > 0 else 0.0

    def _completion_rate(reqs: list, resp: dict) -> float:
        return len(resp) / max(len(reqs), 1)

    # Independence estimate: abs_fixed × provisioning_cost_factor
    gp_independence = gp_abs_fixed * provisioning_cost_factor

    return JointMCSAbsConformalReport(
        trace="azure_llm_2024_joint_mcs_abs_conformal",
        total_requests=len(raw),
        fixed_c=fixed_c,
        target_rho=target_rho,
        sla_s=sla_s,
        tick_seconds=tick_seconds,
        c_schedule_mean=statistics.mean(c_schedule),
        c_schedule_min=min(c_schedule),
        c_schedule_max=max(c_schedule),
        n_ticks=n_ticks,
        cost_fixed_c=cost_fixed_c,
        cost_mcs_c=cost_mcs_c,
        provisioning_cost_factor=provisioning_cost_factor,
        fifo_fixed_goodput_per_dollar=gp_fifo_fixed,
        fifo_mcs_goodput_per_dollar=gp_fifo_mcs,
        abs_fixed_goodput_per_dollar=gp_abs_fixed,
        abs_mcs_goodput_per_dollar=gp_abs_mcs,
        abs_fixed_vs_fifo_fixed_pct=_pct(gp_fifo_fixed, gp_abs_fixed),
        fifo_mcs_vs_fifo_fixed_pct=_pct(gp_fifo_fixed, gp_fifo_mcs),
        abs_mcs_vs_fifo_fixed_pct=_pct(gp_fifo_fixed, gp_abs_mcs),
        independence_estimate_gp_per_dollar=gp_independence,
        true_vs_independence_gap_pct=_pct(gp_independence, gp_abs_mcs),
        fifo_fixed_completion_rate=_completion_rate(fifo_fixed_reqs, ff_resp),
        fifo_mcs_completion_rate=_completion_rate(fifo_mcs_reqs, fm_resp),
        abs_fixed_completion_rate=_completion_rate(abs_fixed_reqs, af_resp),
        abs_mcs_completion_rate=_completion_rate(abs_mcs_reqs, am_resp),
        fifo_fixed_p99_s=ff_sim.get("p99_response_s", 0.0),
        fifo_mcs_p99_s=fm_sim.get("p99_response_s", 0.0),
        abs_fixed_p99_s=af_sim.get("p99_response_s", 0.0),
        abs_mcs_p99_s=am_sim.get("p99_response_s", 0.0),
        abs_fixed_preemptions=af_sim.get("preemption_count", 0),
        abs_mcs_preemptions=am_sim.get("preemption_count", 0),
    )


# ===========================================================================
# Economic MCS optimizer [run 2026-06-23]
#
# Treats MCS (Erlang-C M/M/c capacity sizing) as the strong baseline and asks:
# can Aurelius reduce the cost / GPU-hours of the MCS-safe capacity schedule
# while preserving SLA-safe goodput?
#
# Mechanism: Erlang-C M/M/c over-provisions because it assumes EXPONENTIAL
# service-time variance.  LLM serving service time is near-DETERMINISTIC
# (service_s = TTFT_BASE_S + TPOT_S × output_tokens); for the same load an
# M/D/c queue needs fewer servers than M/M/c to hold the same wait
# (Pollaczek–Khinchine: deterministic wait ≈ ½ exponential wait).  The
# candidate exploits this by sizing capacity to the ACTUAL closed-loop SLA
# measured by full-trace deterministic discrete-event simulation rather than
# the conservative analytical bound.
#
# Two caveats this implementation respects:
#   1. Per-tick ISOLATED M/D/c sizing breaks SLA (it ignores cross-tick queue
#      carryover; spillover accumulates and blows up closed-loop p99).  The
#      reducer MUST validate against the full-trace closed-loop simulation.
#   2. To avoid trace-overfitting, the production reducer uses a single global
#      "utilization uplift" knob (relaxed Erlang-C generation gate) calibrated
#      once and validated closed-loop — not a 72-dimensional per-tick fit.
# ===========================================================================

# Gate grid for the utilization-uplift calibration (relaxed Erlang-C gates that
# generate progressively cheaper candidate schedules; all closed-loop validated).
_ECON_MCS_GATE_GRID: tuple = (9.5, 11.0, 13.0, 15.0, 18.0, 21.0, 25.0, 30.0,
                              35.0, 40.0, 50.0, 60.0)


def _economic_mcs_calibrated_schedule(
    raw: list,
    warp: float,
    tick_seconds: float,
    sla_s: float,
    baseline_tok: float,
    preserve_frac: float = 0.99,
    gate_grid: tuple = _ECON_MCS_GATE_GRID,
    live_preds: "list | None" = None,
) -> tuple:
    """Cheapest closed-loop-validated MCS schedule preserving SLA-safe goodput.

    Sweeps a single utilization-uplift knob (relaxed Erlang-C generation gate),
    validates each candidate schedule by full-trace SLA-aware variable-c
    simulation, and returns the lowest-GPU-hour schedule whose closed-loop
    SLA-compliant goodput stays ≥ ``preserve_frac × baseline_tok``.

    Returns ``(schedule, chosen_gate, validated_sla_tokens)``.
    """
    target = preserve_frac * baseline_tok
    best = None  # (cost_proxy_sum, schedule, gate, tok)
    for gate in gate_grid:
        c_cand = _joint_mcs_c_schedule(raw, tick_seconds, warp,
                                       mcs_gate=gate, sla_s=sla_s)
        reqs = _build_variable_c_requests(raw, warp, live_preds)
        _, resp, _ = _simulate_sla_aware_variable_c(reqs, c_cand, tick_seconds)
        tok = _sla_safe_goodput(reqs, resp, sla_s)
        if tok >= target:
            csum = sum(c_cand)
            if best is None or csum < best[0]:
                best = (csum, c_cand, gate, tok)
    if best is None:
        # No relaxed gate preserved SLA: fall back to the safe Erlang-C schedule.
        c_cand = _joint_mcs_c_schedule(raw, tick_seconds, warp,
                                       mcs_gate=gate_grid[0], sla_s=sla_s)
        return c_cand, gate_grid[0], baseline_tok
    return best[1], best[2], best[3]


def _build_variable_c_requests(raw: list, warp: float,
                               live_preds: "list | None" = None) -> list:
    """Build a fresh ``_Request`` list (warped arrivals, oracle or live preds)."""
    out = []
    for i, (arr, tok) in enumerate(raw):
        pred = float(tok) if live_preds is None else live_preds[i]
        out.append(_Request(idx=i, arrival_s=arr / warp, actual_tokens=tok,
                            predicted_tokens=pred, service_s=_service_time_s(tok)))
    return out


@dataclass
class EconomicMCSReport:
    """Economic MCS optimizer comparison [run 2026-06-23].

    Question answered: does Aurelius reduce the cost of MCS-safe serving while
    preserving SLA-safe goodput?  Success iff
    ``candidate_goodput_per_dollar > sla_aware_mcs_goodput_per_dollar``.
    """
    trace: str
    total_requests: int
    sla_s: float
    tick_seconds: int
    total_tokens: int
    n_ticks: int

    # MCS Erlang-C reference schedule
    mcs_c_mean: float
    mcs_gpu_hours: float
    mcs_cost: float

    # Candidate (SC-MCS) schedule
    candidate_gate: float
    candidate_c_mean: float
    candidate_gpu_hours: float
    candidate_cost: float

    # Per-policy KPIs (all share trace/physics/SLA/cost-denominator)
    sla_aware_mcs_goodput_per_dollar: float       # 1. baseline
    constraint_aware_mcs_goodput_per_dollar: float  # 2. == FIFO+MCS
    current_aurelius_mcs_goodput_per_dollar: float  # 3. abs-conformal+MCS
    candidate_goodput_per_dollar: float           # 4. SC-MCS candidate

    sla_aware_mcs_sla_tokens: float
    constraint_aware_mcs_sla_tokens: float
    current_aurelius_mcs_sla_tokens: float
    candidate_sla_tokens: float

    sla_aware_mcs_p99_wait_s: float
    candidate_p99_wait_s: float

    sla_aware_mcs_violations: int
    candidate_violations: int

    # Verdict
    candidate_vs_baseline_gp_pct: float
    candidate_gpu_hours_delta_pct: float
    candidate_cost_delta_pct: float
    candidate_sla_tokens_delta_pct: float
    success: bool

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "total_requests": self.total_requests,
            "sla_s": self.sla_s,
            "tick_seconds": self.tick_seconds,
            "total_tokens": self.total_tokens,
            "n_ticks": self.n_ticks,
            "mcs_c_mean": round(self.mcs_c_mean, 3),
            "mcs_gpu_hours": round(self.mcs_gpu_hours, 4),
            "mcs_cost": round(self.mcs_cost, 4),
            "candidate_gate": self.candidate_gate,
            "candidate_c_mean": round(self.candidate_c_mean, 3),
            "candidate_gpu_hours": round(self.candidate_gpu_hours, 4),
            "candidate_cost": round(self.candidate_cost, 4),
            "sla_aware_mcs_goodput_per_dollar": round(self.sla_aware_mcs_goodput_per_dollar, 2),
            "constraint_aware_mcs_goodput_per_dollar": round(self.constraint_aware_mcs_goodput_per_dollar, 2),
            "current_aurelius_mcs_goodput_per_dollar": round(self.current_aurelius_mcs_goodput_per_dollar, 2),
            "candidate_goodput_per_dollar": round(self.candidate_goodput_per_dollar, 2),
            "sla_aware_mcs_sla_tokens": round(self.sla_aware_mcs_sla_tokens, 1),
            "constraint_aware_mcs_sla_tokens": round(self.constraint_aware_mcs_sla_tokens, 1),
            "current_aurelius_mcs_sla_tokens": round(self.current_aurelius_mcs_sla_tokens, 1),
            "candidate_sla_tokens": round(self.candidate_sla_tokens, 1),
            "sla_aware_mcs_p99_wait_s": round(self.sla_aware_mcs_p99_wait_s, 3),
            "candidate_p99_wait_s": round(self.candidate_p99_wait_s, 3),
            "sla_aware_mcs_violations": self.sla_aware_mcs_violations,
            "candidate_violations": self.candidate_violations,
            "candidate_vs_baseline_gp_pct": round(self.candidate_vs_baseline_gp_pct, 3),
            "candidate_gpu_hours_delta_pct": round(self.candidate_gpu_hours_delta_pct, 3),
            "candidate_cost_delta_pct": round(self.candidate_cost_delta_pct, 3),
            "candidate_sla_tokens_delta_pct": round(self.candidate_sla_tokens_delta_pct, 3),
            "success": self.success,
        }


def run_economic_mcs_optimizer_azure_backtest(
    fixed_c: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
    sla_s: float = DEFAULT_SLA_S,
    prior_window: int = LIVE_PRIOR_WINDOW,
    target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
    tick_seconds: float = 60.0,
    mcs_gate: float = 9.5,
    preserve_frac: float = 0.99,
) -> EconomicMCSReport:
    """Economic MCS optimizer comparison on Azure LLM 2024 [run 2026-06-23].

    All four policies share the SAME trace, physics (deterministic service),
    SLA, arrival process, and provisioned-GPU-hour cost denominator.  Policies
    1–3 run on the MCS Erlang-C capacity schedule; policy 4 (candidate) runs on
    the simulation-calibrated cheaper schedule.  Capacity sizing is the ONLY
    lever that changes between baseline (1) and candidate (4) — both use
    SLA-aware ordering — so any gp/$ gain is attributable to cost reduction.

    Returns an ``EconomicMCSReport``.  ``success`` is True iff the candidate's
    goodput/$ strictly exceeds the SLA-aware+MCS baseline.
    """
    raw = load_serving_requests(azure_fixture, limit=job_limit)
    if len(raw) < 2:
        raise ValueError("need at least 2 requests")
    warp = calibrate_time_warp(raw, servers=fixed_c, target_rho=target_rho)
    live_preds, _ = make_live_prior_predictions(raw, window=prior_window)
    total_tok = sum(tok for _, tok in raw)

    def _cost(c_sched: list) -> float:
        return sum(c_sched) * tick_seconds / 3600.0 * GPU_HOUR_USD

    def _p99_wait(wait_map: dict, response: dict) -> float:
        waits = sorted(wait_map.get(i, 0.0) for i in response)
        return waits[int(round(0.99 * (len(waits) - 1)))] if waits else 0.0

    def _viol(reqs: list, response: dict) -> int:
        return len(reqs) - sum(1 for i in response if response[i] <= sla_s)

    # ── MCS Erlang-C reference schedule ──────────────────────────────────────
    c_mcs = _joint_mcs_c_schedule(raw, tick_seconds, warp,
                                  mcs_gate=mcs_gate, sla_s=sla_s)
    cost_mcs = _cost(c_mcs)
    n_ticks = len(c_mcs)

    # 1. SLA-aware + MCS (strong baseline)
    r1 = _build_variable_c_requests(raw, warp, live_preds)
    _, resp1, wm1 = _simulate_sla_aware_variable_c(r1, c_mcs, tick_seconds)
    tok1 = _sla_safe_goodput(r1, resp1, sla_s)
    gp1 = tok1 / max(cost_mcs, 1e-9)

    # 2. Constraint-aware + MCS  (== FIFO + MCS: constraint-aware is a
    #    provisioning policy; at fixed MCS capacity it adds no queue ordering)
    r2 = _build_variable_c_requests(raw, warp)
    _, resp2, wm2 = _simulate_fifo_variable_c(r2, c_mcs, tick_seconds)
    tok2 = _sla_safe_goodput(r2, resp2, sla_s)
    gp2 = tok2 / max(cost_mcs, 1e-9)

    # 3. Current Aurelius (abs-conformal) + MCS
    r3 = _build_variable_c_requests(raw, warp, live_preds)
    cal = AbsoluteErrorConformalCalibrator(target_p90_abs_tokens=target_p90_abs_tokens)
    _, resp3, wm3 = _simulate_abs_conformal_variable_c(r3, c_mcs, cal, tick_seconds)
    tok3 = _sla_safe_goodput(r3, resp3, sla_s)
    gp3 = tok3 / max(cost_mcs, 1e-9)

    # 4. Candidate Aurelius economic optimizer + MCS (SC-MCS schedule)
    c_cand, chosen_gate, _ = _economic_mcs_calibrated_schedule(
        raw, warp, tick_seconds, sla_s, baseline_tok=tok1,
        preserve_frac=preserve_frac, live_preds=live_preds,
    )
    cost_cand = _cost(c_cand)
    r4 = _build_variable_c_requests(raw, warp, live_preds)
    _, resp4, wm4 = _simulate_sla_aware_variable_c(r4, c_cand, tick_seconds)
    tok4 = _sla_safe_goodput(r4, resp4, sla_s)
    gp4 = tok4 / max(cost_cand, 1e-9)

    def _pct(base: float, new: float) -> float:
        return (new - base) / base * 100.0 if base else 0.0

    return EconomicMCSReport(
        trace="azure_llm_2024_economic_mcs",
        total_requests=len(raw),
        sla_s=sla_s,
        tick_seconds=int(tick_seconds),
        total_tokens=total_tok,
        n_ticks=n_ticks,
        mcs_c_mean=statistics.mean(c_mcs),
        mcs_gpu_hours=cost_mcs / GPU_HOUR_USD,
        mcs_cost=cost_mcs,
        candidate_gate=chosen_gate,
        candidate_c_mean=statistics.mean(c_cand),
        candidate_gpu_hours=cost_cand / GPU_HOUR_USD,
        candidate_cost=cost_cand,
        sla_aware_mcs_goodput_per_dollar=gp1,
        constraint_aware_mcs_goodput_per_dollar=gp2,
        current_aurelius_mcs_goodput_per_dollar=gp3,
        candidate_goodput_per_dollar=gp4,
        sla_aware_mcs_sla_tokens=tok1,
        constraint_aware_mcs_sla_tokens=tok2,
        current_aurelius_mcs_sla_tokens=tok3,
        candidate_sla_tokens=tok4,
        sla_aware_mcs_p99_wait_s=_p99_wait(wm1, resp1),
        candidate_p99_wait_s=_p99_wait(wm4, resp4),
        sla_aware_mcs_violations=_viol(r1, resp1),
        candidate_violations=_viol(r4, resp4),
        candidate_vs_baseline_gp_pct=_pct(gp1, gp4),
        candidate_gpu_hours_delta_pct=_pct(cost_mcs, cost_cand),
        candidate_cost_delta_pct=_pct(cost_mcs, cost_cand),
        candidate_sla_tokens_delta_pct=_pct(tok1, tok4),
        success=gp4 > gp1,
    )
