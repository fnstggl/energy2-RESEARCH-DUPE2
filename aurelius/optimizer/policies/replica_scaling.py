"""Replica-scaling decision policy — Phase 2/3 architecture extraction.

Extracts the per-tick server-count provisioning logic (SOTSS-MIN / AMCSG MCS)
out of the benchmark monolith into the canonical AureliusOptimizer policy seam.
Follows the same extraction pattern as Phase 2's serving_queue.py.

Decisions governed here:
    - Per-tick replica count (c_schedule) via Erlang-C M/M/c gate sweep (AMCSG)
    - Oracle-loop refinement of cheapest safe c_schedule (SOTSS-MIN)

The benchmark imports ``compute_mcs_c_schedule`` and ``compute_sotss_min_schedule``
back and makes the existing ``_joint_mcs_c_schedule`` / ``_sotss_min_cost_schedule``
thin delegates, so AureliusOptimizer now governs all provisioning decisions.

No circular imports: benchmark → policy (one direction only).
"""

from __future__ import annotations

import heapq
import math
import statistics
from dataclasses import dataclass
from typing import Optional

from .base import OptimizationPolicy

# ---------------------------------------------------------------------------
# Service-time constants — canonical owner (identical to benchmark originals)
# ---------------------------------------------------------------------------
REPLICA_TTFT_BASE_S: float = 0.150
REPLICA_TPOT_S: float = 0.020

# ---------------------------------------------------------------------------
# MCS gate defaults
# ---------------------------------------------------------------------------
REPLICA_SAFE_GATE: float = 12.5       # ceiling gate: AMCSG best-safe schedule
REPLICA_AGGRESSIVE_GATE: float = 100.0  # SOTSS-MIN: minimum stable c per tick

# ---------------------------------------------------------------------------
# SOTSS oracle iteration cap
# ---------------------------------------------------------------------------
REPLICA_MAX_ORACLE_ITERS: int = 500

# ---------------------------------------------------------------------------
# Online SOTSS EWMA decay
# ---------------------------------------------------------------------------
REPLICA_OSOTSS_EWMA_ALPHA: float = 0.1

# ---------------------------------------------------------------------------
# ICP / deployability classification of capacity modes (audit 2026-06-25).
#
# The capacity DECISION — how many ON-DEMAND replicas to run per serving queue —
# is a GPU-FLEET-OPERATOR lever (you autoscale your OWN fleet). But several modes
# here are NOT for the operator ICP and are kept RESEARCH-ONLY:
#   * the SPOT-fleet machinery (GSF / ZFHC / AFMS, ``spot_fraction``, the
#     interruption model) is CLOUD-TENANT arbitrage — buying discounted
#     preemptible instances on someone else's cloud, NOT an operator decision
#     (``research/MCS_AUDIT.md``); and
#   * the ORACLE modes peek at tick-t actual tokens / arrivals.
# Only ``forecasted_mcs`` (on-demand; forecasts arrivals + service from data
# ≤ t-1) is a DEPLOYABLE operator capacity policy — it is the ``optimize_fleet``
# capacity default. NB: even forecasted_mcs is ≈0% over a reactive autoscaler on
# Azure (Phase C) — capacity sizing is not where Aurelius's value is; do not
# invest here. The ``c=4`` fixed schedule is a demoted STRAWMAN, never a baseline.
# ---------------------------------------------------------------------------
DEPLOYABLE_MODES: frozenset = frozenset({"forecasted_mcs"})
RESEARCH_ONLY_MODES: dict = {
    "amcsg": "oracle (tick-t actual arrivals + tokens)",
    "sotss_min": "oracle (actual tokens)",
    "sotss_gsf": "oracle + SPOT-fleet (cloud-tenant arbitrage, out of ICP)",
    "online_sotss": "arrival-oracle (actual tick-t arrival counts)",
}


def is_deployable_mode(mode: str) -> bool:
    """True only for on-demand, operator-deployable capacity modes.

    The single deployable mode is ``forecasted_mcs``. The oracle and spot-fleet
    modes are research-only (see ``RESEARCH_ONLY_MODES`` for the reason each is
    excluded from the GPU-fleet-operator product surface).
    """
    return mode in DEPLOYABLE_MODES


def _replica_service_time_s(output_tokens: int) -> float:
    """Service time for a request with the given number of output tokens."""
    return REPLICA_TTFT_BASE_S + output_tokens * REPLICA_TPOT_S


def _replica_calibrate_warp(
    raw: list[tuple[float, int]],
    servers: int,
    target_rho: float,
) -> float:
    """Time-warp scalar that yields ``target_rho`` utilization on ``servers`` servers.

    Identical to ``calibrate_time_warp`` in the benchmark; canonical owner here.
    """
    if len(raw) < 2:
        return 1.0
    span = raw[-1][0] - raw[0][0]
    if span <= 0:
        return 1.0
    lam_raw = len(raw) / span
    mean_service = statistics.mean(_replica_service_time_s(tok) for _, tok in raw)
    if lam_raw <= 0 or mean_service <= 0:
        return 1.0
    return target_rho * servers / (lam_raw * mean_service)


def _replica_erlang_c_sla_timeout_pct(
    lam: float,
    mean_service_s: float,
    c: int,
    sla_wait_threshold_s: float,
) -> float:
    """Fraction of M/M/c arrivals that wait longer than sla_wait_threshold_s (%).

    Identical to ``_erlang_c_sla_timeout_pct`` in the benchmark monolith — this
    module is the canonical owner. Returns 100.0 when the system is overloaded
    (per-server utilisation ρ ≥ 1).
    """
    mu = 1.0 / max(mean_service_s, 1e-12)
    a = lam / mu          # total traffic intensity (Erlangs)
    rho = a / max(c, 1)   # per-server utilisation

    if rho >= 1.0:
        return 100.0

    log_a = math.log(a) if a > 1e-12 else -1e9
    log_ac_over_cfact = c * log_a - sum(math.log(k) for k in range(1, c + 1))

    log_sum_terms: list[float] = []
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

    excess_rate = c * mu - lam
    prob_exceed = erlang_c_prob * math.exp(-excess_rate * sla_wait_threshold_s)
    return min(100.0, max(0.0, prob_exceed * 100.0))


def compute_mcs_c_schedule(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    mcs_gate: float = REPLICA_SAFE_GATE,
    sla_s: float = 10.0,
) -> list[int]:
    """Per-tick MCS replica counts using Erlang-C M/M/c gate formula.

    Finds the minimum ``c`` per tick where ``P(queue_wait > sla_s − mean_service)``
    is below ``mcs_gate`` percent. Empty ticks return 1 replica.

    Identical algorithm to ``_joint_mcs_c_schedule`` in the benchmark monolith —
    this module is the canonical owner; the benchmark delegates back here.

    Args:
        raw:          ``(arrival_s_unwarped, output_tokens)`` tuples.
        tick_seconds: Tick duration in warped seconds.
        warp:         Time-warp factor (arrival_warped = arrival_raw / warp).
        mcs_gate:     Timeout-rate threshold (%). Ceiling gate = 12.5%.
        sla_s:        E2E SLA budget in seconds (default 10 s).

    Returns:
        ``list[int]`` — per-tick replica count; index k covers
        ``[k*tick_s, (k+1)*tick_s)`` in warped time.
    """
    if not raw:
        return []

    warped = [(t / warp, tok) for t, tok in raw]
    t_max = warped[-1][0]
    n_ticks = max(1, int(t_max / tick_seconds) + 1)

    buckets: list[list[int]] = [[] for _ in range(n_ticks)]
    for t, tok in warped:
        idx = min(n_ticks - 1, int(t / tick_seconds))
        buckets[idx].append(tok)

    c_sched: list[int] = []
    for bucket in buckets:
        if not bucket:
            c_sched.append(1)
            continue

        n_req = len(bucket)
        lam = n_req / tick_seconds
        mean_service = statistics.mean(_replica_service_time_s(tok) for tok in bucket)
        sla_wait = max(0.0, sla_s - mean_service)

        chosen = 1
        for c in range(1, 1024):
            if _replica_erlang_c_sla_timeout_pct(lam, mean_service, c, sla_wait) < mcs_gate:
                chosen = c
                break

        c_sched.append(chosen)

    return c_sched


def _oracle_fifo_response_times(
    pairs: list[tuple[float, float]],
    c_schedule: list[int],
    tick_seconds: float = 60.0,
) -> dict[int, float]:
    """Non-preemptive FIFO M/G/c oracle: returns ``{orig_idx → response_time}``.

    Semantically equivalent to ``_simulate_fifo_variable_c`` but accepts
    ``(arrival_s_warped, service_s)`` pairs directly instead of ``_Request``
    objects, and omits KPI summarisation. Used exclusively by the SOTSS oracle
    loop to avoid a circular import on the benchmark module.

    Drain semantics: servers ≥ c(t) drain (complete current request) but do not
    accept new arrivals. Stale-event detection via per-server version counter.

    Args:
        pairs:        ``(arrival_s_warped, service_s)`` for each request.
        c_schedule:   Per-tick server count; index ``k`` covers tick
                      ``[k*tick_seconds, (k+1)*tick_seconds)``.
        tick_seconds: Tick duration (seconds).

    Returns:
        ``{orig_idx: response_time_s}``; requests not dispatched before all
        events drain are absent (counted as SLA violations by the caller).
    """
    n = len(pairs)
    if n == 0:
        return {}

    max_c = max(c_schedule) if c_schedule else 1
    # Arrival order: sort by (arrival_s, orig_idx) — same tie-break as benchmark
    order = sorted(range(n), key=lambda i: (pairs[i][0], i))

    s_req: list[Optional[int]] = [None] * max_c
    s_ver: list[int] = [0] * max_c
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    # Arrival events: (arrival_s, type=0, sort_pos, -1, -1, orig_idx)
    for pos, orig_idx in enumerate(order):
        arr_s = pairs[orig_idx][0]
        heapq.heappush(events, (arr_s, 0, pos, -1, -1, orig_idx))

    def _c_now(t: float) -> int:
        idx = min(int(t / tick_seconds), len(c_schedule) - 1)
        return max(1, c_schedule[idx])

    # Completion events: (compl_s, type=1, seq, sid, ver, orig_idx, arr_s)
    def _start(sid: int, orig_idx: int, svc_s: float, arr_s: float, t: float) -> None:
        s_req[sid] = orig_idx
        s_ver[sid] += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + svc_s, 1, _en(), sid, v, orig_idx, arr_s))

    response: dict[int, float] = {}
    waiting: list[tuple[float, int]] = []  # FIFO queue: (arrival_t, orig_idx)

    while events:
        ev = heapq.heappop(events)
        t, ety = ev[0], ev[1]
        c = _c_now(t)

        if ety == 0:  # ARRIVAL
            orig_idx = ev[5]
            arr_s, svc_s = pairs[orig_idx]
            free = next((s for s in range(c) if s_req[s] is None), None)
            if free is not None:
                _start(free, orig_idx, svc_s, arr_s, t)
            else:
                waiting.append((t, orig_idx))

        else:  # COMPLETION: (t, 1, seq, sid, ver, orig_idx, arr_s)
            _, _, _, sid, ver, orig_idx, arr_s = ev
            if ver != s_ver[sid]:
                continue
            response[orig_idx] = t - arr_s
            s_req[sid] = None
            s_ver[sid] += 1

            if sid < c and waiting:
                _, nxt_idx = waiting.pop(0)
                nxt_arr_s, nxt_svc_s = pairs[nxt_idx]
                _start(sid, nxt_idx, nxt_svc_s, nxt_arr_s, t)

    return response


def _oracle_stochastic_response_times(
    pairs: list[tuple[float, float]],
    c_schedule: list[int],
    spot_fraction: float = 0.95,
    zfhc_threshold: int = 8,
    p_interrupt_hourly: float = 0.10,
    tick_seconds: float = 60.0,
    seed: int = 42,
) -> dict[int, float]:
    """Stochastic oracle simulation with GSF spot interruptions.

    Applies a Binomial spot-interruption model to ``c_schedule``, producing
    ``c_effective`` per tick, then runs the deterministic FIFO oracle on the
    reduced schedule.  Provides the oracle loop in
    ``compute_sotss_gsf_schedule`` with a stochastic view of capacity so it
    detects ticks that are vulnerable to spot interruptions — not just
    M/G/c queue violations.

    The interruption draw is reproduced identically every call with the same
    ``seed``, so the oracle converges deterministically (fixed-scenario SAA).

    Args:
        pairs:              ``(arrival_s_warped, service_s)`` per request.
        c_schedule:         Per-tick total server count.
        spot_fraction:      GSF spot fraction (≤1.0). Matches the evaluation.
        zfhc_threshold:     All-spot threshold for large fleets (≥ this → all spot).
        p_interrupt_hourly: Per-spot-instance hourly interruption probability.
        tick_seconds:       Tick duration in seconds.
        seed:               RNG seed — must match the evaluation seed for a
                            fair same-conditions comparison.

    Returns:
        ``{orig_idx: response_time_s}`` from deterministic FIFO on c_effective.
    """
    try:
        import numpy as _np
    except ImportError:
        raise ImportError(
            "numpy is required for _oracle_stochastic_response_times / "
            "compute_sotss_gsf_schedule"
        )

    rng = _np.random.default_rng(seed)
    p_survive = (1.0 - p_interrupt_hourly) ** (tick_seconds / 3600.0)

    c_effective: list[int] = []
    for c in c_schedule:
        if c >= zfhc_threshold:
            c_spot = c  # all-spot above ZFHC threshold (identical to backtest)
        else:
            c_spot = min(c, max(round(spot_fraction * c), c - 1))  # GSF formula
        c_demand = c - c_spot
        survived = int(rng.binomial(c_spot, p_survive)) if c_spot > 0 else 0
        c_effective.append(max(1, c_demand + survived))

    return _oracle_fifo_response_times(pairs, c_effective, tick_seconds)


def compute_sotss_min_schedule(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    sla_s: float,
    safe_gate: float = REPLICA_SAFE_GATE,
    aggressive_gate: float = REPLICA_AGGRESSIVE_GATE,
    max_iters: int = REPLICA_MAX_ORACLE_ITERS,
    baseline_n_sla_safe: Optional[int] = None,
) -> tuple[list[int], int, int, int, int]:
    """SOTSS oracle loop: start cheap, selectively increment c on violation ticks.

    Identical algorithm to ``_sotss_min_cost_schedule`` in the benchmark monolith —
    this module is the canonical owner; the benchmark delegates back here.

    Starts from the ``aggressive_gate``-% Erlang-C schedule (cheapest stable c per
    tick) and increments the worst-violation tick's c by 1 each iteration until
    ``n_sla_safe ≥ baseline_n_sla_safe``, capped by ``safe_gate``-% ceiling.

    Args:
        raw:                  ``(arrival_s_unwarped, output_tokens)`` tuples.
        tick_seconds:         Tick duration in warped seconds.
        warp:                 Time-warp scalar.
        sla_s:                E2E SLA budget in seconds.
        safe_gate:            Ceiling gate (%) — AMCSG best-safe schedule.
        aggressive_gate:      Starting gate (%) — SOTSS-MIN uses 100.0 (min stable c).
        max_iters:            Hard iteration cap.
        baseline_n_sla_safe:  Safety floor override; if ``None``, computed from
                              gate=9.5% deterministic simulation.

    Returns:
        ``(c_schedule, n_iters, initial_violations, n_ticks_cheaper,
        baseline_n_sla_safe_used)``
    """
    c_ceil = list(compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=safe_gate, sla_s=sla_s))
    c_sched = list(compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=aggressive_gate, sla_s=sla_s))
    n_ticks = len(c_sched)

    # (arrival_s_warped, service_s) pairs for the oracle FIFO simulator
    pairs = [(arr / warp, _replica_service_time_s(tok)) for arr, tok in raw]

    if baseline_n_sla_safe is None:
        c_base = list(compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=9.5, sla_s=sla_s))
        resp_base = _oracle_fifo_response_times(pairs, c_base, tick_seconds)
        baseline_n_sla_safe = sum(
            1 for i in range(len(pairs)) if i in resp_base and resp_base[i] <= sla_s
        )

    initial_violations: Optional[int] = None
    n_iters = 0

    for iteration in range(max_iters):
        resp = _oracle_fifo_response_times(pairs, c_sched, tick_seconds)

        n_sla_safe = sum(1 for i in range(len(pairs)) if i in resp and resp[i] <= sla_s)

        if initial_violations is None:
            initial_violations = len(pairs) - n_sla_safe

        n_iters = iteration + 1

        if n_sla_safe >= baseline_n_sla_safe:
            break

        violators = [i for i in range(len(pairs)) if i not in resp or resp[i] > sla_s]
        if not violators:
            break

        tick_counts: dict[int, int] = {}
        for i in violators:
            t_idx = min(int(pairs[i][0] / tick_seconds), n_ticks - 1)
            tick_counts[t_idx] = tick_counts.get(t_idx, 0) + 1

        sorted_ticks = sorted(tick_counts, key=lambda k: tick_counts[k], reverse=True)
        incremented = False
        for tk in sorted_ticks:
            if c_sched[tk] < c_ceil[tk]:
                c_sched[tk] += 1
                incremented = True
                break

        if not incremented:
            break

    n_ticks_cheaper = sum(1 for i in range(n_ticks) if c_sched[i] < c_ceil[i])
    return c_sched, n_iters, initial_violations or 0, n_ticks_cheaper, baseline_n_sla_safe


def compute_sotss_gsf_schedule(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    sla_s: float,
    safe_gate: float = REPLICA_SAFE_GATE,
    aggressive_gate: float = REPLICA_AGGRESSIVE_GATE,
    spot_fraction: float = 0.95,
    zfhc_threshold: int = 8,
    p_interrupt_hourly: float = 0.10,
    seed: int = 42,
    max_iters: int = REPLICA_MAX_ORACLE_ITERS,
    baseline_n_sla_safe: Optional[int] = None,
) -> tuple[list[int], int, int, int, int]:
    """SOTSS-GSF: stochastic oracle for spot-interruption-aware capacity planning.

    Extends ``compute_sotss_min_schedule`` by replacing the deterministic FIFO
    simulation in the oracle loop with a stochastic GSF simulation that includes
    spot interruptions.  This allows the oracle to detect and fix ticks that are
    vulnerable to spot interruptions — specifically those where reduced
    ``c_effective`` (after a Binomial interruption draw) causes M/G/c queue
    violations that the purely deterministic oracle misses.

    Algorithm
    ---------
    1. Ceiling schedule: ``gate=safe_gate`` Erlang-C (AMCSG best-safe).
    2. Starting schedule: ``gate=aggressive_gate`` Erlang-C (default 100% = c=1).
    3. **Stochastic oracle loop** (replaces deterministic FIFO in SOTSS-MIN):
       a. Draw per-tick ``c_effective`` via Binomial(c_spot, p_survive) with ``seed``.
       b. Run deterministic FIFO on ``c_effective``.
       c. Count SLA-safe requests.
       d. Increment c on the most-violated tick (bounded by ceiling) if
          ``n_sla_safe < baseline_n_sla_safe``.
       e. Repeat until safe or iteration cap reached.

    The ``baseline_n_sla_safe`` defaults to the AMCSG (safe_gate%) **stochastic**
    n_sla_safe with the same ``seed``.  This ensures SOTSS-GSF meets the same
    stochastic safety floor as AMCSG — the apples-to-apples safety comparison.

    Oracle class note
    -----------------
    Like SOTSS-MIN, this is an **oracle-class** algorithm: it uses actual output
    tokens from the trace AND the fixed spot-interruption realisation (``seed``).
    It should be compared against AMCSG (also oracle-class on actual tokens).
    Results should be labelled: "ORACLE — not directly deployable without a
    live token forecast and historical interruption model."

    Args:
        raw:                 ``(arrival_s_unwarped, output_tokens)`` tuples.
        tick_seconds:        Tick duration in warped seconds.
        warp:                Time-warp scalar.
        sla_s:               E2E SLA budget in seconds.
        safe_gate:           Ceiling gate (%) — AMCSG best-safe schedule.
        aggressive_gate:     Starting gate (%) — 100.0 for minimum stable c.
        spot_fraction:       GSF spot fraction (must match the evaluation).
        zfhc_threshold:      All-spot threshold for large fleets.
        p_interrupt_hourly:  Per-spot-instance hourly interruption probability.
        seed:                RNG seed (must match evaluation seed for a fair
                             same-conditions comparison).
        max_iters:           Hard iteration cap.
        baseline_n_sla_safe: Safety floor; if ``None``, computed from AMCSG
                             stochastic evaluation with the same ``seed``.

    Returns:
        ``(c_schedule, n_iters, initial_violations, n_ticks_cheaper,
         baseline_n_sla_safe_used)``
    """
    c_ceil = list(
        compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=safe_gate, sla_s=sla_s)
    )
    c_sched = list(
        compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=aggressive_gate, sla_s=sla_s)
    )
    n_ticks = len(c_sched)

    pairs = [(arr / warp, _replica_service_time_s(tok)) for arr, tok in raw]

    if baseline_n_sla_safe is None:
        # Safety floor = AMCSG stochastic n_sla_safe (same seed, same physics)
        resp_base = _oracle_stochastic_response_times(
            pairs, c_ceil, spot_fraction, zfhc_threshold, p_interrupt_hourly,
            tick_seconds, seed,
        )
        baseline_n_sla_safe = sum(
            1 for i in range(len(pairs)) if i in resp_base and resp_base[i] <= sla_s
        )

    initial_violations: Optional[int] = None
    n_iters = 0

    for iteration in range(max_iters):
        resp = _oracle_stochastic_response_times(
            pairs, c_sched, spot_fraction, zfhc_threshold, p_interrupt_hourly,
            tick_seconds, seed,
        )

        n_sla_safe = sum(
            1 for i in range(len(pairs)) if i in resp and resp[i] <= sla_s
        )

        if initial_violations is None:
            initial_violations = len(pairs) - n_sla_safe

        n_iters = iteration + 1

        if n_sla_safe >= baseline_n_sla_safe:
            break

        violators = [
            i for i in range(len(pairs))
            if i not in resp or resp[i] > sla_s
        ]
        if not violators:
            break

        tick_counts: dict[int, int] = {}
        for i in violators:
            t_idx = min(int(pairs[i][0] / tick_seconds), n_ticks - 1)
            tick_counts[t_idx] = tick_counts.get(t_idx, 0) + 1

        sorted_ticks = sorted(
            tick_counts, key=lambda k: tick_counts[k], reverse=True
        )
        incremented = False
        for tk in sorted_ticks:
            if c_sched[tk] < c_ceil[tk]:
                c_sched[tk] += 1
                incremented = True
                break

        if not incremented:
            break

    n_ticks_cheaper = sum(1 for i in range(n_ticks) if c_sched[i] < c_ceil[i])
    return c_sched, n_iters, initial_violations or 0, n_ticks_cheaper, baseline_n_sla_safe


def compute_online_sotss_schedule(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    sla_s: float,
    safe_gate: float = REPLICA_SAFE_GATE,
    aggressive_gate: float = REPLICA_AGGRESSIVE_GATE,
    max_iters: int = REPLICA_MAX_ORACLE_ITERS,
    baseline_n_sla_safe: Optional[int] = None,
    ewma_alpha: float = REPLICA_OSOTSS_EWMA_ALPHA,
    ewma_mode: str = "fixed",
    burst_threshold: float = 1.5,
    burst_alpha: float = 0.5,
    burst_cooldown_ticks: int = 2,
    interrupt_safety_margin: int = 0,
    borderline_margin_s: float = 0.0,
) -> tuple[list[int], int, int, int, int]:
    """Online SOTSS: SOTSS oracle loop with causal EWMA service-time predictions.

    Production-deployable variant of ``compute_sotss_min_schedule``: replaces
    oracle actual-token service times with causal per-tick EWMA predictions
    built from past observations only. The oracle loop structure is otherwise
    identical to SOTSS-MIN.

    Causal prediction: for each request arriving in tick k, the predicted service
    time is the EWMA of per-tick mean service times observed in ticks 0..k-1.
    Requests in tick 0 use the global mean as warm-start prior.  This makes the
    algorithm production-deployable — no future token counts are accessed.

    The final SLA evaluation (in the benchmark harness) always uses actual
    service times, so reported goodput/$ reflects real-world performance.

    Args:
        raw:                     ``(arrival_s_unwarped, output_tokens)`` tuples.
        tick_seconds:            Tick duration in warped seconds.
        warp:                    Time-warp scalar.
        sla_s:                   E2E SLA budget in seconds.
        safe_gate:               Ceiling gate (%) — AMCSG best-safe schedule.
        aggressive_gate:         Starting gate (%) — minimum stable c.
        max_iters:               Hard iteration cap.
        baseline_n_sla_safe:     Safety floor override.
        ewma_alpha:              EWMA decay for per-tick mean prediction (default 0.1).
        interrupt_safety_margin: Extra SLA-safe requests the oracle must achieve above
            ``baseline_n_sla_safe`` before converging.  Compensates for the
            stochastic/deterministic mismatch: the oracle uses deterministic FIFO
            (no spot interruptions) while the evaluation uses stochastic Binomial
            interruptions.  Default 0 preserves byte-identical behavior.
        borderline_margin_s:     After primary convergence (violators=[]), add capacity
            to ticks containing requests whose deterministic response time is within
            this margin of the SLA limit.  These are the ticks most vulnerable to
            stochastic spot interruptions reducing effective capacity.  Uses actual
            service-time pairs (correct SLA guarantee).  Default 0.0 disables the
            Oracle Soft-SLA Continuation phase (byte-identical to pre-OSSC behavior).

    Returns:
        ``(c_schedule, n_iters, initial_violations, n_ticks_cheaper,
        baseline_n_sla_safe_used)``
    """
    if not raw:
        return [], 0, 0, 0, 0

    c_ceil = list(compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=safe_gate, sla_s=sla_s))
    c_sched = list(compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=aggressive_gate, sla_s=sla_s))
    n_ticks = len(c_sched)

    # Build causal EWMA predicted service time per tick.
    # Global mean warm-starts the EWMA so tick 0 requests get a reasonable prior.
    global_mean_svc = statistics.mean(_replica_service_time_s(tok) for _, tok in raw)

    warped = [(t / warp, tok) for t, tok in raw]
    n_ticks_build = max(1, int(warped[-1][0] / tick_seconds) + 1)

    tick_svcs: list[list[float]] = [[] for _ in range(n_ticks_build)]
    for arr_w, tok in warped:
        idx = min(n_ticks_build - 1, int(arr_w / tick_seconds))
        tick_svcs[idx].append(_replica_service_time_s(tok))

    # predicted_svc_per_tick[k] = EWMA prediction before observing tick k
    ewma_val = global_mean_svc
    predicted_svc_per_tick: list[float] = []
    burst_cooldown_remaining = 0
    for bucket in tick_svcs:
        predicted_svc_per_tick.append(ewma_val)  # emit BEFORE updating (causal)
        if bucket:
            tick_mean = statistics.mean(bucket)
            if ewma_mode == "adaptive":
                # Boost alpha when actual load spikes above threshold × current EWMA
                # to track burst patterns faster. Purely causal: decision made after
                # observing tick_mean but before the next prediction.
                if tick_mean > burst_threshold * ewma_val:
                    burst_cooldown_remaining = burst_cooldown_ticks
                alpha_t = burst_alpha if burst_cooldown_remaining > 0 else ewma_alpha
                if burst_cooldown_remaining > 0:
                    burst_cooldown_remaining -= 1
            else:
                alpha_t = ewma_alpha
            ewma_val = alpha_t * tick_mean + (1.0 - alpha_t) * ewma_val

    predicted_pairs: list[tuple[float, float]] = []
    for arr_w, tok in warped:
        t_idx = min(n_ticks_build - 1, int(arr_w / tick_seconds))
        predicted_pairs.append((arr_w, predicted_svc_per_tick[t_idx]))

    # Actual service-time pairs for convergence checking — uses real token counts
    # to guarantee the deployed schedule actually meets the SLA baseline, not just
    # the predicted schedule.  This is the dual-simulation design:
    #   - violation identification: predicted pairs (causal, no future tokens)
    #   - convergence criterion:    actual pairs  (correct SLA guarantee)
    actual_pairs: list[tuple[float, float]] = [
        (arr / warp, _replica_service_time_s(tok)) for arr, tok in raw
    ]

    if baseline_n_sla_safe is None:
        c_base = list(compute_mcs_c_schedule(raw, tick_seconds, warp, mcs_gate=9.5, sla_s=sla_s))
        resp_base = _oracle_fifo_response_times(actual_pairs, c_base, tick_seconds)
        baseline_n_sla_safe = sum(
            1 for i in range(len(actual_pairs)) if i in resp_base and resp_base[i] <= sla_s
        )

    initial_violations: Optional[int] = None
    n_iters = 0

    for iteration in range(max_iters):
        # Convergence check uses actual service times → correct SLA guarantee
        resp_actual = _oracle_fifo_response_times(actual_pairs, c_sched, tick_seconds)
        n_sla_safe = sum(1 for i in range(len(actual_pairs)) if i in resp_actual and resp_actual[i] <= sla_s)

        if initial_violations is None:
            initial_violations = len(actual_pairs) - n_sla_safe

        n_iters = iteration + 1

        # Convergence target includes interrupt_safety_margin to account for the
        # stochastic/deterministic mismatch: the oracle runs deterministic FIFO
        # while the final evaluation uses stochastic spot interruptions.
        if n_sla_safe >= baseline_n_sla_safe + interrupt_safety_margin:
            break

        # Violation identification uses predicted service times → causal/production-safe
        resp_pred = _oracle_fifo_response_times(predicted_pairs, c_sched, tick_seconds)
        violators = [i for i in range(len(predicted_pairs)) if i not in resp_pred or resp_pred[i] > sla_s]
        if not violators:
            # Predicted shows no violations; fall back to actual violators
            violators = [i for i in range(len(actual_pairs)) if i not in resp_actual or resp_actual[i] > sla_s]
        if not violators:
            break

        tick_counts: dict[int, int] = {}
        for i in violators:
            t_idx = min(int(predicted_pairs[i][0] / tick_seconds), n_ticks - 1)
            tick_counts[t_idx] = tick_counts.get(t_idx, 0) + 1

        sorted_ticks = sorted(tick_counts, key=lambda k: tick_counts[k], reverse=True)
        incremented = False
        for tk in sorted_ticks:
            if c_sched[tk] < c_ceil[tk]:
                c_sched[tk] += 1
                incremented = True
                break

        if not incremented:
            break

    # Oracle Soft-SLA Continuation (OSSC): after primary convergence
    # (violators=[]), add capacity to ticks with borderline response times.
    # Uses actual service-time pairs so the guarantee is preserved.
    if borderline_margin_s > 0.0:
        remaining_iters = max_iters - n_iters
        for _ in range(remaining_iters):
            resp_bl = _oracle_fifo_response_times(actual_pairs, c_sched, tick_seconds)
            borderline = [
                i for i in range(len(actual_pairs))
                if i in resp_bl and sla_s - borderline_margin_s < resp_bl[i] <= sla_s
            ]
            if not borderline:
                break
            tick_counts_bl: dict[int, int] = {}
            for i in borderline:
                t_idx = min(int(actual_pairs[i][0] / tick_seconds), n_ticks - 1)
                tick_counts_bl[t_idx] = tick_counts_bl.get(t_idx, 0) + 1
            incremented_bl = False
            for tk in sorted(tick_counts_bl, key=lambda k: tick_counts_bl[k], reverse=True):
                if c_sched[tk] < c_ceil[tk]:
                    c_sched[tk] += 1
                    incremented_bl = True
                    break
            if not incremented_bl:
                break
            n_iters += 1

    n_ticks_cheaper = sum(1 for i in range(n_ticks) if c_sched[i] < c_ceil[i])
    return c_sched, n_iters, initial_violations or 0, n_ticks_cheaper, baseline_n_sla_safe


# ---------------------------------------------------------------------------
# C1-Protected Gate Sweep (C1PGS) — run 2026-06-23
# ---------------------------------------------------------------------------
# Motivation:
#   At gate=25% the Erlang-C schedule assigns c=1 on low-load ticks.  With the
#   standard GSF formula, c=1 means 1 spot replica — one interruption drops
#   c_effective to 0 and causes an SLA violation.  On BurstGPT the stochastic
#   Binomial model shows 3-4 violations per run at gate=25%, making it unsafe.
#
#   C1PGS eliminates this risk: whenever c=1, use 0 spot + 1 on-demand.
#   The on-demand replica cannot be interrupted.  For c>1 the standard GSF
#   formula applies (higher c tolerates one interruption; c_effective≥c-1≥1).
#
# Cost comparison (f=0.95, spot=$0.80/hr, on-demand=$2.00/hr):
#   gate=12.5%, c=4, GSF:   4×$0.80 = $3.20/hr
#   gate=25%,  c=1, C1PGS:  1×$2.00 = $2.00/hr  → saves $1.20/hr per such tick
#
# Research basis:
#   SpotServe (arXiv:2311.15566, ASPLOS 2024): minimum on-demand reserve at low
#     capacity prevents SLA cliff from preemption.
#   DynamoLLM (arXiv:2408.00741): guard empty-tick c_effective drop.
# ---------------------------------------------------------------------------


def compute_c1pgs_spot_replicas(
    c: int,
    spot_fraction: float = 0.95,
    zfhc_threshold: int = 8,
) -> int:
    """C1-protected spot replicas: at c=1 use 0 spot (1 on-demand only).

    Eliminates the BurstGPT safety cliff where a single spot interruption at a
    minimum-capacity (c=1) tick reduces c_effective to 0 and causes an SLA
    violation.  For c>1 the standard GSF formula is used — one interruption
    leaves c_effective≥c-1≥1, which remains SLA-safe.

    Args:
        c:               Total replica count for this tick.
        spot_fraction:   GSF spot fraction for c>1 (default 0.95).
        zfhc_threshold:  All-spot threshold for large fleets (default 8).

    Returns:
        Number of spot replicas (on-demand = c − return_value).
    """
    if c == 1:
        return 0  # on-demand only — no interruption risk at minimum capacity
    if c >= zfhc_threshold:
        return c  # all-spot for large fleets (ZFHC)
    return min(c, max(round(spot_fraction * c), c - 1))  # standard GSF


# ---------------------------------------------------------------------------
# Policy dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReplicaScalingConfig:
    """Configuration for :class:`ReplicaScalingPolicy`."""

    mode: str = "sotss_min"
    """Provisioning algorithm:
    ``"amcsg"`` (Erlang-C gate sweep — oracle: actual tick-t arrivals + tokens),
    ``"sotss_min"`` (SOTSS oracle from minimum stable c — oracle),
    ``"online_sotss"`` (causal EWMA *service-time* prediction, but still uses
    actual tick-t arrival counts — **arrival-oracle**, not fully deployable), or
    ``"forecasted_mcs"`` (fully deployable: forecasts BOTH next-tick arrivals AND
    service time from data ≤ t-1; see ``aurelius/benchmarks/forecasted_mcs.py``)."""

    tick_seconds: float = 60.0
    """Tick duration in warped seconds."""

    target_rho: float = 0.85
    """Target per-server utilisation for ``_replica_calibrate_warp``."""

    sla_s: float = 10.0
    """E2E SLA budget in seconds."""

    servers: int = 1
    """Server count used for warp calibration when ``warp`` is not supplied."""

    safe_gate_pct: float = REPLICA_SAFE_GATE
    """Ceiling gate percentage for the AMCSG / SOTSS ceiling schedule."""

    aggressive_gate_pct: float = REPLICA_AGGRESSIVE_GATE
    """Starting gate percentage for the SOTSS oracle loop."""

    max_oracle_iters: int = REPLICA_MAX_ORACLE_ITERS
    """Hard iteration cap for the SOTSS oracle loop."""

    spot_fraction: float = 0.95
    """GSF spot fraction for sotss_gsf mode."""

    zfhc_threshold: int = 8
    """All-spot threshold for sotss_gsf mode."""

    p_interrupt_hourly: float = 0.10
    """Per-spot-instance hourly interruption probability for sotss_gsf mode."""

    seed: int = 42
    """RNG seed for sotss_gsf stochastic oracle (must match evaluation seed)."""

    ewma_alpha: float = REPLICA_OSOTSS_EWMA_ALPHA
    """EWMA decay for Online SOTSS causal service-time prediction (default 0.1)."""

    ewma_mode: str = "fixed"
    """EWMA tracking mode: ``"fixed"`` (constant alpha) or ``"adaptive"`` (burst-
    sensitive alpha boost when actual tick load > ``burst_threshold`` × EWMA)."""

    burst_threshold: float = 1.5
    """Load ratio above which adaptive EWMA boosts alpha (``ewma_mode="adaptive"``)."""

    burst_alpha: float = 0.5
    """Elevated EWMA alpha used during burst cooldown (``ewma_mode="adaptive"``)."""

    burst_cooldown_ticks: int = 2
    """Ticks the boosted alpha persists after a burst is detected."""

    interrupt_safety_margin: int = 0
    """Extra SLA-safe requests the oracle must reach above ``baseline_n_sla_safe``
    before converging.  Compensates for the stochastic/deterministic mismatch:
    the oracle uses deterministic FIFO (no spot interruptions) while the GSF
    evaluation includes Binomial interruptions.  Set to the expected
    interruption-induced SLA misses for the target trace and spot fraction
    (e.g. 15–25 for BurstGPT at p_interrupt=10%/hr, spot_fraction=0.95).
    Default 0 preserves byte-identical behavior with the pre-margin OSOTSS."""

    borderline_margin_s: float = 0.0
    """Oracle Soft-SLA Continuation (OSSC) margin in seconds.  After primary
    convergence (violators=[]), add capacity to ticks whose requests have
    deterministic response times within this margin of the SLA limit.  These
    ticks are most vulnerable to stochastic spot interruptions.  Uses actual
    service-time pairs (correct SLA guarantee).  Default 0.0 disables OSSC
    (byte-identical to pre-OSSC behavior)."""

    baseline_n_sla_safe: Optional[int] = None
    """Safety-floor override for the ``online_sotss`` oracle loop.  When set, the
    oracle converges once predicted n_sla_safe ≥ this value; when ``None`` the
    oracle computes its own floor from a deterministic FIFO simulation (which
    differs from the stochastic AMCSG baseline used by the validated backtest).
    Set to ``amcsg_n_sla_safe`` from the AMCSG stochastic GSF evaluation to
    reproduce the validated OSOTSS result through the canonical path."""

    # ---- Phase 3e: backtest serving modes (constraint_aware / safe_high_utilization) ----
    ca_target_rho: float = 0.65
    """Target per-server utilisation for constraint_aware mode (default 0.65).
    Ignored for all other modes."""

    adaptive_frontier_window: Optional[int] = None
    """Phase 4: causal rolling-window rho adaptation via frontier estimation.
    When set to an integer W, enables per-tick rho selection: for tick k < W,
    uses ca_target_rho; for tick k >= W, calls estimate_frontier over the past
    W ticks and picks the highest SAFE rho, falling back to ca_target_rho on
    INSUFFICIENT_TELEMETRY.  ``None`` (default) disables adaptation and
    preserves byte-identical constraint_aware behavior.  Only applies to
    ``mode="constraint_aware"``."""

    # ---- forecasted_mcs mode (fully deployable; forecasts arrivals + service) ----
    forecast_method: str = "ewma"
    """``forecasted_mcs`` sub-method: ``"ewma"``, ``"quantile"``, or ``"lag1"``."""

    forecast_ewma_alpha: float = 0.5
    """EWMA smoothing for the forecasted_mcs arrival/service forecast."""

    forecast_count_window: int = 8
    """Rolling window for the forecasted_mcs quantile / safety-buffer."""

    forecast_quantile: float = 0.90
    """Rolling arrival quantile for ``forecast_method="quantile"``."""

    forecast_safety_k: float = 0.0
    """One-sided arrival safety buffer (units of recent count std)."""

    forecast_warmup_c: int = 4
    """Cold-start capacity for the first forecasted_mcs tick (no history)."""


@dataclass
class ReplicaScalingResult:
    """Result returned by :class:`ReplicaScalingPolicy`."""

    mode: str
    """Algorithm used: ``"amcsg"``, ``"sotss_min"``, ``"sotss_gsf"``, or ``"online_sotss"``."""

    c_schedule: list
    """Per-tick replica count ``list[int]``."""

    c_mean: float
    """Mean per-tick replica count across all ticks."""

    n_ticks: int
    """Number of ticks in the schedule."""

    warp: float
    """Time-warp scalar used."""

    oracle_iters: int
    """Oracle iterations consumed (0 for ``amcsg`` mode)."""

    n_ticks_cheaper: int
    """Ticks cheaper than the safe-gate ceiling (0 for ``amcsg`` mode)."""

    baseline_n_sla_safe: int
    """Safety floor used by oracle (0 for ``amcsg`` mode)."""

    initial_violations: int = 0
    """Initial FIFO-violation count before the oracle started iterating.
    Non-zero for oracle modes (``sotss_min``, ``sotss_gsf``, ``online_sotss``);
    0 for ``amcsg`` mode."""


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------

class ReplicaScalingPolicy(OptimizationPolicy):
    """Canonical replica-scaling provisioning policy — Phase 2/3.

    Governs the per-tick server count (c_schedule) for serving queues, replacing
    the equivalent logic previously owned by the benchmark monolith.

    Modes:
        ``"amcsg"``        — per-tick Erlang-C MCS gate sweep (deterministic,
                             matches the AMCSG baseline at ``safe_gate_pct=12.5``).
        ``"sotss_min"``    — SOTSS oracle loop starting from minimum stable c
                             (``aggressive_gate_pct=100.0``), the frontier algorithm
                             that achieves +6.29% goodput/$ vs AMCSG on Azure.
        ``"sotss_gsf"``    — SOTSS-GSF stochastic oracle; uses spot interruptions
                             in the oracle loop (oracle-class, not production-safe).
        ``"online_sotss"`` — SOTSS with causal EWMA service-time predictions.
                             NOTE: still sizes from actual tick-t arrival counts
                             (``compute_mcs_c_schedule``) and uses actual arrival
                             times in the violation sim — i.e. an **arrival-oracle**,
                             not fully deployable. Fixes future-tokens, not
                             future-arrivals.
        ``"forecasted_mcs"`` — Fully deployable: forecasts BOTH next-tick arrivals
                             AND service time from data <= t-1 (no tick-t actuals).
                             Delegates to ``aurelius.benchmarks.forecasted_mcs``.
                             ``forecast_method`` selects ewma / quantile / lag1.

    Modes other than ``forecasted_mcs`` are the extracted benchmark algorithms
    (identical constants/tie-breaks; NOT a research change). ``forecasted_mcs`` is
    the only mode that uses no future information — see ``research/MCS_AUDIT.md``.
    """

    name = "replica_scaling"

    def __init__(self, *, config: Optional[ReplicaScalingConfig] = None) -> None:
        self._default_config = config or ReplicaScalingConfig()

    def optimize(
        self,
        raw: list[tuple[float, int]],
        *,
        warp: Optional[float] = None,
        config: Optional[ReplicaScalingConfig] = None,
        **kwargs,
    ) -> ReplicaScalingResult:
        """Compute the per-tick replica count schedule.

        Args:
            raw:    ``(arrival_s_unwarped, output_tokens)`` tuples from trace.
            warp:   Time-warp scalar; computed via ``_replica_calibrate_warp``
                    from ``config`` when ``None``.
            config: Policy config; uses constructor default when ``None``.

        Returns:
            :class:`ReplicaScalingResult` with ``c_schedule`` and diagnostics.
        """
        cfg = config if config is not None else self._default_config

        w = (
            warp
            if warp is not None
            else _replica_calibrate_warp(raw, cfg.servers, cfg.target_rho)
        )

        if cfg.mode == "amcsg":
            c_sched = compute_mcs_c_schedule(
                raw, cfg.tick_seconds, w,
                mcs_gate=cfg.safe_gate_pct,
                sla_s=cfg.sla_s,
            )
            c_mean = statistics.mean(c_sched) if c_sched else 0.0
            return ReplicaScalingResult(
                mode="amcsg",
                c_schedule=c_sched,
                c_mean=c_mean,
                n_ticks=len(c_sched),
                warp=w,
                oracle_iters=0,
                n_ticks_cheaper=0,
                baseline_n_sla_safe=0,
            )

        if cfg.mode == "sotss_min":
            c_sched, n_iters, init_viols, n_ticks_cheaper, baseline_n_sla_safe = (
                compute_sotss_min_schedule(
                    raw, cfg.tick_seconds, w,
                    sla_s=cfg.sla_s,
                    safe_gate=cfg.safe_gate_pct,
                    aggressive_gate=cfg.aggressive_gate_pct,
                    max_iters=cfg.max_oracle_iters,
                )
            )
            c_mean = statistics.mean(c_sched) if c_sched else 0.0
            return ReplicaScalingResult(
                mode="sotss_min",
                c_schedule=c_sched,
                c_mean=c_mean,
                n_ticks=len(c_sched),
                warp=w,
                oracle_iters=n_iters,
                n_ticks_cheaper=n_ticks_cheaper,
                baseline_n_sla_safe=baseline_n_sla_safe,
                initial_violations=init_viols,
            )

        if cfg.mode == "sotss_gsf":
            c_sched, n_iters, _, n_ticks_cheaper, baseline_n_sla_safe = (
                compute_sotss_gsf_schedule(
                    raw, cfg.tick_seconds, w,
                    sla_s=cfg.sla_s,
                    safe_gate=cfg.safe_gate_pct,
                    aggressive_gate=cfg.aggressive_gate_pct,
                    spot_fraction=cfg.spot_fraction,
                    zfhc_threshold=cfg.zfhc_threshold,
                    p_interrupt_hourly=cfg.p_interrupt_hourly,
                    seed=cfg.seed,
                    max_iters=cfg.max_oracle_iters,
                )
            )
            c_mean = statistics.mean(c_sched) if c_sched else 0.0
            return ReplicaScalingResult(
                mode="sotss_gsf",
                c_schedule=c_sched,
                c_mean=c_mean,
                n_ticks=len(c_sched),
                warp=w,
                oracle_iters=n_iters,
                n_ticks_cheaper=n_ticks_cheaper,
                baseline_n_sla_safe=baseline_n_sla_safe,
            )

        if cfg.mode == "online_sotss":
            c_sched, n_iters, init_viols, n_ticks_cheaper, baseline_n_sla_safe = (
                compute_online_sotss_schedule(
                    raw, cfg.tick_seconds, w,
                    sla_s=cfg.sla_s,
                    safe_gate=cfg.safe_gate_pct,
                    aggressive_gate=cfg.aggressive_gate_pct,
                    max_iters=cfg.max_oracle_iters,
                    baseline_n_sla_safe=cfg.baseline_n_sla_safe,
                    ewma_alpha=cfg.ewma_alpha,
                    ewma_mode=cfg.ewma_mode,
                    burst_threshold=cfg.burst_threshold,
                    burst_alpha=cfg.burst_alpha,
                    burst_cooldown_ticks=cfg.burst_cooldown_ticks,
                    interrupt_safety_margin=cfg.interrupt_safety_margin,
                    borderline_margin_s=cfg.borderline_margin_s,
                )
            )
            c_mean = statistics.mean(c_sched) if c_sched else 0.0
            return ReplicaScalingResult(
                mode="online_sotss",
                c_schedule=c_sched,
                c_mean=c_mean,
                n_ticks=len(c_sched),
                warp=w,
                oracle_iters=n_iters,
                n_ticks_cheaper=n_ticks_cheaper,
                baseline_n_sla_safe=baseline_n_sla_safe,
                initial_violations=init_viols,
            )

        if cfg.mode == "forecasted_mcs":
            # Fully deployable: forecasts BOTH next-tick arrivals AND service time
            # from data <= t-1. Lazy import breaks the optimizer<->benchmark cycle
            # (forecasted_mcs reuses the benchmark's Erlang-C/service physics).
            if cfg.forecast_method not in ("ewma", "quantile", "lag1"):
                raise ValueError(
                    f"ReplicaScalingPolicy: unknown forecast_method "
                    f"{cfg.forecast_method!r}. Deployable methods: "
                    "'ewma', 'quantile', 'lag1'. (The oracle is not deployable.)"
                )
            from aurelius.benchmarks.forecasted_mcs import (
                forecast_mcs_c_schedule,
                reactive_lag1_c_schedule,
            )

            if cfg.forecast_method == "lag1":
                c_sched = reactive_lag1_c_schedule(
                    raw, cfg.tick_seconds, w,
                    mcs_gate=cfg.safe_gate_pct, sla_s=cfg.sla_s,
                    warmup_c=cfg.forecast_warmup_c,
                )
            else:
                c_sched, _diag = forecast_mcs_c_schedule(
                    raw, cfg.tick_seconds, w,
                    method=cfg.forecast_method,
                    mcs_gate=cfg.safe_gate_pct, sla_s=cfg.sla_s,
                    ewma_alpha=cfg.forecast_ewma_alpha,
                    count_window=cfg.forecast_count_window,
                    quantile=cfg.forecast_quantile,
                    safety_k=cfg.forecast_safety_k,
                    warmup_c=cfg.forecast_warmup_c,
                )
            c_mean = statistics.mean(c_sched) if c_sched else 0.0
            return ReplicaScalingResult(
                mode="forecasted_mcs",
                c_schedule=c_sched,
                c_mean=c_mean,
                n_ticks=len(c_sched),
                warp=w,
                oracle_iters=0,
                n_ticks_cheaper=0,
                baseline_n_sla_safe=0,
            )

        raise ValueError(
            f"ReplicaScalingPolicy: unknown mode {cfg.mode!r}. Valid modes: "
            "'amcsg', 'sotss_min', 'sotss_gsf', 'online_sotss', 'forecasted_mcs', "
            "'constraint_aware', 'safe_high_utilization'."
        )

    def optimize_from_ticks(
        self,
        ticks,
        *,
        tick_hours: float,
        config: Optional[ReplicaScalingConfig] = None,
    ) -> ReplicaScalingResult:
        """Compute per-tick replica count for CA or SHU backtest serving modes.

        Phase 3e entry point: accepts ArrivalTick-like objects (duck-typed)
        rather than raw ``(arrival_s, output_tokens)`` pairs.  Supports
        ``mode="constraint_aware"`` and ``mode="safe_high_utilization"``.

        Args:
            ticks:      Sequence of tick aggregates exposing arrival_rate_rps,
                        output_tokens_mean, prompt_tokens_mean, request_count,
                        reuse_fraction, and optionally model_mix.
            tick_hours: Tick duration in hours.
            config:     ReplicaScalingConfig with mode set to
                        ``"constraint_aware"`` or ``"safe_high_utilization"``.
                        Uses constructor default when None.

        Returns:
            ReplicaScalingResult with c_schedule set to per-tick replica counts.
        """
        cfg = config if config is not None else self._default_config

        if cfg.mode == "constraint_aware":
            if cfg.adaptive_frontier_window is not None:
                ticks_list = list(ticks)
                rho_sched = compute_frontier_rho_schedule(
                    ticks_list,
                    tick_hours,
                    window=cfg.adaptive_frontier_window,
                    default_rho=cfg.ca_target_rho,
                )
                c_sched = compute_constraint_aware_schedule(
                    ticks_list, tick_hours,
                    ca_target_rho=cfg.ca_target_rho,
                    rho_schedule=rho_sched,
                )
            else:
                c_sched = compute_constraint_aware_schedule(
                    ticks, tick_hours, ca_target_rho=cfg.ca_target_rho
                )
        elif cfg.mode == "safe_high_utilization":
            c_sched = compute_shu_schedule(ticks, tick_hours)
        else:
            raise ValueError(
                f"ReplicaScalingPolicy.optimize_from_ticks: unsupported mode "
                f"{cfg.mode!r}. Supported: 'constraint_aware', 'safe_high_utilization'."
            )

        c_mean = statistics.mean(c_sched) if c_sched else 0.0
        return ReplicaScalingResult(
            mode=cfg.mode,
            c_schedule=c_sched,
            c_mean=c_mean,
            n_ticks=len(c_sched),
            warp=1.0,
            oracle_iters=0,
            n_ticks_cheaper=0,
            baseline_n_sla_safe=0,
        )


# ---------------------------------------------------------------------------
# Phase 3e: BurstGPT/Azure backtest serving physics — CA and SHU modes
# Extracted verbatim from aurelius/traces/backtest.py to enable canonical
# routing through AureliusOptimizer(policy="replica_scaling").
# Constants and formulas are byte-identical to backtest.py originals.
# ---------------------------------------------------------------------------

# Documented benchmark priors — identical to backtest.py originals.
_BT_BASE_TTFT_MS: float = 150.0
_BT_BASE_TPOT_MS: float = 20.0
_BT_TTFT_SLO_MS: float = 2000.0
_BT_TPOT_SLO_MS: float = 50.0
_BT_MIN_REPLICAS: int = 1
_BT_MAX_PREFILL_SAVINGS: float = 0.25
_BT_SHU_TARGET_RHO: float = 0.75
_BT_SHU_TIMEOUT_TOL: float = 0.0
_BT_EWMA_ALPHA: float = 0.5
_BT_FALLBACK_TOKENS_PER_S: float = 2500.0
_BT_MODEL_TOKENS_PER_S: dict = {"ChatGPT": 3400.0, "GPT-4": 1700.0}


def _bt_tick_throughput_tokps(model_mix, request_count: int) -> float:
    """Request-fraction-weighted per-replica token throughput — identical to backtest.py."""
    if not model_mix or request_count == 0:
        return _BT_FALLBACK_TOKENS_PER_S
    total = sum(model_mix.values())
    return sum(
        (cnt / total) * _BT_MODEL_TOKENS_PER_S.get(m, _BT_FALLBACK_TOKENS_PER_S)
        for m, cnt in model_mix.items()
    )


def _bt_size_for_target(
    arrival_rate: float,
    output_mean: float,
    throughput: float,
    target_rho: float,
) -> int:
    """Replicas needed to keep utilization at/below target_rho — identical to backtest.py."""
    mu_full = throughput / max(1.0, output_mean)
    if mu_full <= 0 or arrival_rate <= 0:
        return _BT_MIN_REPLICAS
    return max(_BT_MIN_REPLICAS, int(math.ceil(arrival_rate / (mu_full * target_rho))))


def _bt_timeout_rate_pct(
    arrival_rate_rps: float,
    output_tokens_mean: float,
    prompt_tokens_mean: float,
    throughput_tokps: float,
    replicas: int,
    prefill_savings: float,
) -> float:
    """Compute timeout_rate_pct for backtest serving physics (lazy serving import).

    Replicates the timeout_rate_pct branch of evaluate_tick() in backtest.py
    byte-identically.  Only used by _bt_constraint_trim; omits KPI fields that
    do not affect the trim decision (gpu_hours, energy_cost, queue percentiles).
    """
    from aurelius.simulation.cluster import serving  # lazy — avoids heavy import

    replicas = max(_BT_MIN_REPLICAS, int(replicas))
    output_mean = max(1.0, output_tokens_mean)
    prompt_mean = max(0.0, prompt_tokens_mean)
    arrival_rate = arrival_rate_rps
    throughput = throughput_tokps

    base_service_s = (_BT_BASE_TTFT_MS + _BT_BASE_TPOT_MS * output_mean) / 1000.0
    active_seqs = max(0.0, arrival_rate * base_service_s)

    batch_eff = serving.batching_efficiency(active_seqs, replicas)
    mu_per = max(1e-9, (throughput / max(1.0, output_mean)) * batch_eff)
    rho = arrival_rate / (replicas * mu_per) if replicas > 0 else 1.0

    mean_wait_s = serving.erlang_c_wait_s(arrival_rate, mu_per, replicas)
    if not math.isfinite(mean_wait_s):
        mean_wait_s = 60.0
    mean_wait_s = min(60.0, mean_wait_s * serving.saturation_amplifier(rho))
    mean_wait_ms = mean_wait_s * 1000.0

    _p95_mult, p99_mult = serving.tail_multipliers(rho)

    active_per_replica = active_seqs / replicas  # replicas >= 1 guaranteed above
    eff_prompt = prompt_mean * (1.0 - prefill_savings)
    ttft_compute = serving.ttft_ms(0.0, eff_prompt, active_per_replica, 0.0, 1.0)
    ttft_p50 = mean_wait_ms + ttft_compute
    ttft_p99 = ttft_p50 * p99_mult

    tpot_p50 = serving.tpot_ms(_BT_BASE_TPOT_MS, active_per_replica, 1.0)
    tpot_p99 = tpot_p50 * 4.0

    latency_p99 = ttft_p99 + tpot_p99 * output_mean
    sla_ms = _BT_TTFT_SLO_MS + output_mean * _BT_TPOT_SLO_MS
    if latency_p99 > sla_ms:
        return min(50.0, (latency_p99 - sla_ms) / sla_ms * 10.0)
    return 0.0


def _bt_constraint_trim(
    arrival_rate_rps: float,
    output_tokens_mean: float,
    prompt_tokens_mean: float,
    throughput_tokps: float,
    base: int,
    prefill_savings: float,
    prev_replicas: Optional[int],
    timeout_tol: float = 0.0,
) -> int:
    """Trim replicas below base while timeout_rate <= timeout_tol% — identical to backtest.py.

    tick_hours is omitted because timeout_rate_pct does not depend on it.
    """
    chosen = base
    for r in range(base, _BT_MIN_REPLICAS - 1, -1):
        trate = _bt_timeout_rate_pct(
            arrival_rate_rps, output_tokens_mean, prompt_tokens_mean,
            throughput_tokps, r, prefill_savings,
        )
        if trate <= timeout_tol:
            chosen = r
        else:
            break
    if prev_replicas is not None and abs(chosen - prev_replicas) == 1:
        trate_prev = _bt_timeout_rate_pct(
            arrival_rate_rps, output_tokens_mean, prompt_tokens_mean,
            throughput_tokps, prev_replicas, prefill_savings,
        )
        if trate_prev <= timeout_tol:
            chosen = prev_replicas
    return max(_BT_MIN_REPLICAS, chosen)


def compute_frontier_rho_schedule(
    ticks,
    tick_hours: float,
    *,
    window: int = 10,
    default_rho: float = 0.65,
    max_prefill_savings: float = _BT_MAX_PREFILL_SAVINGS,
) -> list:
    """Causal per-tick rho schedule via rolling-window frontier safety estimation.

    For tick k < window, uses default_rho (cold-start — no history yet).
    For tick k >= window, estimates the frontier over past ticks [k-window, k)
    and selects the highest SAFE rho; falls back to default_rho when no SAFE
    point exists (INSUFFICIENT_TELEMETRY or all UNSAFE).

    Strictly causal: only past-tick telemetry feeds the estimator; no future
    information is used. Five-Failure-Rule compliant — integrates the existing
    frontier estimator module without adding a new optimizer path.

    Args:
        ticks:              Sequence of ArrivalTick-like objects (duck-typed).
        tick_hours:         Tick duration in hours (used by the anticipatory sizer).
        window:             Rolling window size in ticks (default 10).
        default_rho:        Rho used during cold-start and as safety fallback
                            (default 0.65 — same as the fixed constraint_aware rho).
        max_prefill_savings: Cap on prefill cache savings (default 0.25 — matches
                            constraint_aware). The mean per-window value of
                            max_prefill_savings * t.reuse_fraction is passed to
                            FrontierEstimatorConfig so the SLA evaluation uses actual
                            KV-cache reuse telemetry rather than the estimator default
                            of 0.0.

    Returns:
        list[float] — per-tick rho target, one per tick.
    """
    # Lazy imports — avoids module-load-time circular import via backtest.py.
    from aurelius.frontier.estimator import FrontierEstimatorConfig, estimate_frontier
    from aurelius.frontier.models import SafetyStatus, WorkloadFrontierProfile
    from aurelius.frontier.safety import SafetyConfig

    ticks_list = list(ticks)
    n = len(ticks_list)

    profile = WorkloadFrontierProfile(
        workload_id="constraint_aware_adaptive",
        workload_type="llm_serving",
        telemetry_confidence="medium",
        min_rho=0.45,
        max_rho=0.95,
    )
    tick_seconds = tick_hours * 3600.0
    safety = SafetyConfig()  # max_timeout_pct=10.0, max_queue_p99_ms=2000.0

    rho_schedule: list = []
    for k in range(n):
        if k < window:
            rho_schedule.append(default_rho)
        else:
            tel_window = ticks_list[k - window : k]
            if not tel_window:
                rho_schedule.append(default_rho)
                continue
            # Use actual mean prefill savings from the causal window so the
            # estimator's SLA evaluation uses real KV-cache reuse telemetry.
            mean_prefill = sum(
                max_prefill_savings * getattr(t, "reuse_fraction", 0.0)
                for t in tel_window
            ) / len(tel_window)
            est_cfg = FrontierEstimatorConfig(
                mode="anticipatory",
                tick_seconds=tick_seconds,
                prefill_savings=mean_prefill,
            )
            points = estimate_frontier(
                profile,
                tel_window,
                predictor_config=est_cfg,
                safety_config=safety,
            )
            safe_points = [p for p in points if p.safety_status == SafetyStatus.SAFE]
            best_rho = max(p.rho_target for p in safe_points) if safe_points else default_rho
            rho_schedule.append(best_rho)

    return rho_schedule


def compute_constraint_aware_schedule(
    ticks,
    tick_hours: float,
    *,
    ca_target_rho: float = 0.65,
    ewma_alpha: float = _BT_EWMA_ALPHA,
    max_prefill_savings: float = _BT_MAX_PREFILL_SAVINGS,
    rho_schedule: Optional[list] = None,
) -> list:
    """Per-tick replica counts for the constraint_aware policy.

    Extracted verbatim from the constraint_aware branch of
    _run_policy() in aurelius/traces/backtest.py.  Algorithm:
      EWMA-anticipatory sizing (max of current + smoothed peak),
      size to target_rho, exploit cache prefill savings,
      damp churn with hysteresis (prev_replicas).

    Args:
        ticks:              Sequence of ArrivalTick-like objects (duck-typed).
                            Must expose arrival_rate_rps, output_tokens_mean,
                            prompt_tokens_mean, request_count, reuse_fraction,
                            and optionally model_mix (dict).
        tick_hours:         Tick duration in hours (passed through but not
                            used by timeout physics; kept for interface parity).
        ca_target_rho:      Target per-server utilisation (default 0.65).
                            Used as the rho for every tick unless rho_schedule
                            overrides per-tick rho.
        ewma_alpha:         EWMA smoothing factor (default 0.5).
        max_prefill_savings: Cache prefill savings cap (default 0.25).
        rho_schedule:       Optional per-tick rho override list (one float per
                            tick).  When provided, rho_schedule[i] replaces
                            ca_target_rho for tick i.  ``None`` (default)
                            preserves byte-identical behavior with the fixed
                            ca_target_rho path.

    Returns:
        list[int] — per-tick replica counts, one per tick.
    """
    c_schedule: list = []
    ewma_rate: float = 0.0
    ewma_out: float = 0.0
    prev_replicas: Optional[int] = None

    for i, t in enumerate(ticks):
        if t.request_count > 0:
            ewma_rate = (
                ewma_alpha * t.arrival_rate_rps + (1.0 - ewma_alpha) * ewma_rate
                if ewma_rate else t.arrival_rate_rps
            )
            ewma_out = (
                ewma_alpha * t.output_tokens_mean + (1.0 - ewma_alpha) * ewma_out
                if ewma_out else t.output_tokens_mean
            )

        target_rho = rho_schedule[i] if rho_schedule is not None else ca_target_rho

        prefill_savings = max_prefill_savings * t.reuse_fraction
        throughput = _bt_tick_throughput_tokps(
            getattr(t, "model_mix", None), t.request_count
        )
        plan_rate = max(t.arrival_rate_rps, ewma_rate)
        plan_out = max(t.output_tokens_mean, ewma_out) if t.request_count else ewma_out
        base = _bt_size_for_target(plan_rate, max(1.0, plan_out), throughput, target_rho)
        replicas = _bt_constraint_trim(
            t.arrival_rate_rps,
            t.output_tokens_mean,
            getattr(t, "prompt_tokens_mean", 0.0),
            throughput,
            base,
            prefill_savings,
            prev_replicas,
            timeout_tol=0.0,
        )
        c_schedule.append(replicas)
        prev_replicas = replicas

    return c_schedule


def compute_shu_schedule(
    ticks,
    tick_hours: float,
    *,
    ewma_alpha: float = _BT_EWMA_ALPHA,
    max_prefill_savings: float = _BT_MAX_PREFILL_SAVINGS,
) -> list:
    """Per-tick replica counts for the safe_high_utilization policy.

    Extracted verbatim from the safe_high_utilization branch of
    _run_policy() in aurelius/traces/backtest.py.  Algorithm:
      EWMA-anticipatory sizing at rho=0.75 (higher utilisation than CA),
      exploit cache prefill savings, no hysteresis (prev_replicas=None).

    Args:
        ticks:              Sequence of ArrivalTick-like objects (duck-typed).
        tick_hours:         Tick duration in hours (interface parity).
        ewma_alpha:         EWMA smoothing factor (default 0.5).
        max_prefill_savings: Cache prefill savings cap (default 0.25).

    Returns:
        list[int] — per-tick replica counts, one per tick.
    """
    c_schedule: list = []
    ewma_rate: float = 0.0
    ewma_out: float = 0.0

    for t in ticks:
        if t.request_count > 0:
            ewma_rate = (
                ewma_alpha * t.arrival_rate_rps + (1.0 - ewma_alpha) * ewma_rate
                if ewma_rate else t.arrival_rate_rps
            )
            ewma_out = (
                ewma_alpha * t.output_tokens_mean + (1.0 - ewma_alpha) * ewma_out
                if ewma_out else t.output_tokens_mean
            )

        prefill_savings = max_prefill_savings * t.reuse_fraction
        throughput = _bt_tick_throughput_tokps(
            getattr(t, "model_mix", None), t.request_count
        )
        plan_rate = max(t.arrival_rate_rps, ewma_rate)
        plan_out = max(t.output_tokens_mean, ewma_out) if t.request_count else ewma_out
        base = _bt_size_for_target(
            plan_rate, max(1.0, plan_out), throughput, _BT_SHU_TARGET_RHO
        )
        replicas = _bt_constraint_trim(
            t.arrival_rate_rps,
            t.output_tokens_mean,
            getattr(t, "prompt_tokens_mean", 0.0),
            throughput,
            base,
            prefill_savings,
            None,  # no hysteresis for SHU
            timeout_tol=_BT_SHU_TIMEOUT_TOL,
        )
        c_schedule.append(replicas)

    return c_schedule


__all__ = [
    "REPLICA_TTFT_BASE_S",
    "REPLICA_TPOT_S",
    "REPLICA_SAFE_GATE",
    "REPLICA_AGGRESSIVE_GATE",
    "REPLICA_MAX_ORACLE_ITERS",
    "REPLICA_OSOTSS_EWMA_ALPHA",
    "DEPLOYABLE_MODES",
    "RESEARCH_ONLY_MODES",
    "is_deployable_mode",
    "_replica_service_time_s",
    "_replica_calibrate_warp",
    "_replica_erlang_c_sla_timeout_pct",
    "compute_mcs_c_schedule",
    "_oracle_fifo_response_times",
    "_oracle_stochastic_response_times",
    "compute_sotss_min_schedule",
    "compute_sotss_gsf_schedule",
    "compute_online_sotss_schedule",
    "compute_c1pgs_spot_replicas",
    "compute_frontier_rho_schedule",
    "compute_constraint_aware_schedule",
    "compute_shu_schedule",
    "_bt_timeout_rate_pct",
    "_bt_constraint_trim",
    "_bt_size_for_target",
    "_BT_SHU_TARGET_RHO",
    "_BT_SHU_TIMEOUT_TOL",
    "_BT_EWMA_ALPHA",
    "_BT_MAX_PREFILL_SAVINGS",
    "ReplicaScalingConfig",
    "ReplicaScalingResult",
    "ReplicaScalingPolicy",
]
