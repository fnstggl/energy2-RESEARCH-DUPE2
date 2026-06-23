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
from dataclasses import dataclass, field
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
    """Provisioning algorithm: ``"amcsg"`` (Erlang-C gate sweep) or
    ``"sotss_min"`` (SOTSS oracle from minimum stable c)."""

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


@dataclass
class ReplicaScalingResult:
    """Result returned by :class:`ReplicaScalingPolicy`."""

    mode: str
    """Algorithm used: ``"amcsg"`` or ``"sotss_min"``."""

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


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------

class ReplicaScalingPolicy(OptimizationPolicy):
    """Canonical replica-scaling provisioning policy — Phase 2/3.

    Governs the per-tick server count (c_schedule) for serving queues, replacing
    the equivalent logic previously owned by the benchmark monolith.

    Modes:
        ``"amcsg"``     — per-tick Erlang-C MCS gate sweep (deterministic,
                          matches the AMCSG baseline at ``safe_gate_pct=12.5``).
        ``"sotss_min"`` — SOTSS oracle loop starting from minimum stable c
                          (``aggressive_gate_pct=100.0``), the frontier algorithm
                          that achieves +6.29% goodput/$ vs AMCSG on Azure.

    Extraction pattern follows Phase 2 (serving_queue.py): identical algorithm,
    constants and tie-breaks; NOT a research change.
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
            c_sched, n_iters, _, n_ticks_cheaper, baseline_n_sla_safe = (
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
            )

        raise ValueError(
            f"ReplicaScalingPolicy: unknown mode {cfg.mode!r}. "
            "Valid modes: 'amcsg', 'sotss_min'."
        )


__all__ = [
    "REPLICA_TTFT_BASE_S",
    "REPLICA_TPOT_S",
    "REPLICA_SAFE_GATE",
    "REPLICA_AGGRESSIVE_GATE",
    "REPLICA_MAX_ORACLE_ITERS",
    "_replica_service_time_s",
    "_replica_calibrate_warp",
    "_replica_erlang_c_sla_timeout_pct",
    "compute_mcs_c_schedule",
    "_oracle_fifo_response_times",
    "compute_sotss_min_schedule",
    "compute_c1pgs_spot_replicas",
    "ReplicaScalingConfig",
    "ReplicaScalingResult",
    "ReplicaScalingPolicy",
]
