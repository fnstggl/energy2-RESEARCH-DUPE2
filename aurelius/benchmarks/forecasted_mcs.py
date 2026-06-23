"""Deployable forecasted Min-Cost-Safe (MCS) provisioning.

This module replaces the **oracle** MCS capacity scheduler
(``_joint_mcs_c_schedule`` in :mod:`aurelius.benchmarks.srtf_serving_backtest`)
with a strictly **causal** forecaster.

Motivation (see ``research/MCS_AUDIT.md``)
------------------------------------------
The existing MCS family (joint-MCS, spot-fleet MCS, AMCSG, GSF, ZFHC,
abs-floor, SOTSS, DLAG) all derive the per-tick replica count ``c[t]`` from
``_joint_mcs_c_schedule``, which buckets the **actual** requests that arrive in
tick ``t`` and sizes ``c[t]`` from that tick's **actual** arrival count and
**actual** output-token counts.  That is a clairvoyant capacity planner — it
peeks at the demand it is supposed to provision *for*, and at output-token
counts that are not even known until a request completes.  It is an upper
bound, not a deployable policy.

This module keeps **everything else identical** — the same Erlang-C gate
physics (``_erlang_c_sla_timeout_pct``), the same service-time model
(``_service_time_s``), the same discrete-event FIFO simulator
(``_simulate_fifo_variable_c``), the same provisioned-GPU-hour cost denominator
and the same SLA definition — and changes only the *information set* used to
choose ``c[t]``:

    At the decision boundary for tick ``t`` the forecaster may use the realised
    arrivals and output tokens of ticks ``0 .. t-1`` only.  It never sees tick
    ``t``'s arrivals or tokens.

Forecast targets:
  * next-tick arrival count   (EWMA or rolling-quantile of past per-tick counts)
  * next-tick mean service    (EWMA of past per-tick mean service, derived from a
                               causal running-median output-token estimate)
  * optional one-sided safety buffer (``safety_k``·std or a high rolling quantile)
    to trade a little cost for SLA headroom under forecast error.

The first ``warmup_ticks`` ticks have no history, so they fall back to a fixed
cold-start ``warmup_c`` — a deployable design-time guess, not trace knowledge.

Directional simulator evidence only — NOT production savings
(``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_SLA_S,
    GPU_HOUR_USD,
    AbsoluteErrorConformalCalibrator,
    _erlang_c_sla_timeout_pct,
    _Request,
    _service_time_s,
    _simulate_abs_conformal_variable_c,
    _simulate_fifo_variable_c,
    _sla_safe_goodput,
    simulate_queue,
)

# Disciplines that operate at a *fixed* server count (run via simulate_queue).
# For these the c_schedule must be constant; provisioned cost is still computed
# the same way (sum(c)*tick_hr) so the denominator stays identical.
_FIXED_C_DISCIPLINES = frozenset({"fifo_fixed", "sla_aware", "srtf", "decoupled_hybrid_abs_conformal"})

# Erlang-C search ceiling — identical to _joint_mcs_c_schedule's range(1, 1024).
_MAX_C: int = 1024


# ---------------------------------------------------------------------------
# Trace -> per-tick realised series (ground truth; fed causally to forecaster)
# ---------------------------------------------------------------------------

def bucketize(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
) -> tuple[list[int], list[list[int]], int]:
    """Bucket warped arrivals into per-tick (count, token-list).

    Uses exactly the same warping and bucket-index arithmetic as
    ``_joint_mcs_c_schedule`` so the tick grid is identical to the oracle's.

    Returns ``(counts, token_lists, n_ticks)`` where ``counts[t]`` is the number
    of requests that arrived in tick ``t`` and ``token_lists[t]`` is the list of
    their output-token counts.  These are the *realised* values; the forecaster
    is only allowed to read indices ``< t`` when sizing tick ``t``.
    """
    if not raw:
        return [], [], 0
    warped = [(t / warp, tok) for t, tok in raw]
    t_max = warped[-1][0]
    n_ticks = max(1, int(t_max / tick_seconds) + 1)
    counts = [0] * n_ticks
    token_lists: list[list[int]] = [[] for _ in range(n_ticks)]
    for t, tok in warped:
        idx = min(n_ticks - 1, int(t / tick_seconds))
        counts[idx] += 1
        token_lists[idx].append(tok)
    return counts, token_lists, n_ticks


def _min_safe_c(
    lam: float,
    mean_service: float,
    sla_wait: float,
    mcs_gate: float,
) -> int:
    """Smallest c with Erlang-C SLA-timeout < gate.  Identical to the oracle's
    inner loop, just driven by *forecasted* lam/mean_service instead of actuals.
    """
    for c in range(1, _MAX_C):
        if _erlang_c_sla_timeout_pct(lam, mean_service, c, sla_wait) < mcs_gate:
            return c
    return _MAX_C - 1


# ---------------------------------------------------------------------------
# Deployable forecasted MCS
# ---------------------------------------------------------------------------

@dataclass
class ForecastDiag:
    """Forecast-quality diagnostics (causal, measured against realised ticks)."""

    method: str
    n_ticks: int
    warmup_ticks: int
    # arrival-count forecast error over non-warmup, non-empty-history ticks
    arr_mae: float = 0.0
    arr_rel_mae_pct: float = 0.0
    arr_bias: float = 0.0          # mean(forecast - actual); >0 = over-provision bias
    # service-time forecast error (seconds)
    svc_mae_s: float = 0.0
    # capacity outcome
    c_mean: float = 0.0
    c_min: int = 0
    c_max: int = 0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_ticks": self.n_ticks,
            "warmup_ticks": self.warmup_ticks,
            "arr_mae": round(self.arr_mae, 4),
            "arr_rel_mae_pct": round(self.arr_rel_mae_pct, 2),
            "arr_bias": round(self.arr_bias, 4),
            "svc_mae_s": round(self.svc_mae_s, 4),
            "c_mean": round(self.c_mean, 4),
            "c_min": self.c_min,
            "c_max": self.c_max,
        }


def forecast_mcs_c_schedule(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    *,
    method: str = "ewma",
    mcs_gate: float = 9.5,
    sla_s: float = DEFAULT_SLA_S,
    ewma_alpha: float = 0.5,
    count_window: int = 8,
    token_window: int = 200,
    quantile: float = 0.90,
    safety_k: float = 0.0,
    warmup_c: int = 4,
    warmup_ticks: int = 1,
) -> tuple[list[int], ForecastDiag]:
    """Strictly-causal per-tick MCS replica schedule.

    For each tick ``t`` (processed in order):

    1. If ``t < warmup_ticks`` (no usable history): ``c[t] = warmup_c``.
    2. Otherwise forecast tick ``t`` from ticks ``0 .. t-1`` only:
         * ``arr_hat`` — EWMA (``method="ewma"``) or rolling ``quantile``
           (``method="quantile"``) of past per-tick arrival counts;
         * optional one-sided buffer ``+ safety_k * std(recent counts)``;
         * ``svc_hat`` — EWMA of past per-tick mean service, where each past
           tick's mean service is itself derived from realised tokens (this is
           information available after that past tick completed).
       Then ``c[t] = min c : ErlangC_timeout(arr_hat/tick, svc_hat, c) < gate``.
    3. *After* committing ``c[t]``, the realised tick-``t`` arrivals/tokens are
       revealed and folded into the running history for tick ``t+1``.

    No tick-``t`` actual ever influences ``c[t]``.

    Args:
        method: ``"ewma"`` (default) or ``"quantile"`` arrival forecaster.
        ewma_alpha: EWMA smoothing for arrival count and mean service.
        count_window: window for the rolling quantile / recent-std buffer.
        token_window: window for the causal running-median token estimate.
        quantile: rolling quantile (e.g. 0.90) for ``method="quantile"``.
        safety_k: one-sided arrival buffer in units of recent count std.
        warmup_c: fixed cold-start capacity before history exists.
        warmup_ticks: number of leading ticks served at ``warmup_c``.

    Returns ``(c_schedule, ForecastDiag)``.
    """
    counts, token_lists, n_ticks = bucketize(raw, tick_seconds, warp)
    if n_ticks == 0:
        return [], ForecastDiag(method=method, n_ticks=0, warmup_ticks=warmup_ticks)

    c_sched: list[int] = []

    # Running EWMA state.
    ewma_count: Optional[float] = None
    ewma_svc: Optional[float] = None
    # Rolling history.
    hist_counts: list[int] = []
    token_hist: list[int] = []   # output tokens in arrival order (causal)

    # Forecast-error accumulators (only over forecasted, history-present ticks).
    arr_abs_err: list[float] = []
    arr_signed_err: list[float] = []
    svc_abs_err: list[float] = []

    for t in range(n_ticks):
        if t < warmup_ticks or not hist_counts:
            c_sched.append(max(1, warmup_c))
        else:
            # --- arrival-count forecast (causal) ---
            if method == "quantile":
                win = hist_counts[-count_window:]
                arr_hat = _quantile(win, quantile)
            else:  # ewma
                arr_hat = ewma_count if ewma_count is not None else hist_counts[-1]

            if safety_k > 0.0 and len(hist_counts) >= 2:
                recent = hist_counts[-count_window:]
                sd = statistics.pstdev(recent) if len(recent) >= 2 else 0.0
                arr_hat = arr_hat + safety_k * sd

            arr_hat = max(0.0, arr_hat)

            # --- mean-service forecast (causal) ---
            svc_hat = ewma_svc if ewma_svc is not None else _service_time_s(
                _running_median(token_hist, token_window)
            )

            # Record forecast error vs the (now to-be-revealed) actual.
            actual_count = counts[t]
            arr_abs_err.append(abs(arr_hat - actual_count))
            arr_signed_err.append(arr_hat - actual_count)
            if token_lists[t]:
                actual_svc = statistics.mean(_service_time_s(tok) for tok in token_lists[t])
                svc_abs_err.append(abs(svc_hat - actual_svc))

            lam = arr_hat / tick_seconds
            sla_wait = max(0.0, sla_s - svc_hat)
            if arr_hat <= 0.0:
                c_sched.append(1)
            else:
                c_sched.append(_min_safe_c(lam, svc_hat, sla_wait, mcs_gate))

        # ---- reveal realised tick t and update running state ----
        obs_count = counts[t]
        hist_counts.append(obs_count)
        if token_lists[t]:
            token_hist.extend(token_lists[t])
            tick_mean_svc = statistics.mean(_service_time_s(tok) for tok in token_lists[t])
            ewma_svc = tick_mean_svc if ewma_svc is None else (
                ewma_alpha * tick_mean_svc + (1.0 - ewma_alpha) * ewma_svc
            )
        ewma_count = float(obs_count) if ewma_count is None else (
            ewma_alpha * obs_count + (1.0 - ewma_alpha) * ewma_count
        )

    diag = ForecastDiag(
        method=method,
        n_ticks=n_ticks,
        warmup_ticks=warmup_ticks,
        arr_mae=statistics.mean(arr_abs_err) if arr_abs_err else 0.0,
        arr_rel_mae_pct=(
            100.0 * statistics.mean(arr_abs_err) / max(1e-9, statistics.mean(counts))
            if arr_abs_err else 0.0
        ),
        arr_bias=statistics.mean(arr_signed_err) if arr_signed_err else 0.0,
        svc_mae_s=statistics.mean(svc_abs_err) if svc_abs_err else 0.0,
        c_mean=statistics.mean(c_sched),
        c_min=min(c_sched),
        c_max=max(c_sched),
    )
    return c_sched, diag


def reactive_lag1_c_schedule(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    *,
    mcs_gate: float = 9.5,
    sla_s: float = DEFAULT_SLA_S,
    warmup_c: int = 4,
) -> list[int]:
    """Naive deployable forecaster: ``c[t]`` sized from tick ``t-1``'s actuals.

    This is the simplest possible causal forecast ("next tick looks like the
    last tick").  It is the deployable analogue closest to the oracle (the
    oracle uses tick ``t``; this uses tick ``t-1``).  Empty previous tick -> 1.
    """
    counts, token_lists, n_ticks = bucketize(raw, tick_seconds, warp)
    if n_ticks == 0:
        return []
    c_sched: list[int] = []
    for t in range(n_ticks):
        if t == 0:
            c_sched.append(max(1, warmup_c))
            continue
        prev_tokens = token_lists[t - 1]
        if not prev_tokens:
            c_sched.append(1)
            continue
        lam = len(prev_tokens) / tick_seconds
        mean_service = statistics.mean(_service_time_s(tok) for tok in prev_tokens)
        sla_wait = max(0.0, sla_s - mean_service)
        c_sched.append(_min_safe_c(lam, mean_service, sla_wait, mcs_gate))
    return c_sched


def sla_aware_fixed_c(
    raw: list[tuple[float, int]],
    tick_seconds: float,
    warp: float,
    *,
    mcs_gate: float = 9.5,
    sla_s: float = DEFAULT_SLA_S,
) -> int:
    """A single fixed capacity sized for the SLA from *aggregate* trace stats.

    Deployable design-time sizing: uses the global mean arrival rate and global
    mean service (the kind of capacity-plan number an operator computes once
    from historical aggregates), then picks the smallest c meeting the gate.
    Constant across all ticks (the "provision once for SLA" baseline).
    """
    counts, token_lists, n_ticks = bucketize(raw, tick_seconds, warp)
    if n_ticks == 0:
        return 1
    total_req = sum(counts)
    lam = total_req / (n_ticks * tick_seconds)
    all_tokens = [tok for toks in token_lists for tok in toks]
    mean_service = statistics.mean(_service_time_s(tok) for tok in all_tokens) if all_tokens else _service_time_s(1)
    sla_wait = max(0.0, sla_s - mean_service)
    return _min_safe_c(lam, mean_service, sla_wait, mcs_gate)


# ---------------------------------------------------------------------------
# Evaluation — one physics model, one cost denominator, one SLA, one trace
# ---------------------------------------------------------------------------

@dataclass
class PolicyKPI:
    """KPIs for one capacity policy under the shared simulator."""

    policy: str
    discipline: str            # "fifo" or "abs_conformal"
    uses_future_info: bool
    deployable: bool
    classification: str

    n_ticks: int
    c_mean: float
    c_min: int
    c_max: int

    gpu_hours: float           # provisioned = sum(c)*tick_hr
    cost_usd: float            # gpu_hours * GPU_HOUR_USD
    sla_safe_goodput: float    # SLA-safe output tokens
    goodput_per_dollar: float

    n_total: int
    n_sla_safe: int
    sla_violations: int        # n_total - n_sla_safe (incl. dropped)
    completion_rate: float
    p50_wait_s: float
    p95_wait_s: float
    p99_wait_s: float
    p99_response_s: float

    forecast: Optional[dict] = None
    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "policy": self.policy,
            "discipline": self.discipline,
            "uses_future_info": self.uses_future_info,
            "deployable": self.deployable,
            "classification": self.classification,
            "n_ticks": self.n_ticks,
            "c_mean": round(self.c_mean, 4),
            "c_min": self.c_min,
            "c_max": self.c_max,
            "gpu_hours": round(self.gpu_hours, 4),
            "cost_usd": round(self.cost_usd, 4),
            "sla_safe_goodput": round(self.sla_safe_goodput, 2),
            "goodput_per_dollar": round(self.goodput_per_dollar, 2),
            "n_total": self.n_total,
            "n_sla_safe": self.n_sla_safe,
            "sla_violations": self.sla_violations,
            "completion_rate": round(self.completion_rate, 5),
            "p50_wait_s": round(self.p50_wait_s, 4),
            "p95_wait_s": round(self.p95_wait_s, 4),
            "p99_wait_s": round(self.p99_wait_s, 4),
            "p99_response_s": round(self.p99_response_s, 4),
            "runtime_s": round(self.runtime_s, 4),
        }
        if self.forecast is not None:
            d["forecast"] = self.forecast
        return d


def _wait_percentile(wait_map: dict, p: float) -> float:
    if not wait_map:
        return 0.0
    xs = sorted(wait_map.values())
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def evaluate_c_schedule(
    raw: list[tuple[float, int]],
    c_schedule: list[int],
    tick_seconds: float,
    warp: float,
    sla_s: float,
    *,
    policy: str,
    uses_future_info: bool,
    deployable: bool,
    classification: str,
    discipline: str = "fifo",
    predicted_tokens: Optional[list[float]] = None,
    forecast: Optional[dict] = None,
    runtime_s: float = 0.0,
) -> PolicyKPI:
    """Run a capacity schedule through the shared simulator and compute KPIs.

    All policies share: the provisioned-GPU-hour cost denominator
    ``sum(c)*tick_hr*GPU_HOUR_USD``, the SLA-safe goodput numerator, and the
    same discrete-event simulator.  ``discipline="abs_conformal"`` uses the
    strongest validated Aurelius serving discipline (Decoupled-Hybrid SRPT +
    absolute-error conformal alpha) with ``predicted_tokens`` for ordering.
    """
    if predicted_tokens is None:
        reqs = [
            _Request(idx=i, arrival_s=arr / warp, actual_tokens=tok,
                     predicted_tokens=float(tok), service_s=_service_time_s(tok))
            for i, (arr, tok) in enumerate(raw)
        ]
    else:
        reqs = [
            _Request(idx=i, arrival_s=arr / warp, actual_tokens=tok,
                     predicted_tokens=float(predicted_tokens[i]),
                     service_s=_service_time_s(tok))
            for i, (arr, tok) in enumerate(raw)
        ]

    if discipline == "abs_conformal":
        cal = AbsoluteErrorConformalCalibrator()
        summary, response, wait_map = _simulate_abs_conformal_variable_c(
            reqs, c_schedule, cal, tick_seconds
        )
    elif discipline in _FIXED_C_DISCIPLINES:
        # Fixed-server-count discipline via simulate_queue. The schedule must be
        # constant (a fixed-c baseline); provisioned cost = sum(c)*tick_hr as for
        # every other policy, so the cost denominator is identical.
        servers = c_schedule[0] if c_schedule else 1
        sim_disc = "fifo" if discipline == "fifo_fixed" else discipline
        summary, response, wait_map = simulate_queue(reqs, servers, sim_disc)
    else:
        summary, response, wait_map = _simulate_fifo_variable_c(
            reqs, c_schedule, tick_seconds
        )

    gpu_hours = sum(c_schedule) * tick_seconds / 3600.0
    cost = max(gpu_hours * GPU_HOUR_USD, 1e-9)
    goodput = _sla_safe_goodput(reqs, response, sla_s)
    n_sla_safe = sum(1 for r in reqs if r.idx in response and response[r.idx] <= sla_s)
    n_total = len(reqs)

    return PolicyKPI(
        policy=policy,
        discipline=discipline,
        uses_future_info=uses_future_info,
        deployable=deployable,
        classification=classification,
        n_ticks=len(c_schedule),
        c_mean=statistics.mean(c_schedule) if c_schedule else 0.0,
        c_min=min(c_schedule) if c_schedule else 0,
        c_max=max(c_schedule) if c_schedule else 0,
        gpu_hours=gpu_hours,
        cost_usd=cost,
        sla_safe_goodput=goodput,
        goodput_per_dollar=goodput / cost,
        n_total=n_total,
        n_sla_safe=n_sla_safe,
        sla_violations=n_total - n_sla_safe,
        completion_rate=len(response) / max(1, n_total),
        p50_wait_s=_wait_percentile(wait_map, 50),
        p95_wait_s=_wait_percentile(wait_map, 95),
        p99_wait_s=_wait_percentile(wait_map, 99),
        p99_response_s=summary.get("p99_response_s", 0.0),
        forecast=forecast,
        runtime_s=runtime_s,
    )


# ---------------------------------------------------------------------------
# small numeric helpers (pure-python; no numpy dependency)
# ---------------------------------------------------------------------------

def _quantile(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _running_median(xs: list, window: int) -> float:
    if not xs:
        return 1.0
    win = xs[-window:]
    s = sorted(win)
    return float(s[len(s) // 2])
