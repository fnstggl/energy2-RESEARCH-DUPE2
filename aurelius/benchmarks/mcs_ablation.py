"""Ablation models for forecasted-MCS component evaluation.

Composes the deployable forecasted-MCS capacity policy with three optional
Aurelius components and measures each against the forecasted-MCS baseline under
ONE physics model / ONE SLA / ONE cost denominator (provisioned GPU-hours over
the fixed trace window):

  * queue policy  — abs-conformal SRTF ordering (numerator lever); handled
    directly via ``evaluate_c_schedule(discipline="abs_conformal")``.
  * energy routing — real CAISO/PJM/ERCOT day-ahead prices added to the GPU-hour
    cost as an energy term, routed to the cheapest region per tick (cost lever).
  * placement — a real heterogeneous GPU menu (median on-demand $/gpu-hr from the
    committed price overlay) paired with documented decode-throughput ratios; per
    tick the policy picks the cheapest GPU whose min-c still meets the Erlang-C
    gate at the forecasted load (joint capacity+hardware cost lever).

Cost levers are also applied to the no-MCS fixed baseline so the *interaction*
(does the component help forecasted MCS MORE than it helps the baseline?) can be
isolated — a denominator discount that helps every policy equally is a
procurement choice, not a capacity component (see research/MCS_AUDIT.md).

Directional simulator evidence only — NOT production savings (docs/RESULTS.md §8).
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass

from aurelius.benchmarks.forecasted_mcs import _wait_percentile, bucketize
from aurelius.benchmarks.srtf_serving_backtest import (
    GPU_HOUR_USD,
    TPOT_S,
    TTFT_BASE_S,
    _erlang_c_sla_timeout_pct,
    _Request,
    _service_time_s,
    _simulate_fifo_variable_c,
    _sla_safe_goodput,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OVERLAY = os.path.join(
    _REPO_ROOT, "data", "external", "economic_overlay", "economic_overlay_samples"
)

# --- energy model constants -------------------------------------------------
# A10-class serving GPU board+host power. The benchmark's $2.00/gpu-hr ≈ an A10
# on-demand rate; A10 TDP 150 W, host-normalized to ~0.40 kW per replica.
GPU_POWER_KW: float = 0.40
ENERGY_PRICE_FILES = {
    "CAISO": "caiso_da_energy_price_7day.jsonl",
    "PJM": "pjm_da_energy_price_14day.jsonl",
    "ERCOT": "ercot_da_energy_price_14day.jsonl",
}

# --- placement model: real median $/gpu-hr (gpu_price_overlay_multiday.jsonl,
# on-demand) paired with documented relative decode throughput for 7-13B serving.
# tpot_s = decode seconds/token; ttft_s = prefill component. A10 is the
# benchmark's reference physics (TPOT_S=0.020, TTFT_BASE_S=0.150).
@dataclass(frozen=True)
class Gpu:
    name: str
    price_usd_hr: float
    tpot_s: float
    ttft_s: float


GPU_MENU: tuple = (
    Gpu("H100", 9.73, 0.0080, 0.100),   # ~125 tok/s (2.5x), fast+pricey
    Gpu("A100", 3.52, 0.0133, 0.120),   # ~75 tok/s (1.5x)
    Gpu("A10", 2.00, TPOT_S, TTFT_BASE_S),  # ~50 tok/s — benchmark reference
    Gpu("L4", 1.18, 0.0400, 0.180),     # ~25 tok/s (0.5x)
    Gpu("T4", 1.15, 0.0571, 0.200),     # ~17.5 tok/s (0.35x), cheap+slow
)


# ---------------------------------------------------------------------------
# Causal per-tick demand forecast (same EWMA logic as forecast_mcs_c_schedule,
# but also exposes arr_hat / svc_hat so placement can re-size per GPU).
# ---------------------------------------------------------------------------

def causal_tick_forecasts(
    raw: list,
    tick_seconds: float,
    warp: float,
    *,
    ewma_alpha: float = 0.5,
    warmup_ticks: int = 1,
    warmup_arr: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Return ``(arr_hat, svc_hat)`` per tick, each causal (uses ticks < t only).

    ``arr_hat[t]`` is the EWMA of past per-tick arrival counts; ``svc_hat[t]`` is
    the EWMA of past per-tick mean service. Warmup ticks emit ``warmup_arr`` /
    global-first estimates. Mirrors ``forecast_mcs_c_schedule`` exactly.
    """
    counts, token_lists, n_ticks = bucketize(raw, tick_seconds, warp)
    arr_hat: list[float] = []
    svc_hat: list[float] = []
    ewma_count = None
    ewma_svc = None
    hist = False
    for t in range(n_ticks):
        if t < warmup_ticks or not hist:
            arr_hat.append(float(warmup_arr))
            svc_hat.append(_service_time_s(1))
        else:
            arr_hat.append(ewma_count if ewma_count is not None else 0.0)
            svc_hat.append(ewma_svc if ewma_svc is not None else _service_time_s(1))
        obs = counts[t]
        if token_lists[t]:
            tick_mean = statistics.mean(_service_time_s(tok) for tok in token_lists[t])
            ewma_svc = tick_mean if ewma_svc is None else ewma_alpha * tick_mean + (1 - ewma_alpha) * ewma_svc
            hist = True
        ewma_count = float(obs) if ewma_count is None else ewma_alpha * obs + (1 - ewma_alpha) * ewma_count
    return arr_hat, svc_hat


# ---------------------------------------------------------------------------
# Energy routing
# ---------------------------------------------------------------------------

def _load_prices(path: str) -> list[float]:
    out: list[float] = []
    full = os.path.join(_OVERLAY, path)
    if not os.path.exists(full):
        return out
    with open(full) as fh:
        for line in fh:
            try:
                out.append(float(json.loads(line)["price_per_mwh"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return out


def per_tick_energy_usd_per_gpu_hr(n_ticks: int, *, route: bool) -> list[float]:
    """Energy $/gpu-hr per tick from real day-ahead prices.

    ``route=True``: per tick take the cheapest of CAISO/PJM/ERCOT (energy-aware
    region routing). ``route=False``: a single flat region (PJM, a typical
    datacenter grid) at its median — the no-routing reference.
    Prices are cycled across ticks (the warped trace window is shorter than the
    price series, so this samples a contiguous slice).
    """
    series = {k: _load_prices(v) for k, v in ENERGY_PRICE_FILES.items()}
    series = {k: v for k, v in series.items() if v}
    if not series:
        return [0.0] * n_ticks
    if route:
        out = []
        for t in range(n_ticks):
            mwh = min(v[t % len(v)] for v in series.values())
            out.append(GPU_POWER_KW * max(0.0, mwh) / 1000.0)  # $/kWh = $/MWh/1000
        return out
    flat = series.get("PJM") or next(iter(series.values()))
    med = sorted(flat)[len(flat) // 2]
    return [GPU_POWER_KW * med / 1000.0] * n_ticks


def evaluate_with_energy(
    raw: list,
    c_schedule: list,
    tick_seconds: float,
    warp: float,
    sla_s: float,
    *,
    route: bool,
) -> dict:
    """FIFO sim (goodput unchanged) with an added real energy term in the $/gpu-hr.

    Total $/gpu-hr = GPU_HOUR_USD (infra/capital, already net of nominal energy)
    + energy(t). Energy routing only moves the energy term, which is ~0.5% of the
    GPU-hour cost — the measurement quantifies exactly how immaterial it is.
    """
    reqs = [
        _Request(idx=i, arrival_s=arr / warp, actual_tokens=tok,
                 predicted_tokens=float(tok), service_s=_service_time_s(tok))
        for i, (arr, tok) in enumerate(raw)
    ]
    summary, response, wait_map = _simulate_fifo_variable_c(reqs, c_schedule, tick_seconds)
    energy = per_tick_energy_usd_per_gpu_hr(len(c_schedule), route=route)
    tick_hr = tick_seconds / 3600.0
    cost = sum(c * tick_hr * (GPU_HOUR_USD + energy[t]) for t, c in enumerate(c_schedule))
    cost = max(cost, 1e-9)
    goodput = _sla_safe_goodput(reqs, response, sla_s)
    n_safe = sum(1 for r in reqs if r.idx in response and response[r.idx] <= sla_s)
    return {
        "goodput_per_dollar": goodput / cost,
        "cost_usd": cost,
        "energy_cost_usd": sum(c * tick_hr * energy[t] for t, c in enumerate(c_schedule)),
        "gpu_hours": sum(c_schedule) * tick_hr,
        "sla_violations": len(reqs) - n_safe,
        "p99_wait_s": _wait_percentile(wait_map, 99),
    }


# ---------------------------------------------------------------------------
# Placement (heterogeneous GPU menu)
# ---------------------------------------------------------------------------

def _min_c_for(arr_hat: float, svc_hat: float, sla_s: float, gate: float) -> int:
    if arr_hat <= 0.0:
        return 1
    lam = arr_hat
    sla_wait = max(0.0, sla_s - svc_hat)
    for c in range(1, 1024):
        if _erlang_c_sla_timeout_pct(lam, svc_hat, c, sla_wait) < gate:
            return c
    return 1023


def placement_schedule(
    raw: list,
    tick_seconds: float,
    warp: float,
    sla_s: float,
    *,
    gate: float = 9.5,
    ewma_alpha: float = 0.5,
    warmup_c: int = 4,
) -> tuple[list, list]:
    """Per-tick joint (GPU, c) choice from a causal forecast.

    For each tick, for each GPU g: scale the forecasted mean service to g's TPOT,
    size min-c meeting the gate, and cost = c_g * price_g. Pick the cheapest GPU.
    Returns ``(c_schedule, gpu_per_tick)`` — both deployable (forecast-driven).
    """
    arr_hat, svc_hat = causal_tick_forecasts(
        raw, tick_seconds, warp, ewma_alpha=ewma_alpha
    )
    tick_hr = tick_seconds / 3600.0
    c_sched: list[int] = []
    gpu_sched: list = []
    for t in range(len(arr_hat)):
        a = arr_hat[t] / tick_seconds
        # forecasted mean tokens from the A10-reference svc_hat
        mean_tokens = max(0.0, (svc_hat[t] - TTFT_BASE_S) / TPOT_S)
        if t == 0 or arr_hat[t] <= 0.0:
            c_sched.append(max(1, warmup_c) if t == 0 else 1)
            gpu_sched.append(GPU_MENU[2])  # A10 reference cold-start
            continue
        best = None
        for g in GPU_MENU:
            svc_g = g.ttft_s + mean_tokens * g.tpot_s
            c_g = _min_c_for(a, svc_g, sla_s, gate)
            cost_g = c_g * g.price_usd_hr * tick_hr
            if best is None or cost_g < best[0]:
                best = (cost_g, c_g, g)
        c_sched.append(best[1])
        gpu_sched.append(best[2])
    return c_sched, gpu_sched


def evaluate_with_placement(
    raw: list,
    c_schedule: list,
    gpu_per_tick: list,
    tick_seconds: float,
    warp: float,
    sla_s: float,
) -> dict:
    """Re-simulate with per-tick GPU: per-request service uses its arrival-tick's
    GPU TPOT/TTFT, and cost uses that tick's GPU price.
    """
    n_ticks = len(c_schedule)

    def _tick_of(arr_w: float) -> int:
        return min(int(arr_w / tick_seconds), n_ticks - 1)

    reqs = []
    for i, (arr, tok) in enumerate(raw):
        arr_w = arr / warp
        g = gpu_per_tick[_tick_of(arr_w)]
        svc = g.ttft_s + tok * g.tpot_s
        reqs.append(_Request(idx=i, arrival_s=arr_w, actual_tokens=tok,
                             predicted_tokens=float(tok), service_s=svc))
    summary, response, wait_map = _simulate_fifo_variable_c(reqs, c_schedule, tick_seconds)
    tick_hr = tick_seconds / 3600.0
    cost = max(sum(c * tick_hr * gpu_per_tick[t].price_usd_hr
                   for t, c in enumerate(c_schedule)), 1e-9)
    goodput = _sla_safe_goodput(reqs, response, sla_s)
    n_safe = sum(1 for r in reqs if r.idx in response and response[r.idx] <= sla_s)
    from collections import Counter
    mix = Counter(g.name for g in gpu_per_tick)
    return {
        "goodput_per_dollar": goodput / cost,
        "cost_usd": cost,
        "gpu_hours": sum(c_schedule) * tick_hr,
        "sla_violations": len(reqs) - n_safe,
        "p99_wait_s": _wait_percentile(wait_map, 99),
        "gpu_mix": dict(mix),
    }


__all__ = [
    "GPU_MENU",
    "GPU_POWER_KW",
    "causal_tick_forecasts",
    "per_tick_energy_usd_per_gpu_hr",
    "evaluate_with_energy",
    "placement_schedule",
    "evaluate_with_placement",
]
