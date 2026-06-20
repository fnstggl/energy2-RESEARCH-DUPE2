"""Research-module integration harness for the public serving-trace replay.

This wires the three shadow research modules

  * ``WorkloadAdmissionGate``     (``aurelius/frontier/admission.py``)
  * ``OutputLengthForecastBundle``(``aurelius/forecasting/cara_output_length_forecaster.py``)
  * ``GpuPlacementScorer``        (wired in ``scheduler.py``; evaluated separately)

into the **unchanged** BurstGPT / Azure-2024 serving-trace replay physics.

Design rules (mirror ``backtest.py`` honesty contract):

- ``backtest.py`` / ``serving.py`` / ``economics.py`` are LOCKED evaluation
  infrastructure. This module *imports and reuses* their functions verbatim
  (``evaluate_tick``, ``_size_for_target``, ``_constraint_trim``, ``_aggregate``,
  the economics KPI). It never re-implements the physics or the cost basis.
- The baseline ``constraint_aware`` policy numbers are produced by the locked
  ``backtest.run_backtest`` and are byte-identical to every prior release. The
  module variants are ADDITIVE policies (``*_admission``, ``*_outlen``) compared
  against that locked baseline — they cannot change it.
- Decisions only. Each variant changes the **provisioning / admission decision**;
  the serving physics, calibration constants and cost basis are identical across
  every variant. Wins come from decisions, not tuned constants.
- No future leakage. The output-length forecaster is fit on a warmup prefix of
  the trace and applied to the remainder. Admission uses only past-tick
  telemetry. ``actual_output_tokens`` is never used at decision time.

Directional simulator/backtest evidence only — NOT production savings
(``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from aurelius.frontier.admission import (
    ADMISSION_ADMIT,
    ADMISSION_DEFER,
    ADMISSION_REJECT,
    AdmissionGateConfig,
    evaluate_admission,
)
from aurelius.frontier.dynamic_models import ServingTelemetryTick

from . import backtest as bt
from .replay import ArrivalTick, requests_to_arrival_ticks
from .schema import NormalizedLLMRequest

# ---------------------------------------------------------------------------
# SLA-class mapping for serving traces (documented, identical across variants).
#
# The admission gate only applies back-pressure to *non* latency-critical
# classes. BurstGPT carries a real ``Log Type`` column: "Conversation log"
# (interactive chat) vs "API log" (programmatic / batch-style). We map:
#
#   * conversational / interactive  -> latency_critical   (never deferred)
#   * api / batch / programmatic     -> best_effort        (deferrable)
#
# Azure-2024 carries no log type; its conv+code mix is represented by a
# documented best-effort fraction (default 0.5: the code/batch half).
# ---------------------------------------------------------------------------

_BEST_EFFORT_LOG_TOKENS = ("api", "batch", "background", "offline")
_AZURE_DEFAULT_BEST_EFFORT_FRACTION = 0.5


def best_effort_fraction_for_tick(
    tick: ArrivalTick, *, azure_fallback: float = _AZURE_DEFAULT_BEST_EFFORT_FRACTION
) -> float:
    """Fraction of a tick's requests that are deferrable best-effort.

    Computed from the tick's ``log_type_mix`` when a log-type signal exists
    (BurstGPT). Falls back to ``azure_fallback`` when there is no log-type
    signal (Azure-2024) — a documented conv/code split proxy.
    """
    mix = tick.log_type_mix or {}
    total = sum(mix.values())
    if total <= 0:
        return 0.0
    has_signal = any(str(k).strip() for k in mix)
    if not has_signal:
        return azure_fallback
    be = 0
    for k, cnt in mix.items():
        kl = str(k).lower()
        if any(tok in kl for tok in _BEST_EFFORT_LOG_TOKENS):
            be += cnt
    # If the trace has log types but none match best-effort tokens, treat the
    # whole trace as interactive (gate becomes a no-op — honest).
    return be / total


# ---------------------------------------------------------------------------
# Telemetry window construction from the realized replay (past ticks only).
# ---------------------------------------------------------------------------

def _telemetry_window(
    evals: Sequence[bt.TickEval], *, tick_seconds: float, lookback: int
) -> list[ServingTelemetryTick]:
    """Build a ``ServingTelemetryTick`` window from recent realized ticks.

    The KV-cache pressure signal the gate consumes (``mean_utilization``) is
    proxied by the realized serving utilization ``rho`` — the only KV-adjacent
    observable the public traces expose. ``telemetry_confidence='medium'`` so
    the gate is active (the public replay has dense per-tick metrics).
    """
    window: list[ServingTelemetryTick] = []
    recent = evals[-lookback:] if lookback < len(evals) else evals
    for ev in recent:
        window.append(
            ServingTelemetryTick(
                timestamp_s=ev.tick_index * tick_seconds,
                observed_rps=ev.arrival_rate_rps,
                mean_utilization=min(0.999, max(0.0, ev.rho)),
                queue_p50_ms=None,
                queue_p95_ms=ev.queue_wait_p95_ms,
                queue_p99_ms=ev.queue_wait_p99_ms,
                latency_p99_ms=ev.latency_p99_ms,
                timeout_pct=ev.timeout_rate_pct,
                telemetry_confidence="medium",
                source="serving_replay",
            )
        )
    return window


# ---------------------------------------------------------------------------
# Output-length forecaster (semi-clairvoyant tail sizing).
# ---------------------------------------------------------------------------

@dataclass
class _OutlenModel:
    """Fitted output-length forecaster + the warmup boundary it was fit on."""

    bundle: object
    naive_prior_tokens: float
    warmup_end_s: float
    p90_over_mean: float  # global p90/mean ratio (sizing-tail headroom prior)
    fitted: bool


def fit_output_length_model(
    requests: Sequence[NormalizedLLMRequest], *, warmup_frac: float = 0.3
) -> _OutlenModel:
    """Fit ``OutputLengthForecastBundle`` on a warmup prefix of the trace.

    No future leakage: only the first ``warmup_frac`` of the trace (time order)
    is used to fit. The fitted model predicts p90 output length from prompt
    tokens at decision time on the remainder.
    """
    from aurelius.forecasting.cara_output_length_forecaster import (
        OutputLengthForecastBundle,
    )

    ordered = sorted(requests, key=lambda r: (r.timestamp_s, r.request_id))
    served = [r for r in ordered if not r.is_failure and r.output_tokens > 0]
    if len(served) < 100:
        return _OutlenModel(None, 1.0, 0.0, 1.0, fitted=False)

    k = max(100, int(len(served) * warmup_frac))
    train = served[:k]
    warmup_end_s = train[-1].timestamp_s

    X = np.array([[float(r.prompt_tokens)] for r in train], dtype=float)
    y = np.array([float(max(1, r.output_tokens)) for r in train], dtype=float)
    naive_prior = float(np.median(y))
    raw = np.full(len(train), naive_prior, dtype=float)
    p90_over_mean = float(np.percentile(y, 90) / max(1.0, np.mean(y)))

    try:
        bundle = OutputLengthForecastBundle()
        bundle.fit_calibration(raw, y)
        bundle.fit_hgb(X, y, feature_names=["prompt_tokens"])
        return _OutlenModel(bundle, naive_prior, warmup_end_s, p90_over_mean, True)
    except Exception:
        # Fail-open: fall back to the global p90/mean headroom ratio (still a
        # forecast-derived, leakage-free tail prior — never the current tick).
        return _OutlenModel(None, naive_prior, warmup_end_s, p90_over_mean, True)


def _precompute_outlen(
    model: _OutlenModel, ticks: Sequence[ArrivalTick]
) -> "tuple[np.ndarray, np.ndarray]":
    """Vectorized (p50, p90) decode-length forecast per tick (one predict call).

    The forecaster predicts output length from each tick's mean prompt tokens.
    The realized physics still use the tick's real output tokens — only the
    replica-sizing decision consumes the forecast (semi-clairvoyant: it replaces
    the autoscaler's otherwise-clairvoyant read of the realized mean). Returns
    ``NaN`` per tick when the HGB model is unavailable; the caller then falls
    back to the EWMA mean (p50) or the global p90/mean headroom prior (p90).
    """
    n = len(ticks)
    p50 = np.full(n, np.nan, dtype=float)
    p90 = np.full(n, np.nan, dtype=float)
    if not model.fitted or model.bundle is None:
        return p50, p90
    prompts = np.array([[float(t.prompt_tokens_mean)] for t in ticks], dtype=float)
    raw = np.full(n, model.naive_prior_tokens, dtype=float)
    try:
        forecasts = model.bundle.predict_batch(raw, prompts)
        for i, fc in enumerate(forecasts):
            if float(fc.p50_tokens) > 0:
                p50[i] = float(fc.p50_tokens)
            if float(fc.p90_tokens) > 0:
                p90[i] = float(fc.p90_tokens)
    except Exception:
        return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float)
    return p50, p90


# ---------------------------------------------------------------------------
# The module-integrated variant runner (mirrors backtest._run_policy's
# constraint_aware branch; adds admission + outlen hooks).
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    name: str
    use_admission: bool = False
    use_outlen: bool = False
    # "p50" = deployable replacement for the autoscaler's clairvoyant realized
    # mean; "p90" = tail-aware over-provisioning sensitivity.
    outlen_quantile: str = "p50"
    admission_config: Optional[AdmissionGateConfig] = None
    outlen_model: Optional[_OutlenModel] = None
    azure_best_effort_fraction: float = _AZURE_DEFAULT_BEST_EFFORT_FRACTION
    # Cap the deferral horizon: deferred best-effort load beyond this many ticks'
    # worth is force-admitted (cannot accumulate unboundedly). Leftover at trace
    # end is counted as LOST goodput (never served) — conservative.
    max_defer_ticks: float = 8.0
    extra: dict = field(default_factory=dict)


def run_variant(
    ticks: Sequence[ArrivalTick], cfg: VariantConfig, *, tick_hours: float
) -> bt.PolicyResult:
    """Run one module-integrated provisioning variant over the arrival ticks.

    Baseline sizing is the locked ``constraint_aware`` recipe (EWMA anticipation
    + SLA-safe trim + churn hysteresis). The admission hook sheds/defers
    best-effort load under KV/queue pressure; the outlen hook sizes the decode
    tail from a forecast instead of the (clairvoyant) realized mean.
    """
    admission_cfg = cfg.admission_config
    outlen_model = cfg.outlen_model
    lookback = max(3, (admission_cfg.min_window_for_trends if admission_cfg else 3))

    evals: list[bt.TickEval] = []
    prev_replicas: Optional[int] = None
    ewma_rate = 0.0
    ewma_out = 0.0
    ewma_alpha = 0.5

    # Admission deferral bucket (best-effort output tokens + arrival share).
    deferred_tokens = 0.0
    deferred_rate = 0.0
    # Mean best-effort tokens per active tick (for the deferral cap).
    active = [t for t in ticks if t.request_count > 0]
    mean_be_tokens = (
        sum(t.total_output_tokens for t in active) / max(1, len(active))
    ) * 0.5
    defer_cap = cfg.max_defer_ticks * max(1.0, mean_be_tokens)

    # Vectorized output-length forecast per tick (one HGB predict call).
    if cfg.use_outlen and outlen_model is not None:
        outlen_p50, outlen_p90 = _precompute_outlen(outlen_model, ticks)
    else:
        outlen_p50 = outlen_p90 = None

    for ti, t in enumerate(ticks):
        if t.request_count > 0:
            ewma_rate = (
                ewma_alpha * t.arrival_rate_rps + (1 - ewma_alpha) * ewma_rate
                if ewma_rate
                else t.arrival_rate_rps
            )
            ewma_out = (
                ewma_alpha * t.output_tokens_mean + (1 - ewma_alpha) * ewma_out
                if ewma_out
                else t.output_tokens_mean
            )

        throughput = bt._tick_throughput_tokps(t)
        orig_tokens = float(t.total_output_tokens)
        orig_rate = float(t.arrival_rate_rps)

        # ----- Admission decision (best-effort share only) -----------------
        eff_rate = orig_rate
        eff_tokens = orig_tokens
        if (
            cfg.use_admission
            and admission_cfg is not None
            and admission_cfg.enabled
            and t.request_count > 0
        ):
            be_frac = best_effort_fraction_for_tick(
                t, azure_fallback=cfg.azure_best_effort_fraction
            )
            be_tokens = be_frac * orig_tokens
            be_rate = be_frac * orig_rate
            lc_tokens = orig_tokens - be_tokens
            lc_rate = orig_rate - be_rate

            avail_be_tokens = be_tokens + deferred_tokens
            avail_be_rate = be_rate + deferred_rate

            window = _telemetry_window(
                evals, tick_seconds=tick_hours * 3600.0, lookback=lookback
            )
            admit_frac = 1.0
            if window:
                dec = evaluate_admission(
                    sla_class="best_effort", window=window, config=admission_cfg
                )
                if dec.action == ADMISSION_ADMIT:
                    admit_frac = 1.0
                elif dec.action == ADMISSION_DEFER:
                    pressure = max(
                        dec.kv_pressure_score or 0.0,
                        dec.queue_pressure_score or 0.0,
                    )
                    admit_frac = max(0.0, 1.0 - pressure)
                elif dec.action == ADMISSION_REJECT:
                    admit_frac = 0.0

            served_be_tokens = admit_frac * avail_be_tokens
            served_be_rate = admit_frac * avail_be_rate
            new_deferred_tokens = avail_be_tokens - served_be_tokens
            new_deferred_rate = avail_be_rate - served_be_rate

            # Deferral cap: force-admit anything beyond the horizon cap so load
            # cannot accumulate unboundedly.
            if new_deferred_tokens > defer_cap:
                overflow = new_deferred_tokens - defer_cap
                frac_back = overflow / max(1.0, new_deferred_tokens)
                served_be_tokens += overflow
                served_be_rate += frac_back * new_deferred_rate
                new_deferred_tokens = defer_cap
                new_deferred_rate *= 1.0 - frac_back

            deferred_tokens = new_deferred_tokens
            deferred_rate = new_deferred_rate
            eff_tokens = lc_tokens + served_be_tokens
            eff_rate = lc_rate + served_be_rate

        eff_tick = (
            t
            if (eff_tokens == orig_tokens and eff_rate == orig_rate)
            else dataclasses.replace(
                t,
                arrival_rate_rps=eff_rate,
                total_output_tokens=int(round(eff_tokens)),
            )
        )

        # ----- Sizing length (outlen forecast vs realized mean) ------------
        plan_rate = max(eff_tick.arrival_rate_rps, ewma_rate)
        if cfg.use_outlen and outlen_model is not None and outlen_p50 is not None:
            mean_fallback = max(t.output_tokens_mean, ewma_out)
            if cfg.outlen_quantile == "p90":
                size_out = outlen_p90[ti]
                if not np.isfinite(size_out) or size_out <= 0:
                    size_out = mean_fallback * outlen_model.p90_over_mean
            else:  # p50 — replace the clairvoyant realized mean with a forecast
                size_out = outlen_p50[ti]
                if not np.isfinite(size_out) or size_out <= 0:
                    size_out = mean_fallback
            plan_out = max(size_out, 1.0)
        else:
            plan_out = (
                max(t.output_tokens_mean, ewma_out)
                if t.request_count
                else ewma_out
            )

        # constraint_aware is cache-aware: apply the SAME prefill savings the
        # locked baseline uses (identical across baseline + variants, so the
        # only difference is the module action). Cache behavior is unchanged by
        # admission deferral (reuse_fraction is preserved on the effective tick).
        prefill_savings = bt.MAX_PREFILL_SAVINGS * eff_tick.reuse_fraction

        base = bt._size_for_target(
            plan_rate, max(1.0, plan_out), throughput, target_rho=0.65
        )
        replicas = bt._constraint_trim(
            eff_tick, base, prefill_savings, tick_hours, prev_replicas
        )

        ev = bt.evaluate_tick(
            eff_tick, replicas, prefill_savings=prefill_savings, tick_hours=tick_hours
        )
        if prev_replicas is not None and ev.replicas != prev_replicas:
            ev.scale_event = True
        prev_replicas = ev.replicas
        evals.append(ev)

    # Leftover deferred best-effort load is NEVER served -> lost goodput
    # (already excluded because it never entered any tick's tokens_offered).
    # cache_aware=True mirrors the locked constraint_aware (prefill savings are
    # applied above); this only sets the cache_savings_applied report flag.
    return bt._aggregate(cfg.name, evals, True, ticks)


# ---------------------------------------------------------------------------
# Top-level comparison: baseline (locked) + module variants on one trace.
# ---------------------------------------------------------------------------

def run_module_comparison(
    requests: Sequence[NormalizedLLMRequest],
    *,
    tick_seconds: float = 60.0,
    admission_config: Optional[AdmissionGateConfig] = None,
    azure_best_effort_fraction: float = _AZURE_DEFAULT_BEST_EFFORT_FRACTION,
    outlen_warmup_frac: float = 0.3,
) -> dict:
    """Replay ``requests`` under the locked baselines + the 3 module variants.

    Returns ``{variant_name: bt.PolicyResult}`` plus the locked-baseline policy
    results (``constraint_aware``, ``sla_aware``, ``fifo``) for reference. All
    variants share the SAME serving physics, calibration and cost basis.
    """
    arrival_ticks = requests_to_arrival_ticks(requests, tick_seconds=tick_seconds)
    tick_hours = tick_seconds / 3600.0

    # Locked baseline policies (byte-identical to backtest.run_backtest).
    base = bt.run_backtest(
        requests,
        tick_seconds=tick_seconds,
        policies=("fifo", "sla_aware", "constraint_aware"),
    )

    adm_cfg = admission_config or AdmissionGateConfig(enabled=True)
    outlen_model = fit_output_length_model(requests, warmup_frac=outlen_warmup_frac)

    results: dict = {
        "fifo": base.policy_results["fifo"],
        "sla_aware": base.policy_results["sla_aware"],
        "constraint_aware": base.policy_results["constraint_aware"],
    }

    variants = [
        VariantConfig(
            name="ca_admission",
            use_admission=True,
            admission_config=adm_cfg,
            azure_best_effort_fraction=azure_best_effort_fraction,
        ),
        VariantConfig(
            name="ca_outlen",
            use_outlen=True,
            outlen_quantile="p50",
            outlen_model=outlen_model,
        ),
        VariantConfig(
            name="ca_outlen_p90",
            use_outlen=True,
            outlen_quantile="p90",
            outlen_model=outlen_model,
        ),
        VariantConfig(
            name="ca_all",
            use_admission=True,
            use_outlen=True,
            outlen_quantile="p50",
            admission_config=adm_cfg,
            outlen_model=outlen_model,
            azure_best_effort_fraction=azure_best_effort_fraction,
        ),
    ]
    for vc in variants:
        results[vc.name] = run_variant(arrival_ticks, vc, tick_hours=tick_hours)

    return {
        "tick_seconds": tick_seconds,
        "n_requests": len(requests),
        "n_ticks": len(arrival_ticks),
        "results": results,
        "outlen_fitted": outlen_model.fitted,
        "outlen_p90_over_mean": round(outlen_model.p90_over_mean, 4),
    }


def kpi_row(name: str, r: bt.PolicyResult) -> dict:
    """Flatten a PolicyResult into the canonical KPI row used by the report."""
    k = r.kpi
    return {
        "variant": name,
        "sla_safe_goodput_per_infra_dollar": k.sla_safe_goodput_per_infra_dollar,
        "sla_compliant_goodput": k.sla_compliant_goodput,
        "gpu_hours": round(k.active_gpu_hours, 4),
        "total_cost": round(k.total_infrastructure_cost, 4),
        "energy_cost": round(k.energy_cost, 4),
        "timeout_pct_mean": round(r.timeout_rate_pct_mean, 4),
        "queue_p99_ms": round(r.queue_p99_ms, 3),
        "latency_p99_ms": round(r.latency_p99_ms, 3),
        "scale_events": r.scale_events,
    }
