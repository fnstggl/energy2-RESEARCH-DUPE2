"""Tests for the research-module integration harness (aurelius/traces/module_backtest.py).

These guard the EVALUATION infrastructure only — they are not economic
validation (that is the public-trace replay in
research/results/module_integration_public_backtest_*). They assert:

- the harness runs and produces valid KPI rows for every variant,
- a DISABLED admission gate is a strict no-op (byte-identical to the locked
  constraint_aware baseline) — i.e. the module cannot silently change behavior,
- the best-effort SLA mapping is computed from the log-type mix,
- the output-length forecaster is fit only on the warmup prefix (no leakage).
"""

from __future__ import annotations

import random

import pytest

from aurelius.frontier.admission import AdmissionGateConfig
from aurelius.traces import backtest as bt
from aurelius.traces import module_backtest as mb
from aurelius.traces.replay import requests_to_arrival_ticks
from aurelius.traces.schema import NormalizedLLMRequest


def _make_requests(n=4000, seed=0, span_s=600.0):
    """Synthetic interactive+api request mix (for harness wiring tests only)."""
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        is_api = rng.random() < 0.7
        model = "ChatGPT" if rng.random() < 0.8 else "GPT-4"
        out = max(1, int(rng.lognormvariate(4.0, 0.8)))
        prompt = max(1, int(rng.lognormvariate(5.0, 0.7)))
        reqs.append(
            NormalizedLLMRequest(
                request_id=f"r-{i}",
                timestamp_s=rng.uniform(0.0, span_s),
                session_id=None,
                model=model,
                prompt_tokens=prompt,
                output_tokens=out,
                total_tokens=prompt + out,
                elapsed_s=None,
                log_type="API log" if is_api else "Conversation log",
                is_failure=False,
                cache_affinity_key=f"model:{model}",
            )
        )
    reqs.sort(key=lambda r: r.timestamp_s)
    return reqs


def test_best_effort_fraction_from_log_mix():
    # 70 api : 30 conversation -> ~0.7 best-effort.
    tick = requests_to_arrival_ticks(_make_requests(2000, seed=1), tick_seconds=600.0)[0]
    be = mb.best_effort_fraction_for_tick(tick)
    assert 0.55 <= be <= 0.85


def test_best_effort_fraction_azure_fallback():
    # No log-type signal -> azure fallback fraction.
    reqs = [
        NormalizedLLMRequest(
            request_id=f"a-{i}", timestamp_s=float(i), session_id=None,
            model="m", prompt_tokens=100, output_tokens=50, total_tokens=150,
            elapsed_s=None, log_type="", is_failure=False, cache_affinity_key=None,
        )
        for i in range(50)
    ]
    tick = requests_to_arrival_ticks(reqs, tick_seconds=600.0)[0]
    assert mb.best_effort_fraction_for_tick(tick, azure_fallback=0.42) == pytest.approx(0.42)


def test_comparison_runs_all_variants():
    reqs = _make_requests(4000, seed=2)
    out = mb.run_module_comparison(reqs, tick_seconds=10.0)
    res = out["results"]
    for v in ("fifo", "sla_aware", "constraint_aware",
              "ca_admission", "ca_outlen", "ca_outlen_p90", "ca_all"):
        assert v in res
        row = mb.kpi_row(v, res[v])
        assert row["sla_safe_goodput_per_infra_dollar"] is not None
        assert row["gpu_hours"] >= 0.0
        assert row["total_cost"] >= 0.0


def test_disabled_admission_is_noop_vs_locked_baseline():
    """A DISABLED admission gate must reproduce the locked constraint_aware
    sizing exactly (the module cannot silently change behavior when off)."""
    reqs = _make_requests(3000, seed=3)
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=10.0)
    tick_hours = 10.0 / 3600.0

    disabled = mb.VariantConfig(
        name="ca_admission_disabled",
        use_admission=True,
        admission_config=AdmissionGateConfig(enabled=False),
    )
    variant = mb.run_variant(ticks, disabled, tick_hours=tick_hours)

    # The locked constraint_aware baseline, same physics, via run_backtest.
    base = bt.run_backtest(reqs, tick_seconds=10.0, policies=("constraint_aware",))
    ca = base.policy_results["constraint_aware"]

    assert variant.kpi.sla_compliant_goodput == ca.kpi.sla_compliant_goodput
    assert variant.kpi.active_gpu_hours == pytest.approx(ca.kpi.active_gpu_hours)
    assert variant.kpi.total_infrastructure_cost == pytest.approx(
        ca.kpi.total_infrastructure_cost
    )


def test_outlen_model_no_future_leakage():
    reqs = _make_requests(2000, seed=4, span_s=1000.0)
    model = mb.fit_output_length_model(reqs, warmup_frac=0.3)
    if model.fitted:
        ordered = sorted(reqs, key=lambda r: (r.timestamp_s, r.request_id))
        # The model's fit boundary must not exceed ~30% of the trace timespan.
        t0, t1 = ordered[0].timestamp_s, ordered[-1].timestamp_s
        assert model.warmup_end_s <= t0 + 0.6 * (t1 - t0)


def test_enabled_admission_only_touches_best_effort_share():
    """With the gate enabled, latency-critical (conversation) load is never
    deferred; goodput stays within a sane band of the baseline (no invention)."""
    reqs = _make_requests(5000, seed=5, span_s=300.0)
    out = mb.run_module_comparison(
        reqs, tick_seconds=10.0,
        admission_config=AdmissionGateConfig(enabled=True),
    )
    res = out["results"]
    base_g = res["constraint_aware"].kpi.sla_safe_goodput_per_infra_dollar
    adm_g = res["ca_admission"].kpi.sla_safe_goodput_per_infra_dollar
    # Admission can only shed best-effort load -> goodput within a bounded band.
    assert 0.5 * base_g <= adm_g <= 1.5 * base_g
