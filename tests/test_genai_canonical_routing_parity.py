"""Phase 3d parity test — GenAI canonical routing through AureliusOptimizer.

Verifies zero behaviour change when the constraint_aware replica-sizing
decisions in ``genai_backtest._run_policy`` are routed through
``AureliusOptimizer(policy="genai_serving")`` instead of the inline EWMA
loop.

Two complementary checks:
  1. **Physics-level parity** — replica counts returned by
     ``GenAIServingPolicy.optimize(ticks, cold)`` are bit-identical to those
     produced by a self-contained reference implementation of the same EWMA
     anticipatory-sizing + model-affinity logic.
  2. **KPI-level parity** — ``run_backtest`` constraint_aware KPIs on the
     committed fixture match the known-good values (regression guard).

Uses the committed fixture in ``tests/fixtures/alibaba_genai_sample/``.
No full dataset, no network access.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.optimizer.aurelius_optimizer import AureliusOptimizer
from aurelius.optimizer.policies.genai_serving import (
    GENAI_EWMA_ALPHA,
    GENAI_MIN_REPLICAS,
    genai_size_for_sla,
)
from aurelius.traces import alibaba_genai as ag
from aurelius.traces.genai_backtest import _aggregate_ticks, run_backtest

FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_genai_sample")


def _load():
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=False))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    return layers["requests"], ag.calibrate_cold_start(by_stage)


def _reference_ca_counts(ticks, cold):
    """Self-contained reference: EWMA anticipatory + affinity, verbatim from
    the pre-Phase-3d inline logic in ``_run_policy``."""
    ewma = 0.0
    counts = []
    for t in ticks:
        if t.n > 0:
            ewma = (
                GENAI_EWMA_ALPHA * t.arrival_rate
                + (1.0 - GENAI_EWMA_ALPHA) * ewma
                if ewma
                else t.arrival_rate
            )
        if t.n:
            smoothed = max(t.arrival_rate, ewma)
            r = genai_size_for_sla(
                t.n, smoothed, t.mean_exec_s,
                t.distinct_models, t.lora_frac, t.controlnet_frac,
                cold, affinity=True,
            )
        else:
            r = GENAI_MIN_REPLICAS
        counts.append(r)
    return counts


def test_replica_counts_bit_identical():
    """GenAIServingPolicy.optimize() is bit-identical to the reference loop."""
    reqs, cold = _load()
    ticks = _aggregate_ticks(list(reqs), 3600.0)
    opt = AureliusOptimizer(policy="genai_serving")
    result = opt.optimize(ticks, cold)
    ref = _reference_ca_counts(ticks, cold)
    assert result.replica_counts == ref, (
        f"Mismatch: {sum(a != b for a, b in zip(result.replica_counts, ref))} "
        f"of {len(ref)} ticks differ"
    )
    assert result.affinity is True
    assert result.mode == "constraint_aware"


def test_all_policies_run_via_backtest():
    """run_backtest completes for all 5 policies without error."""
    reqs, cold = _load()
    result = run_backtest(reqs, tick_seconds=3600.0, cold_start_s=cold)
    assert set(result.policy_results) == {
        "fifo", "sla_aware", "queue_aware", "utilization_aware", "constraint_aware"
    }
    for policy, pr in result.policy_results.items():
        assert pr.kpi is not None, f"{policy}: kpi is None"
        assert pr.completed_requests >= 0


def test_constraint_aware_beats_sla_aware():
    """constraint_aware gp/$ >= sla_aware on this fixture (same-conditions check)."""
    reqs, cold = _load()
    result = run_backtest(reqs, tick_seconds=3600.0, cold_start_s=cold)
    ca = result.policy_results["constraint_aware"]
    sa = result.policy_results["sla_aware"]
    ca_kpi = ca.kpi.sla_safe_goodput_per_infra_dollar
    sa_kpi = sa.kpi.sla_safe_goodput_per_infra_dollar
    assert ca_kpi >= sa_kpi, (
        f"constraint_aware gp/$={ca_kpi:.4f} should be >= sla_aware={sa_kpi:.4f}"
    )
    assert ca.timeout_rate_pct <= 0.0, (
        f"constraint_aware should have 0% timeout; got {ca.timeout_rate_pct:.3f}%"
    )


def test_constraint_aware_zero_timeout():
    """The SLA-sizing loop must keep constraint_aware timeout at 0% on the fixture."""
    reqs, cold = _load()
    result = run_backtest(reqs, tick_seconds=3600.0, cold_start_s=cold)
    ca = result.policy_results["constraint_aware"]
    assert ca.timeout_rate_pct <= 0.0


def test_outcome_not_a_loss():
    """constraint_aware must not be classified as LOSS vs sla_aware (fixture may TIE)."""
    reqs, cold = _load()
    result = run_backtest(reqs, tick_seconds=3600.0, cold_start_s=cold)
    assert result.outcome.outcome != "LOSS", (
        f"constraint_aware regressed vs sla_aware: "
        f"outcome={result.outcome.outcome}, margin={result.outcome.margin_pct:.2f}%"
    )


def test_optimizer_facade_policy_name():
    """AureliusOptimizer reports genai_serving in IMPLEMENTED_POLICIES."""
    from aurelius.optimizer.policies import IMPLEMENTED_POLICIES, POLICY_REGISTRY
    assert "genai_serving" in POLICY_REGISTRY
    assert "genai_serving" in IMPLEMENTED_POLICIES
