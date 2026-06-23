"""Phase 3 guard — public benchmark entry points route through AureliusOptimizer
with 0% KPI drift (architecture unification, not a research change).

Pins the Phase 3 contract of the canonical-optimizer unification
(``research/OPTIMIZER_UNIFICATION_PLAN.md``):

  * the routed energy benchmarks construct ``AureliusOptimizer`` (not
    ``JobScheduler`` directly), and the serving benchmark dispatches through
    ``AureliusOptimizer(policy="serving_queue")``;
  * routing is behavior-identical (energy: routed == direct scheduler; serving:
    shim == extracted impl) → 0% KPI drift;
  * no benchmark definitions/constants changed, no new priors added, no FIFO-only
    claim promoted, no actual-output-token decision-time leakage.

The authoritative 0%-drift benchmarks are also pinned by
``tests/test_canonical_energy_backtest.py`` (canonical golden snapshot, now
exercising the routed code) and ``tests/test_abs_conformal_backtest.py`` (serving
discipline via the benchmark re-export), plus
``research/results/canonical_optimizer_phase3_benchmark_routing_parity_2026-06-22.md``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

import aurelius.benchmarks.canonical_backtests as canon
import aurelius.benchmarks.gpu_routing_backtest as gpu_rt
import aurelius.benchmarks.srtf_backtest as srtf_bt
import aurelius.benchmarks.srtf_contention_backtest as srtf_cont
import aurelius.benchmarks.srtf_serving_backtest as serving
import aurelius.optimizer.policies.serving_queue as sq
from aurelius.models import OptimizationConfig
from aurelius.optimization.scheduler import JobScheduler
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.policies.serving_queue import AbsoluteErrorConformalCalibrator

REPO = Path(__file__).resolve().parent.parent
ENERGY_BENCH = [canon, srtf_bt, srtf_cont, gpu_rt]

# Matches an actual JobScheduler *construction* (assignment / return), not a
# docstring mention like ``JobScheduler(cfg)``.
_JOBSCHED_CTOR = re.compile(r"(=|return)\s*JobScheduler\(")


# ---------------------------------------------------------------------------
# 1. Entry points route through AureliusOptimizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod", ENERGY_BENCH, ids=lambda m: m.__name__.split(".")[-1])
def test_energy_benchmark_constructs_aurelius_optimizer(mod):
    src = Path(mod.__file__).read_text()
    assert "AureliusOptimizer(config=" in src, (
        f"{mod.__name__} does not construct AureliusOptimizer"
    )
    offenders = [ln.strip() for ln in src.splitlines() if _JOBSCHED_CTOR.search(ln)]
    assert not offenders, (
        f"{mod.__name__} still constructs JobScheduler directly: {offenders}"
    )


def test_serving_benchmark_routes_through_optimizer():
    assert isinstance(serving._SERVING_QUEUE_OPTIMIZER, AureliusOptimizer)
    assert serving._SERVING_QUEUE_OPTIMIZER.policy_name == "serving_queue"
    src = Path(serving.__file__).read_text()
    assert "_SERVING_QUEUE_OPTIMIZER.optimize(" in src, (
        "serving shim does not route through the AureliusOptimizer facade"
    )


# ---------------------------------------------------------------------------
# 2. Routing is behavior-identical (0% KPI drift)
# ---------------------------------------------------------------------------

def test_energy_routing_matches_direct_scheduler():
    jobs = canon.build_canonical_jobs(canon.CANONICAL_SEED, 1000)
    da, _rt = canon.load_canonical_price_data()
    carbon = {r: {} for r in da}
    cfg = OptimizationConfig(default_region=canon.REGION_PJM, min_power_fraction=1.0)

    direct = JobScheduler(cfg).solve(jobs, da, carbon, method=canon.CANONICAL_METHOD)
    routed = AureliusOptimizer(config=cfg).optimize(jobs, da, carbon, method=canon.CANONICAL_METHOD)

    def _h(sched):
        return [(d.job_id, d.region, d.start_time.isoformat(), round(d.power_fraction, 6))
                for d in sched]

    assert _h(routed.schedule) == _h(direct.schedule)
    assert round(routed.objective.total, 9) == round(direct.objective.total, 9)
    # baseline step the benchmark also uses
    db = {d.job_id: d.region for d in JobScheduler(cfg).create_baseline_schedule(jobs)}
    rb = {d.job_id: d.region for d in AureliusOptimizer(config=cfg).create_baseline_schedule(jobs)}
    assert rb == db


def _mk_serving(n=60, seed=5):
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(2.0)
        actual = rng.randint(5, 800)
        reqs.append(serving._Request(
            idx=i, arrival_s=t, actual_tokens=actual,
            predicted_tokens=max(1.0, actual + rng.gauss(0.0, 50.0)),
            service_s=serving._service_time_s(actual),
        ))
    return reqs


def test_serving_shim_routing_matches_extracted_impl():
    reqs = _mk_serving()
    shim = serving._simulate_decoupled_hybrid_abs_conformal(
        reqs, 4, AbsoluteErrorConformalCalibrator()
    )
    impl = sq.simulate_decoupled_hybrid_abs_conformal(
        reqs, 4, AbsoluteErrorConformalCalibrator(), summarize=serving._summarize
    )
    assert shim[1] == impl[1]
    assert shim[2] == impl[2]
    assert shim[0] == impl[0]


# ---------------------------------------------------------------------------
# 3. No benchmark definitions/constants changed
# ---------------------------------------------------------------------------

def test_no_benchmark_definitions_changed():
    assert serving.TTFT_BASE_S == 0.150
    assert serving.TPOT_S == 0.020
    assert serving.CONFORMAL_ALPHA_MAX == 0.001
    assert serving.CONFORMAL_WARMUP == 100
    assert serving.CONFORMAL_WINDOW == 200
    assert serving.CONFORMAL_ABS_TARGET_P90_TOKENS == 500.0
    assert canon.CANONICAL_SEED == 20260201
    assert canon.CANONICAL_JOB_COUNT == 1000
    assert canon.CANONICAL_GPU_HOUR_USD == 2.0
    assert canon.CANONICAL_METHOD == "greedy"


# ---------------------------------------------------------------------------
# 4. No new priors added by routing; 5. no FIFO-only claim; 6. no leakage
# ---------------------------------------------------------------------------

def test_routing_added_no_priors():
    # routing is construction-only; the decision module stays prior-free
    sqsrc = Path(sq.__file__).read_text()
    for banned in ("sklearn", "lightgbm", "HistGradientBoosting", "import numpy", "import pandas"):
        assert banned not in sqsrc, f"a prior dependency leaked into the decision module: {banned}"


def test_no_fifo_only_claim_promoted():
    d = json.loads((REPO / "research/results/abs_conformal_backtest_2026-06-22.json").read_text())
    for tr in ("azure_llm_2024", "burstgpt_hf"):
        t = d["traces"][tr]
        assert "oracle_delta_pct" in t
        assert "rel_conformal_delta_pct" in t
        assert t.get("shadow_tag")


def test_no_actual_token_leakage_static():
    uses = [ln.strip() for ln in Path(sq.__file__).read_text().splitlines() if ".actual_tokens" in ln]
    assert uses
    for ln in uses:
        assert "calibrator.update(" in ln, f"decision-time actual-token leakage: {ln}"
