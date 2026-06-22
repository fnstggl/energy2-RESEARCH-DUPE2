"""Phase 2 guard — the serving-queue discipline extracted into the canonical
optimizer is parity-identical to the benchmark and introduces no research change.

Pins the Phase 2 contract of the canonical-optimizer unification
(``research/OPTIMIZER_UNIFICATION_PLAN.md``):

  * the extracted policy is accessible through ``AureliusOptimizer`` and produces
    the same result as the benchmark's (re-exported) path,
  * no benchmark assumptions/constants changed,
  * no FIFO-only claim is promoted (multi-baseline + shadow-tagged reporting),
  * no actual-output-token leakage at decision time (``actual_tokens`` is read
    only by post-completion calibration, never to order pending work), and
  * no new calculated priors are introduced (the discipline reads
    ``predicted_tokens``; it never computes a predictor).

The authoritative 0% KPI-drift proof (serving + energy benchmarks) is recorded in
``research/results/canonical_optimizer_phase2_serving_policy_parity_2026-06-22.md``;
the 17 tests in ``tests/test_abs_conformal_backtest.py`` also exercise the
extracted calibrator + discipline via the benchmark's re-export.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import aurelius.optimizer.policies.serving_queue as sq
from aurelius.benchmarks import srtf_serving_backtest as bench
from aurelius.benchmarks.srtf_serving_backtest import (
    _Request,
    _service_time_s,
    _simulate_decoupled_hybrid_abs_conformal,
    _summarize,
)
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.policies import IMPLEMENTED_POLICIES, ServingQueuePolicy
from aurelius.optimizer.policies.serving_queue import AbsoluteErrorConformalCalibrator

REPO = Path(__file__).resolve().parent.parent


def _mk_requests(n=60, seed=0, service_from="actual"):
    """Synthetic serving requests with a noisy prediction (no leakage)."""
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(2.0)
        actual = rng.randint(5, 800)
        predicted = max(1.0, actual + rng.gauss(0.0, 60.0))
        base = actual if service_from == "actual" else predicted
        reqs.append(_Request(
            idx=i, arrival_s=t, actual_tokens=actual,
            predicted_tokens=predicted, service_s=_service_time_s(base),
        ))
    return reqs


# ---------------------------------------------------------------------------
# 1. Accessible through AureliusOptimizer / policy interface
# ---------------------------------------------------------------------------

def test_serving_policy_accessible_via_optimizer():
    assert "serving_queue" in IMPLEMENTED_POLICIES
    opt = AureliusOptimizer(policy="serving_queue")
    assert isinstance(opt.policy, ServingQueuePolicy)
    assert opt.policy_name == "serving_queue"
    reqs = _mk_requests(30)
    summary, resp, wait = opt.optimize(reqs, 3, summarize=_summarize)
    assert isinstance(summary, dict)
    assert len(resp) == len(reqs) == len(wait)


# ---------------------------------------------------------------------------
# 2. Extracted policy == benchmark path (parity of the extraction)
# ---------------------------------------------------------------------------

def test_policy_path_matches_benchmark_shim():
    reqs = _mk_requests(80, seed=7)
    # benchmark path (its shim injects the benchmark's _summarize)
    shim_summary, shim_resp, shim_wait = _simulate_decoupled_hybrid_abs_conformal(
        reqs, 4, AbsoluteErrorConformalCalibrator()
    )
    # canonical optimizer path (same extracted impl, same _summarize)
    pol_summary, pol_resp, pol_wait = AureliusOptimizer(policy="serving_queue").optimize(
        reqs, 4, summarize=_summarize, calibrator=AbsoluteErrorConformalCalibrator()
    )
    assert pol_resp == shim_resp
    assert pol_wait == shim_wait
    assert pol_summary == shim_summary


def test_benchmark_reexports_the_extracted_objects():
    # the benchmark must re-export the moved symbols so existing imports work
    assert bench.AbsoluteErrorConformalCalibrator is sq.AbsoluteErrorConformalCalibrator
    # the benchmark's discipline is now a thin shim over the extracted impl
    assert bench._abs_conformal_impl is sq.simulate_decoupled_hybrid_abs_conformal


def test_extraction_is_deterministic():
    reqs = _mk_requests(50, seed=3)
    a = _simulate_decoupled_hybrid_abs_conformal(reqs, 4, AbsoluteErrorConformalCalibrator())
    b = _simulate_decoupled_hybrid_abs_conformal(reqs, 4, AbsoluteErrorConformalCalibrator())
    assert a[1] == b[1] and a[2] == b[2] and a[0] == b[0]


# ---------------------------------------------------------------------------
# 3. No benchmark assumptions/constants changed
# ---------------------------------------------------------------------------

def test_benchmark_constants_unchanged():
    assert bench.TTFT_BASE_S == 0.150
    assert bench.TPOT_S == 0.020
    assert bench.DECOUPLED_HYBRID_ALPHA_DEFAULT == 0.001
    assert bench.CONFORMAL_ALPHA_MAX == 0.001
    assert bench.CONFORMAL_WARMUP == 100
    assert bench.CONFORMAL_WINDOW == 200
    assert bench.CONFORMAL_ABS_TARGET_P90_TOKENS == 500.0
    # extracted module constants match the benchmark originals exactly
    assert sq.CONFORMAL_ALPHA_MAX == bench.CONFORMAL_ALPHA_MAX
    assert sq.CONFORMAL_WARMUP == bench.CONFORMAL_WARMUP
    assert sq.CONFORMAL_WINDOW == bench.CONFORMAL_WINDOW
    assert sq.CONFORMAL_ABS_TARGET_P90_TOKENS == bench.CONFORMAL_ABS_TARGET_P90_TOKENS
    # calibrator defaults preserved
    c = AbsoluteErrorConformalCalibrator()
    assert (c.alpha_max, c.warmup, c.window, c.target_p90_abs_tokens) == (
        0.001, 100, 200, 500.0,
    )


# ---------------------------------------------------------------------------
# 4. No FIFO-only claim promoted (multi-baseline + shadow-tagged reporting)
# ---------------------------------------------------------------------------

def test_no_fifo_only_claim_promoted():
    d = json.loads((REPO / "research/results/abs_conformal_backtest_2026-06-22.json").read_text())
    for trace in ("azure_llm_2024", "burstgpt_hf"):
        t = d["traces"][trace]
        # reported against oracle + rel-conformal, not FIFO alone, and shadow-tagged
        assert "oracle_delta_pct" in t
        assert "rel_conformal_delta_pct" in t
        assert "abs_vs_oracle_retention_pct" in t
        assert t.get("shadow_tag")
    # the extracted decision module promotes no standalone headline claim itself
    src = Path(sq.__file__).read_text()
    assert "goodput_per_dollar" not in src  # no KPI/claim computation in the policy


# ---------------------------------------------------------------------------
# 5. No actual-output-token leakage at decision time
# ---------------------------------------------------------------------------

def test_no_actual_token_leakage_static():
    """Every read of a request's ``.actual_tokens`` must be a post-completion
    calibration update — never part of an ordering/dispatch decision."""
    src_lines = Path(sq.__file__).read_text().splitlines()
    uses = [ln.strip() for ln in src_lines if ".actual_tokens" in ln]
    assert uses, "expected the calibration update to read .actual_tokens"
    for ln in uses:
        assert "calibrator.update(" in ln, (
            f"actual_tokens read outside calibrator.update (decision-time leakage?): {ln}"
        )


def test_no_actual_token_leakage_behavioral():
    """With calibration inert (sub-warmup, alpha == alpha_max throughout),
    scrambling each pending request's actual_tokens must not change the schedule:
    ordering depends only on predicted-derived service, never on actual length."""
    n = 50  # < CONFORMAL_WARMUP (100) => current_alpha() is constant alpha_max
    reqs = _mk_requests(n, seed=11, service_from="predicted")
    rng = random.Random(99)
    scrambled = [
        _Request(idx=r.idx, arrival_s=r.arrival_s, actual_tokens=rng.randint(5, 800),
                 predicted_tokens=r.predicted_tokens, service_s=r.service_s)
        for r in reqs
    ]
    base = _simulate_decoupled_hybrid_abs_conformal(reqs, 4, AbsoluteErrorConformalCalibrator())
    scr = _simulate_decoupled_hybrid_abs_conformal(scrambled, 4, AbsoluteErrorConformalCalibrator())
    assert base[1] == scr[1], "pending requests' actual_tokens leaked into scheduling"
    assert base[2] == scr[2]


# ---------------------------------------------------------------------------
# 6. No new calculated priors introduced
# ---------------------------------------------------------------------------

def test_no_new_priors_introduced():
    src = Path(sq.__file__).read_text()
    for banned in ("sklearn", "lightgbm", "HistGradientBoosting", "RandomForest",
                   "import numpy", "import pandas"):
        assert banned not in src, f"extraction introduced a prior dependency: {banned}"
    # the discipline READS predicted_tokens; it must not COMPUTE/ASSIGN predictions
    assert "predicted_tokens =" not in src
    assert ".predicted_tokens =" not in src
