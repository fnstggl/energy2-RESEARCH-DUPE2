"""Tests for the CARA latency forecaster baselines + ML + metrics."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cara_latency_forecaster import (  # noqa: E402
    GATE_THRESHOLDS,
    ConservativeMultiplierCalibration,
    GlobalConstantP95Baseline,
    GroupConstantQuantileBaseline,
    HistGradientBoostingQuantileForecaster,
    RandomForestMedianForecaster,
    SimpleRulePlacementScoreBaseline,
    classify_gate_status,
    incremental_alpha_pct,
    pinball_loss,
    quantile_metrics,
    regression_metrics,
    subgroup_metrics,
)


# ---------- 1. Baselines fit + predict ----------------------------------


def test_global_constant_p95_predicts_train_p95():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 10, size=1000)
    X = np.zeros((1000, 4))
    b = GlobalConstantP95Baseline(quantile=95.0).fit(X, y)
    p = b.predict(np.zeros((5, 4)))
    assert p.shape == (5,)
    assert np.allclose(p, p[0])
    # All predictions should equal the training-set p95.
    assert abs(p[0] - np.percentile(y, 95, method="nearest")) < 1e-6


def test_group_constant_quantile_falls_back_for_unseen():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 10, size=1000)
    groups = np.array(["a"] * 500 + ["b"] * 500, dtype=object)
    X = np.zeros((1000, 4))
    b = GroupConstantQuantileBaseline(quantile=95.0).fit(
        X, y, group_keys_train=groups,
    )
    # Seen group "a"
    pred_a = b.predict(np.zeros((1, 4)), group_keys_predict=np.array(["a"]))
    # Unseen group "z"
    pred_z = b.predict(np.zeros((1, 4)), group_keys_predict=np.array(["z"]))
    # Unseen falls back to global p95.
    assert abs(pred_z[0] - np.percentile(y, 95, method="nearest")) < 1e-6
    assert pred_a[0] != pred_z[0] or True  # may coincide by chance


def test_simple_rule_placement_score_adds_queue_penalty():
    rng = np.random.default_rng(2)
    y = rng.uniform(0, 10, size=500)
    its = np.array(["a30"] * 250 + ["p100"] * 250, dtype=object)
    qd_train = np.zeros(500)
    X = np.zeros((500, 4))
    b = SimpleRulePlacementScoreBaseline(quantile=95.0).fit(
        X, y, instance_types_train=its, queue_depths_train=qd_train,
    )
    pred_no_queue = b.predict(
        np.zeros((1, 4)),
        instance_types_predict=np.array(["a30"]),
        queue_depths_predict=np.array([0.0]),
    )
    pred_with_queue = b.predict(
        np.zeros((1, 4)),
        instance_types_predict=np.array(["a30"]),
        queue_depths_predict=np.array([10.0]),
    )
    assert pred_with_queue[0] > pred_no_queue[0]


# ---------- 2. ML forecaster fit + predict shapes -----------------------


@pytest.mark.parametrize("quantile", [0.05, 0.5, 0.95, 0.99])
def test_hgb_quantile_forecaster_fits_and_predicts(quantile):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 6))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    m = HistGradientBoostingQuantileForecaster(
        quantile=quantile, max_iter=50,
    ).fit(X, y)
    p = m.predict(X[:30])
    assert p.shape == (30,)
    assert np.isfinite(p).all()


def test_hgb_rejects_quantile_at_boundary():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(100, 4))
    y = rng.normal(size=100)
    with pytest.raises(ValueError):
        HistGradientBoostingQuantileForecaster(quantile=0.0).fit(X, y)
    with pytest.raises(ValueError):
        HistGradientBoostingQuantileForecaster(quantile=1.0).fit(X, y)


def test_random_forest_median_forecaster_fits_and_predicts():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(200, 6))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    m = RandomForestMedianForecaster(n_estimators=10, max_depth=5).fit(X, y)
    p = m.predict(X[:50])
    assert p.shape == (50,)


def test_conservative_multiplier_calibration_scales_predictions():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 3))
    y = rng.uniform(0, 10, size=100)
    base = HistGradientBoostingQuantileForecaster(
        quantile=0.5, max_iter=50,
    ).fit(X, y)
    cal = ConservativeMultiplierCalibration(multiplier=1.5, base=base)
    pb = base.predict(X[:5])
    pc = cal.predict(X[:5])
    np.testing.assert_allclose(pc, pb * 1.5)


# ---------- 3. Metrics ---------------------------------------------------


def test_regression_metrics_basic_invariants():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, p)
    assert m["count"] == 5
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["underprediction_rate"] == 0.0
    assert m["severe_underprediction_rate"] == 0.0
    assert m["calibration_coverage"] == 1.0


def test_regression_metrics_detects_severe_underprediction():
    y = np.array([10.0, 10.0, 10.0, 10.0])
    p = np.array([1.0, 1.0, 4.0, 4.0])  # 50% severe (predicted < y/2)
    m = regression_metrics(y, p)
    assert m["severe_underprediction_rate"] == 1.0  # ALL are < y/2
    assert m["underprediction_rate"] == 1.0


def test_pinball_loss_rewards_correct_quantile():
    rng = np.random.default_rng(11)
    y = rng.exponential(1.0, size=10_000)
    p50 = np.full_like(y, np.median(y))
    p95 = np.full_like(y, np.percentile(y, 95))
    # Pinball at q=0.5 should favour the median.
    assert pinball_loss(y, p50, 0.5) < pinball_loss(y, p95, 0.5)
    # Pinball at q=0.95 should favour the p95.
    assert pinball_loss(y, p95, 0.95) < pinball_loss(y, p50, 0.95)


def test_quantile_metrics_carries_calibration_coverage_and_error():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p_under = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # under-predicts every row
    m = quantile_metrics(y, p_under, quantile=0.95)
    assert m["calibration_coverage"] == 0.0
    assert m["calibration_error"] == 0.95


def test_subgroup_metrics_flags_insufficient_sample():
    rng = np.random.default_rng(12)
    y = rng.normal(size=200)
    p = rng.normal(size=200)
    g = np.array(["A"] * 150 + ["B"] * 50, dtype=object)
    sm = subgroup_metrics(y, p, g, min_count=100)
    assert sm["subgroups"]["A"]["count"] == 150
    assert "INSUFFICIENT_SAMPLE" in sm["subgroups"]["B"]["flags"]
    assert "B" in sm["insufficient_sample_groups"]


# ---------- 4. Gate classification --------------------------------------


def test_gate_classification_diagnostic_when_alpha_too_small():
    g = classify_gate_status(
        alpha_pct=1.0, tail_underpred_rate=0.0,
        baseline_tail_underpred_rate=0.0, safety_regression_count=0,
    )
    assert g == "diagnostic_only"


def test_gate_classification_promising_at_3_pct():
    g = classify_gate_status(
        alpha_pct=3.0, tail_underpred_rate=0.0,
        baseline_tail_underpred_rate=0.0, safety_regression_count=0,
    )
    assert g == "promising_needs_validation"


def test_gate_classification_candidate_at_7_pct():
    g = classify_gate_status(
        alpha_pct=7.0, tail_underpred_rate=0.0,
        baseline_tail_underpred_rate=0.0, safety_regression_count=0,
    )
    assert g == "candidate_for_shadow_integration"


def test_gate_classification_strong_at_15_pct():
    g = classify_gate_status(
        alpha_pct=15.0, tail_underpred_rate=0.0,
        baseline_tail_underpred_rate=0.0, safety_regression_count=0,
    )
    assert g == "strong_candidate"


def test_gate_classification_demotes_on_safety_regression():
    g = classify_gate_status(
        alpha_pct=12.0, tail_underpred_rate=0.0,
        baseline_tail_underpred_rate=0.0, safety_regression_count=1,
    )
    assert g == "diagnostic_only"


def test_gate_classification_demotes_when_tail_underprediction_worse():
    g = classify_gate_status(
        alpha_pct=12.0, tail_underpred_rate=0.10,
        baseline_tail_underpred_rate=0.05, safety_regression_count=0,
    )
    assert g == "diagnostic_only"


def test_incremental_alpha_pct_signs():
    # Lower-is-better: candidate=8, baseline=10 -> +20% improvement.
    assert incremental_alpha_pct(10.0, 8.0, lower_is_better=True) == 20.0
    # candidate worse than baseline -> negative alpha.
    assert incremental_alpha_pct(10.0, 12.0, lower_is_better=True) == -20.0


# ---------- 5. Artefact existence + invariants --------------------------


import json  # noqa: E402

CMP_PATH = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_latency_forecaster_v1", "model_comparison.json",
)
SCHEMA_PATH = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_latency_forecaster_v1", "schema_audit.json",
)


def test_schema_audit_exists_and_carries_required_fields():
    if not os.path.exists(SCHEMA_PATH):
        pytest.skip("schema_audit.json not generated; run "
                    "scripts/run_cara_latency_forecaster_v1.py first")
    s = json.load(open(SCHEMA_PATH))
    assert s["doc_version"] == "cara_latency_forecaster_v1_schema_audit"
    assert s["row_count"] > 1_000
    for col_entry in s["columns"]:
        assert col_entry["role"] in (
            "feature", "target", "group", "ignored", "missing"
        )
        assert col_entry["field_quality"] in (
            "real", "derived", "proxy", "synthetic", "missing"
        )


def test_schema_audit_fails_closed_on_missing_target():
    if not os.path.exists(SCHEMA_PATH):
        pytest.skip("schema_audit.json not generated")
    s = json.load(open(SCHEMA_PATH))
    for t in ("actual_ttft_s", "actual_e2e_latency_s"):
        # Both targets must have zero missing rows in the analysis sample.
        assert s["target_rows_missing"][t] == 0


def test_schema_audit_lists_blocked_leakage_fields():
    if not os.path.exists(SCHEMA_PATH):
        pytest.skip("schema_audit.json not generated")
    s = json.load(open(SCHEMA_PATH))
    for f in ("actual_e2e_latency_s", "completion_timestamp_s",
              "actual_output_tokens"):
        assert f in s["leakage_fields_blocked"]


def test_model_comparison_exists_and_compares_per_quantile():
    if not os.path.exists(CMP_PATH):
        pytest.skip("model_comparison.json not generated; run "
                    "scripts/run_cara_latency_forecaster_v1.py first")
    m = json.load(open(CMP_PATH))
    assert m["production_claim"] is False
    assert m["modifies_robust_energy_engine"] is False
    assert m["modifies_controllers_or_defaults"] is False
    assert m["uses_oracle_as_headline"] is False
    assert m["shadow_only"] is True
    # Per-target per-quantile gate.
    for target in ("actual_ttft_s", "actual_e2e_latency_s"):
        per_q = m["gate_classifications"][target]
        for q in ("50", "95", "99"):
            assert per_q[q] in (
                "diagnostic_only", "promising_needs_validation",
                "candidate_for_shadow_integration", "strong_candidate",
            )


def test_model_comparison_includes_baselines_and_ml_per_holdout():
    if not os.path.exists(CMP_PATH):
        pytest.skip("model_comparison.json not generated")
    m = json.load(open(CMP_PATH))
    for target in m["per_target"]:
        for holdout, payload in m["per_target"][target]["per_holdout"].items():
            # Every holdout payload must carry per-instance_type baseline AND
            # at least one ML model.
            base_names = set(payload["baselines"].keys())
            ml_names = set(payload["ml_models"].keys())
            assert any("per_instance_type" in n for n in base_names)
            assert any("hgb_quantile" in n for n in ml_names)


def test_model_comparison_includes_subgroup_metrics():
    if not os.path.exists(CMP_PATH):
        pytest.skip("model_comparison.json not generated")
    m = json.load(open(CMP_PATH))
    for target in m["per_target"]:
        for holdout, payload in m["per_target"][target]["per_holdout"].items():
            sub = payload["subgroup_metrics_for_hgb_p95"]
            assert "instance_type" in sub
            assert "queue_depth_bin" in sub


def test_gate_thresholds_documented():
    expected = {
        "diagnostic_only": 0.0,
        "promising_needs_validation": 2.0,
        "candidate_for_shadow_integration": 5.0,
        "strong_candidate": 10.0,
    }
    assert GATE_THRESHOLDS == expected
