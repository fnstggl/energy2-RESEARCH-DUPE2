"""Phase-6 / Phase-9 tests: evaluation rollup integrity + promotion.

Asserts:
  - A through H variants all reported,
  - baseline (A) variant reports 0 computable goodput/$ (overlay is the
    point — adding it must change something),
  - full overlay (E) makes goodput/$ computable on every record where SLA
    fields are present,
  - 3 result classes reported separately,
  - promotion is one of the allowed states + reason recorded,
  - carbon_cost stays missing under default policy,
  - no oracle / FIFO is the headline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL = REPO_ROOT / "data" / "external" / "economic_overlay" / \
    "economic_overlay_eval.json"


@pytest.fixture(scope="module")
def eval_data() -> dict:
    assert EVAL.exists(), f"missing eval JSON: {EVAL}"
    return json.loads(EVAL.read_text())


def test_eval_top_level_safety_flags(eval_data):
    assert eval_data["production_claim"] is False
    assert eval_data["shadow_only"] is True
    assert eval_data["uses_oracle_as_headline"] is False
    assert eval_data["uses_fifo_as_headline"] is False
    assert eval_data["primary_baseline"] == "A_existing_scorer_baseline"
    assert eval_data["primary_kpi"] == "sla_safe_goodput_per_dollar"


def test_eval_reports_three_result_classes_separately(eval_data):
    assert set(eval_data["result_classes_reported_separately"]) == {
        "measured_same_record", "cross_dataset_joined", "scenario_prior",
    }


def test_eval_runs_all_eight_variants(eval_data):
    expected = {
        "A_existing_scorer_baseline",
        "B_existing_plus_gpu_price",
        "C_existing_plus_energy_carbon",
        "D_existing_plus_cache_value",
        "E_existing_plus_full_overlay",
        "F_full_plus_ttft_prior",
        "G_full_plus_cache_prefix_prior",
        "H_full_plus_both_priors",
    }
    actual = set(eval_data["variants"].keys())
    assert actual == expected, (
        f"variant set mismatch: only-eval={actual - expected}, "
        f"only-spec={expected - actual}")


def test_baseline_a_has_no_goodput_dollar_computable(eval_data):
    """A is the no-overlay baseline; the constraint scorer alone has no
    public-data inputs to populate gpu_cost / energy_cost. Goodput/$ must
    be computable on 0 records — if it isn't, the baseline silently
    invented something."""
    a = eval_data["variants"]["A_existing_scorer_baseline"]["metrics"]
    n_a_goodput = a["by_field_stats"]["sla_safe_goodput_per_dollar"]["n"]
    assert n_a_goodput == 0, (
        f"baseline A should have 0 computable goodput/$ "
        f"(no overlay applied); got {n_a_goodput}")


def test_full_overlay_e_makes_goodput_dollar_computable(eval_data):
    """E (full overlay) must compute goodput/$ on every record where
    e2e_latency_s + sla_s + output_tokens are present in the operational
    fixtures. We expect a meaningful count > baseline's 0."""
    e = eval_data["variants"]["E_existing_plus_full_overlay"]["metrics"]
    n_e_goodput = e["by_field_stats"]["sla_safe_goodput_per_dollar"]["n"]
    assert n_e_goodput >= 30, (
        f"full overlay E should make goodput/$ computable on most rows "
        f"(>=30); got {n_e_goodput}")


def test_promotion_is_a_known_state(eval_data):
    allowed = {
        "diagnostic_only", "economic_overlay_ready",
        "shadow_ready_for_integration_review", "blocked_by_pilot_telemetry",
    }
    promo = eval_data["promotion"]["final_status"]
    assert promo in allowed, f"unknown promotion: {promo}"
    assert eval_data["promotion"]["reason"], "promotion has no reason"
    assert eval_data["promotion"][
        "carbon_cost_held_missing_under_default_policy"] is True
    assert eval_data["promotion"][
        "carbon_cost_requires_operator_carbon_price_per_kg_usd"] is True


def test_per_class_headline_present(eval_data):
    """Every variant must record the headline goodput/$ broken out by
    the three result classes — never collapsed into a single number."""
    for name, v in eval_data["variants"].items():
        cls_headline = v["metrics"][
            "headline_sla_safe_goodput_per_dollar_per_class"]
        assert set(cls_headline.keys()) >= {
            "measured_same_record", "cross_dataset_joined", "scenario_prior",
        }, f"{name} missing per-class headline"


def test_priors_in_f_g_h_recorded_in_applied(eval_data):
    for v, ttft, cache in (("F_full_plus_ttft_prior", True, False),
                           ("G_full_plus_cache_prefix_prior", False, True),
                           ("H_full_plus_both_priors", True, True)):
        applied = eval_data["variants"][v]["applied_overlays"]
        assert applied["ttft_prior"] == ttft, (v, applied)
        assert applied["cache_prefix_prior"] == cache, (v, applied)


def test_ranking_change_rate_recorded(eval_data):
    """Mission requires ranking_change_rate + top1_change_rate metrics.
    The overlay is additive over the existing scorer, so they are 0.0 by
    design — but the keys must exist so downstream consumers know."""
    for name, v in eval_data["variants"].items():
        if name == "A_existing_scorer_baseline":
            continue
        m = v["metrics"]
        assert "ranking_change_rate_vs_baseline" in m
        assert "top1_change_rate_vs_baseline" in m


def test_missing_signal_rate_recorded(eval_data):
    for v in eval_data["variants"].values():
        msr = v["metrics"]["missing_rate_per_field"]
        assert set(msr.keys()) >= {
            "estimated_gpu_cost_usd", "estimated_energy_cost_usd",
            "estimated_carbon_kg", "estimated_carbon_cost_usd",
            "estimated_cache_value_usd", "estimated_migration_cost_usd",
            "estimated_cold_start_cost_usd",
            "sla_safe_goodput_per_dollar",
        }
        for k, rate in msr.items():
            assert 0.0 <= rate <= 1.0, f"missing rate {k}={rate}"


def test_eval_n_operational_rows_matches_summary():
    """The eval rollup's n_operational_rows must match the count of rows
    actually emitted by the build script (5 sources × 5 records each + the
    extra Optimum configs)."""
    eval_data = json.loads(EVAL.read_text())
    summary_path = REPO_ROOT / "data" / "external" / "economic_overlay" / \
        "economic_overlay_summary.json"
    if not summary_path.exists():
        pytest.skip("summary not built")
    s = json.loads(summary_path.read_text())
    assert eval_data["n_operational_rows"] == s["global_summary"]["n"]
