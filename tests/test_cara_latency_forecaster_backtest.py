"""Tests for the CARA latency forecaster v1 routing/placement backtest."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


BACKTEST_PATH = os.path.join(
    REPO_ROOT, "data", "external", "forecasting",
    "cara_latency_forecaster_v1", "backtest_summary.json",
)


@pytest.fixture(scope="module")
def backtest():
    if not os.path.exists(BACKTEST_PATH):
        pytest.skip("backtest_summary.json not generated; run "
                    "scripts/run_cara_latency_forecaster_v1_backtest.py first")
    with open(BACKTEST_PATH) as fh:
        return json.load(fh)


# ---------- 1. Backtest payload invariants -------------------------------


def test_backtest_payload_invariants(backtest):
    assert backtest["doc_version"] == "cara_latency_forecaster_v1_backtest_v1"
    assert backtest["production_claim"] is False
    assert backtest["modifies_robust_energy_engine"] is False
    assert backtest["modifies_controllers_or_defaults"] is False
    assert backtest["uses_oracle_as_headline"] is False
    assert backtest["shadow_only"] is True


def test_backtest_carries_counterfactual_caveat(backtest):
    note = backtest.get("result_quality_for_counterfactuals", "")
    assert "counterfactual" in note.lower() or "bucket" in note.lower()


def test_backtest_covers_required_targets(backtest):
    assert set(backtest["per_target"].keys()) >= {
        "actual_ttft_s", "actual_e2e_latency_s",
    }


# ---------- 2. Policies present + comparable -----------------------------


REQUIRED_POLICIES = (
    "round_robin",
    "per_instance_type_p95",
    "per_instance_type_p95_with_queue",
    "ml_hgb_p95",
)


def test_every_target_lists_required_policies(backtest):
    for target, payload in backtest["per_target"].items():
        for policy in REQUIRED_POLICIES:
            assert policy in payload["policies"], (
                f"{target} missing policy '{policy}'"
            )


def test_every_policy_reports_p50_p95_p99_realised_latency(backtest):
    for target, payload in backtest["per_target"].items():
        for name, p in payload["policies"].items():
            for k in ("realised_latency_p50", "realised_latency_p95",
                      "realised_latency_p99"):
                assert k in p, f"{target}/{name} missing {k}"


def test_policy_reports_instance_pick_counts(backtest):
    for target, payload in backtest["per_target"].items():
        for name, p in payload["policies"].items():
            counts = p["instance_pick_counts"]
            assert isinstance(counts, dict) and counts


# ---------- 3. Counterfactual quality label is honest -------------------


def test_per_target_results_labelled_counterfactual_proxy(backtest):
    for target, payload in backtest["per_target"].items():
        assert payload["result_quality_label"] == \
            "counterfactual_bucket_mean_proxy", (
                f"{target} result_quality_label must be "
                "counterfactual_bucket_mean_proxy when CARA cannot supply "
                "ground-truth counterfactuals."
            )


# ---------- 4. Tail-delta + promotion classification --------------------


def test_tail_delta_keys_present(backtest):
    for target, payload in backtest["per_target"].items():
        td = payload["tail_delta_ml_vs_strongest_baseline_pct"]
        for k in ("p50", "p90", "p95", "p99"):
            assert k in td


def test_promotion_classification_in_closed_enum(backtest):
    valid = {
        "diagnostic_only", "promising_needs_validation",
        "candidate_for_shadow_integration",
        "strong_candidate_for_shadow_integration",
    }
    for target, payload in backtest["per_target"].items():
        assert payload["ml_routing_promotion_classification"] in valid


def test_goodput_per_dollar_explicitly_not_evaluated(backtest):
    for target, payload in backtest["per_target"].items():
        assert payload["goodput_per_dollar"] is None
        reason = payload["goodput_per_dollar_skipped_reason"]
        assert "no real cost mapping" in reason.lower()


# ---------- 5. No scheduler / no production claim in code ---------------


def test_backtest_script_does_not_reference_executor_paths():
    """The backtest must remain shadow-only — no executor invocation."""
    script = os.path.join(
        REPO_ROOT, "scripts", "run_cara_latency_forecaster_v1_backtest.py",
    )
    with open(script) as fh:
        src = fh.read()
    banned = ("execute_frontier_decision", "set_replicas",
              "apply_replica_scale", "RUN_FOR_REAL")
    for b in banned:
        assert b not in src, (
            f"backtest script references executor token '{b}'"
        )


def test_no_production_savings_phrase_in_script():
    script = os.path.join(
        REPO_ROOT, "scripts", "run_cara_latency_forecaster_v1_backtest.py",
    )
    with open(script) as fh:
        src = fh.read().lower()
    for phrase in ("production savings", "hyperscaler-validated"):
        assert phrase not in src


# ---------- 6. Raw + analysis samples gitignored -----------------------


def test_cara_train_raw_and_analysis_sample_remain_gitignored():
    raw = os.path.join(
        REPO_ROOT, "data", "external", "hf",
        "asdwb__cara_latency_prediction", "raw", "train.jsonl",
    )
    analysis = os.path.join(
        REPO_ROOT, "data", "external", "hf",
        "asdwb__cara_latency_prediction", "train_flat",
        "processed", "analysis_sample.jsonl",
    )
    for path in (raw, analysis):
        if not os.path.exists(path):
            continue
        r = subprocess.run(
            ["git", "check-ignore", path], cwd=REPO_ROOT,
            capture_output=True, text=True,
        )
        assert r.returncode == 0, (
            f"CARA train data must remain gitignored: {path}"
        )


# ---------- 7. Candidate_for_shadow_integration only when gates pass --


def test_no_shadow_promotion_unless_alpha_above_threshold(backtest):
    """Each per-target routing promotion must respect the 5% threshold + zero
    safety regression rule."""
    for target, payload in backtest["per_target"].items():
        td = payload["tail_delta_ml_vs_strongest_baseline_pct"]
        safety = payload["safety_regression"]
        promotion = payload["ml_routing_promotion_classification"]
        max_alpha = max(
            v if v is not None else -1e9
            for v in (td.get("p95"), td.get("p99"))
        )
        if promotion == "candidate_for_shadow_integration":
            assert max_alpha >= 5.0
            assert safety == 0
        if promotion == "strong_candidate_for_shadow_integration":
            assert max_alpha >= 10.0
            assert safety == 0
        if promotion == "diagnostic_only":
            # Either alpha too small or safety regression.
            assert max_alpha < 2.0 or safety == 1
