"""Tests for the Batch Inference Frontier Incremental Alpha Audit.

Pins:

1.  The audit summary JSON (when committed) carries the do-not-claim
    flags and the required keys.
2.  The pre-registered verdict is SHADOW_DIAGNOSTIC. A regression to
    PROPOSE_INTEGRATION fires loudly — flipping the verdict requires
    re-running the audit, deliberately updating this test, AND
    providing the alpha evidence in the PR.
3.  The audit script imports the dynamic batch estimator (i.e. the new
    module is wired into the audit).
4.  The audit script does NOT modify any constraint_aware or scheduler
    module — the audit is research-only.
5.  The audit doc (docs/BATCH_INFERENCE_FRONTIER_INCREMENTAL_ALPHA_AUDIT.md)
    is committed, references the required prior-art docs, and contains
    no unhedged production-savings phrases.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY = (REPO_ROOT / "data" / "external" / "frontier"
           / "batch_inference_frontier_incremental_alpha_audit_summary.json")
DOC = (REPO_ROOT / "docs"
       / "BATCH_INFERENCE_FRONTIER_INCREMENTAL_ALPHA_AUDIT.md")
SCRIPT = (REPO_ROOT / "scripts"
          / "run_batch_inference_frontier_incremental_alpha_audit.py")

REQUIRED_DO_NOT_FLAGS = (
    "production_claim",
    "ml_training",
    "modifies_serving_rho_controller",
    "modifies_constraint_aware_default",
    "uses_oracle_as_headline",
    "executable_in_real_cluster",
)

REQUIRED_DOC_REFS = (
    "docs/RESULTS.md",
    "docs/PUBLIC_TRACE_BACKTESTS.md",
    "docs/EVAL_AND_BATCH_FRONTIER_RESULTS.md",
    "docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md",
    "docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md",
    "docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md",
)

BANNED = (
    "production savings",
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)


@pytest.mark.skipif(not SUMMARY.exists(),
                    reason="audit summary not committed yet")
def test_audit_summary_shape_and_do_not_flags():
    with open(SUMMARY) as fh:
        payload = json.load(fh)
    assert payload["doc_version"] == (
        "batch_inference_frontier_incremental_alpha_audit_v1")
    for k in REQUIRED_DO_NOT_FLAGS:
        assert k in payload, f"missing flag {k}"
        assert payload[k] is False, f"{k} must be false; got {payload[k]!r}"
    assert "alpha_decomposition_pct" in payload
    assert "acceptance_gate" in payload
    assert "policy_kpi_table" in payload
    # Alpha decomposition required keys
    ad = payload["alpha_decomposition_pct"]
    for k in ("duplicated_serving_frontier_alpha_pct",
              "deadline_flex_scenario_alpha_pct",
              "true_incremental_alpha_vs_dynamic_serving_pct"):
        assert k in ad, f"missing alpha key {k}"
    # Acceptance gate required keys
    ag = payload["acceptance_gate"]
    assert ag["incremental_alpha_gate_pct"] == 2.0
    assert "verdict" in ag
    assert "alpha_gate_passed" in ag


@pytest.mark.skipif(not SUMMARY.exists(),
                    reason="audit summary not committed yet")
def test_audit_verdict_is_shadow_diagnostic_on_committed_fixture():
    """Pre-registered: the audit on the committed Azure 2024 fixture
    at 100x scale yields SHADOW_DIAGNOSTIC. A regression to
    PROPOSE_INTEGRATION fires this test loudly — flipping requires a
    deliberate review."""
    with open(SUMMARY) as fh:
        payload = json.load(fh)
    src = payload["source"]
    # Sanity: the committed summary was produced from the committed
    # fixture at the audit's primary scale.
    assert src["scale_rps"] == 100.0
    assert src["tick_seconds"] == 60.0
    assert "azure_llm_2024_sample.csv" in src["path"]
    verdict = payload["acceptance_gate"]["verdict"]
    assert verdict == "SHADOW_DIAGNOSTIC", (
        f"audit verdict changed to {verdict!r}; this requires a "
        f"deliberate review and a docs/BATCH_INFERENCE_FRONTIER_INCREMENTAL"
        f"_ALPHA_AUDIT.md update. See test docstring.")


@pytest.mark.skipif(not SUMMARY.exists(),
                    reason="audit summary not committed yet")
def test_incremental_alpha_below_gate():
    with open(SUMMARY) as fh:
        payload = json.load(fh)
    inc = payload["alpha_decomposition_pct"][
        "true_incremental_alpha_vs_dynamic_serving_pct"]
    gate = payload["acceptance_gate"]["incremental_alpha_gate_pct"]
    # The pinned-fixture verdict requires this be <= 2.0; if the
    # estimator improves enough to flip the verdict, this test fires.
    assert inc is not None
    assert inc <= gate, (
        f"incremental alpha {inc:+.4f}% > gate {gate}%; this should "
        f"flip the verdict to PROPOSE_INTEGRATION — update the audit "
        f"doc + the pinned verdict test together")


def test_audit_script_imports_dynamic_batch_estimator():
    assert SCRIPT.exists()
    src = SCRIPT.read_text(encoding="utf-8")
    assert "dynamic_batch_inference_estimator" in src
    assert "estimate_dynamic_batch_frontier" in src
    assert "choose_dynamic_batch_decision" in src
    # Compares against the existing dynamic serving frontier — required.
    assert "estimate_dynamic_frontier" in src
    assert "choose_dynamic_rho" in src
    # Same KPI: imports compute_economic_kpi.
    assert "compute_economic_kpi" in src


def test_audit_script_does_not_modify_scheduler():
    src = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "from aurelius.constraints import",
        "from aurelius.optimization import",
        "from aurelius.constraints.frontier_integration import",
        "constraint_aware_default_rho =",
        "register_with_scheduler",
    )
    for f in forbidden:
        assert f not in src, (
            f"audit script must not modify scheduler/constraint paths "
            f"(found `{f}`)")


def test_audit_doc_committed_and_references_prior_art():
    assert DOC.exists(), f"missing {DOC}"
    text = DOC.read_text(encoding="utf-8")
    missing = [d for d in REQUIRED_DOC_REFS if d not in text]
    assert not missing, f"audit doc missing prior-art refs: {missing}"
    # Verdict appears.
    assert "SHADOW_DIAGNOSTIC" in text
    # Gate pre-registered.
    assert "+2.0 %" in text or "+2.0%" in text or "> +2.0" in text
    # Honesty notes
    assert "NOT production savings" in re.sub(r"[\s>]+", " ", text)


def test_audit_doc_no_unhedged_banned_phrases():
    text = DOC.read_text(encoding="utf-8").lower()
    for phrase in BANNED:
        for line in text.splitlines():
            if phrase not in line:
                continue
            if any(hedge in line for hedge in (
                "not ", "no ", "never", "do not", "must not", "n't",
            )):
                continue
            pytest.fail(
                f"unhedged banned phrase {phrase!r} in audit doc line: "
                f"{line!r}")


def test_audit_doc_pinned_kpi_is_unchanged_canonical():
    # The audit cannot silently swap to a different KPI.
    text = DOC.read_text(encoding="utf-8")
    assert "sla_safe_goodput_per_infrastructure_dollar" in text
