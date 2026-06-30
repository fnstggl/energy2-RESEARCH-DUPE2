"""Diagnostic-only tests for the PR #123 vs PR #124 headline reconciliation.

Pins the pure logic (`_bundle_from_action`, `_pct`) and the conclusions the reconciliation rests on, read from
the committed artifact (no market rebuild): hierarchical did NOT regress (its absolute gp/$ is higher in the
single-decision harness than the episode harness), it is Pareto-safe vs BOTH baselines in BOTH harnesses, and
the +1273%/+165% gap is the numerator×denominator harness decomposition. If the artifact is absent the
artifact-dependent tests skip (the script regenerates it).
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.diagnose_headline_reconciliation import _bundle_from_action, _pct

_ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "external", "mpc_controller", "headline_reconciliation.json")


def test_bundle_from_action_maps_connected_surfaces():
    ab = _bundle_from_action({"capacity": "backlog_aware", "ordering": "abs_conformal",
                              "admission": "class_aware", "capacity_multiplier": 1.25,
                              "batching_policy": "balanced", "routing_policy": "kv_aware",
                              "placement_policy": "rack_local", "precision_policy": "bf16",
                              "clock_policy": "base", "migration_policy": "off", "spec_decode_policy": "off"})
    assert ab.capacity_policy == "backlog_aware" and ab.ordering_policy == "abs_conformal"
    assert ab.admission_policy == "class_aware" and ab.capacity_multiplier == 1.25
    assert ab.routing_policy == "kv_aware" and ab.placement_policy == "rack_local"
    assert ab.precision_policy == "bf16" and ab.clock_policy == "base"   # production: no economic arbitrage


def test_bundle_from_action_defaults_are_noop():
    ab = _bundle_from_action({})
    assert ab.capacity_policy == "reactive_lag1" and ab.precision_policy == "bf16" and ab.clock_policy == "base"


def test_pct():
    assert _pct(200.0, 100.0) == 100.0
    assert _pct(105924.1 * 13.733, 105924.1) == pytest.approx(1273.3, abs=1.0)
    assert _pct(1.0, 0.0) is None


def _load():
    if not os.path.exists(_ARTIFACT):
        pytest.skip("headline_reconciliation.json not present (run scripts.diagnose_headline_reconciliation)")
    return json.load(open(_ARTIFACT))


def test_hierarchical_did_not_regress_absolute_gp_higher_in_setup_a():
    d = _load()
    ha = d["setup_a_pr123_tournament"]["rows"]["aurelius_mpc_hierarchical_search"]["gp_per_dollar"]
    hb = d["setup_b_pr124_ladder"]["rows"]["aurelius_mpc_hierarchical_search"]["gp_per_dollar"]
    # the SAME method reads HIGHER in the single-decision harness — not a regression, a harness effect.
    assert ha > hb


def test_hierarchical_pareto_safe_vs_both_baselines_in_both_harnesses():
    d = _load()
    for setup in ("setup_a_pr123_tournament", "setup_b_pr124_ladder"):
        h = d[setup]["rows"]["aurelius_mpc_hierarchical_search"]
        assert h["vs_sla_aware"]["pct"] > 0 and h["vs_sla_aware"]["sla_not_worse"]
        if "vs_production_scheduler" in h:
            assert h["vs_production_scheduler"]["pct"] > 0 and h["vs_production_scheduler"]["sla_not_worse"]


def test_gap_is_harness_decomposition():
    d = _load()
    dec = d["decomposition"]
    # ratio_of_ratios == numerator_ratio × denominator_ratio (the gap is a measurement decomposition).
    assert dec["ratio_of_ratios"] == pytest.approx(
        dec["numerator_ratio_A_over_B"] * dec["denominator_ratio_B_over_A"], rel=0.02)
    # and Setup A's percent is much larger than Setup B's (the thing being explained).
    assert dec["setup_a_pct_vs_sla_aware"] > 5 * dec["setup_b_pct_vs_sla_aware"]


def test_production_scheduler_present_in_pr123_harness():
    d = _load()
    # task 2: production_scheduler added as an arm to the PR #123 setup, and hierarchical beats it there too.
    a = d["setup_a_pr123_tournament"]["rows"]
    assert "production_scheduler" in a
    assert a["aurelius_mpc_hierarchical_search"]["vs_production_scheduler"]["pct"] > 0
