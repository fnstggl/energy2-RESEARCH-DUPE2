"""Diagnostic-only tests for the PR #123/#124 variable-isolation reconciliation matrix.

Pins the pure helpers and the scientific conclusions the matrix establishes, read from the committed artifact
(no market rebuild): the evaluation harness is the dominant variable (≫ req_cap, electricity), hierarchical is
Pareto-safe in every cell, and the cumulative bridge lands at the PR #124 figure. Artifact-dependent tests skip
if the JSON is absent.
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.diagnose_reconciliation_matrix import _cell, _pct

_ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "external", "mpc_controller", "reconciliation_matrix.json")


def test_pct_and_cell_pure():
    assert _pct(200.0, 100.0) == 100.0 and _pct(1.0, 0.0) is None
    rows = {"hierarchical": {"gp_per_dollar": 300.0, "sla_violation_rate": 0.0},
            "sla_aware": {"gp_per_dollar": 100.0, "sla_violation_rate": 0.2}}
    c = _cell("X", rows, "desc")
    assert c["hier_vs_sla_aware_pct"] == 200.0 and c["hier_vs_sla_aware_abs"] == 200.0
    assert c["hier_sla_not_worse"] is True


def _load():
    if not os.path.exists(_ARTIFACT):
        pytest.skip("reconciliation_matrix.json not present (run scripts.diagnose_reconciliation_matrix)")
    return json.load(open(_ARTIFACT))


def test_harness_is_the_dominant_variable():
    m = _load()["marginal_effects_on_hier_vs_sla_pct_from_PR123"]
    harness = abs(m["harness_A_to_B (cap80, 1 period)"])
    reqcap = abs(m["request_cap_80_to_56 (Harness A)"])
    elec = abs(m["electricity_const_to_real (Harness A)"])
    # the harness swings the headline by far more than req_cap or electricity (≥10× each).
    assert harness >= 10 * reqcap
    assert harness >= 10 * elec


def test_e0_reproduces_pr123_and_e5_reproduces_pr124():
    cells = _load()["cells"]
    assert cells["E0_PR123_exact"]["hier_vs_sla_aware_pct"] > 1000          # the +1273% regime
    assert 120 < cells["E5_PR124_exact"]["hier_vs_sla_aware_pct"] < 220     # the +164% regime


def test_hierarchical_pareto_safe_in_every_cell():
    for k, c in _load()["cells"].items():
        assert c["hier_vs_sla_aware_pct"] > 0, k
        assert c["hier_sla_not_worse"], k


def test_cumulative_bridge_lands_at_pr124():
    bridge = _load()["cumulative_bridge_PR123_to_PR124"]
    assert bridge[0][1] > 1000                       # starts at PR123 (+1273%)
    assert 120 < bridge[-1][1] < 220                 # ends at PR124 (+164%)
    # the single biggest drop is the first step (the harness change).
    drops = [bridge[i - 1][1] - bridge[i][1] for i in range(1, len(bridge))]
    assert drops[0] == max(drops) and drops[0] > 500
