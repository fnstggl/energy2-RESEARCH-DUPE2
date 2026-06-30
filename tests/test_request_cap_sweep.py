"""Minimal diagnostic tests for the Benchmark V1 request-cap sweep helpers + artifact invariants.

Pure-helper tests always run; artifact-dependent tests skip if the sweep JSON is absent (it is produced by the
long `scripts.run_request_cap_sweep` run).
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.run_request_cap_sweep import _actual_and_binding, _capped_per, _pct

_ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "external", "mpc_controller", "request_cap_sweep.json")


def test_capped_per_slices_per_period():
    full = {1: list(range(200)), 2: list(range(150))}
    assert {p: len(v) for p, v in _capped_per(full, 80).items()} == {1: 80, 2: 80}
    assert {p: len(v) for p, v in _capped_per(full, None).items()} == {1: 200, 2: 150}   # None = uncapped


def test_actual_and_binding():
    fc = {1: 200, 2: 150, 3: 300}
    assert _actual_and_binding(fc, [1, 2, 3], 80) == (240, True)        # 80×3, binds (every period > 80)
    assert _actual_and_binding(fc, [1, 2, 3], None) == (650, False)     # uncapped, never binds
    assert _actual_and_binding(fc, [1, 2, 3], 1000) == (650, False)     # cap above all counts → not binding


def test_pct():
    assert _pct(200.0, 100.0) == 100.0 and _pct(1.0, 0.0) is None


def _load():
    if not os.path.exists(_ARTIFACT):
        pytest.skip("request_cap_sweep.json not present (run scripts.run_request_cap_sweep)")
    d = json.load(open(_ARTIFACT))
    if "recommended_benchmark_cap" not in d:
        pytest.skip("request_cap_sweep.json is mid-run (no recommendation yet)")
    return d


def test_recommended_cap_is_set():
    d = _load()
    assert "recommended_benchmark_cap" in d
    # it is either "uncapped" or a positive integer request cap.
    rec = d["recommended_benchmark_cap"]
    assert rec == "uncapped" or (isinstance(rec, int) and rec > 0)


def test_cap_chosen_is_a_completing_cap_for_all_arms():
    """The recommended cap (when numeric) must be one where every required arm COMPLETED — never chosen on a
    timed-out cell."""
    d = _load()
    rec = d["recommended_benchmark_cap"]
    if not isinstance(rec, int):
        return
    # for at least one market, all three arms completed at the recommended cap.
    cells = d["cells"]
    markets = {k.split("|")[0] for k in cells}
    ok_market = any(all(cells.get(f"{m}|{rec}|{a}", {}).get("status") == "COMPLETED"
                        for a in ("sla_aware", "production_scheduler", "aurelius_mpc_hierarchical_search"))
                    for m in markets)
    assert ok_market
