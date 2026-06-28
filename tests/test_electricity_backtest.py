"""Checkpointed electricity backtest runner tests (focused — isolation, lifts, timeout/failed/resume)."""

from __future__ import annotations

import time

from scripts import run_checkpointed_electricity_backtest as bt


# --- arm isolation -----------------------------------------------------------
def test_arm_isolation_sets():
    # DVFS-only is price-aware but NOT deferrable; deferrable-only is deferrable but NOT price-aware
    assert "real_price_dvfs_only" in bt._PRICE_AWARE_ARMS
    assert "real_price_dvfs_only" not in bt._DEFERRABLE_ARMS
    assert "real_price_deferrable_only" in bt._DEFERRABLE_ARMS
    assert "real_price_deferrable_only" not in bt._PRICE_AWARE_ARMS
    # flat / real "no actions" arms are neither
    for a in ("current_main_mpc_flat_price", "current_main_mpc_real_price", "baseline_sla_aware"):
        assert a not in bt._PRICE_AWARE_ARMS and a not in bt._DEFERRABLE_ARMS


# --- lift + interaction attribution ------------------------------------------
def _state(real, dvfs, defr, both, base, sla_both=0.01, sla_base=0.02):
    def cell(gp, sla=0.01, defer=None):
        r = {"gp_per_dollar": gp, "sla_violation_rate": sla}
        if defer is not None:
            r["deferrable"] = defer
        return {"status": "COMPLETED", "result": r}
    return {"cells": {
        "pjm|expensive|baseline_sla_aware": cell(base, sla_base),
        "pjm|expensive|current_main_mpc_real_price": cell(real),
        "pjm|expensive|real_price_dvfs_only": cell(dvfs),
        "pjm|expensive|real_price_deferrable_only": cell(defr),
        "pjm|expensive|real_price_dvfs_plus_deferrable": cell(both, sla_both,
            defer={"shifting_saving": 0.4, "deadlines_respected": True}),
    }}


def test_summarize_lifts_and_interaction():
    s = bt.summarize(_state(real=100.0, dvfs=110.0, defr=104.0, both=116.0, base=80.0))["pjm|expensive"]
    assert s["dvfs_lift_gp$"] == 10.0                       # 110 − 100
    assert s["deferrable_serving_gp$_delta"] == 4.0          # 104 − 100
    assert s["combined_lift_gp$"] == 16.0                    # 116 − 100
    assert s["interaction_gp$"] == 2.0                       # 116 − 110 − 104 + 100
    assert s["all_elec_vs_baseline_pct"] == 45.0             # (116−80)/80
    assert s["headline_safe"] is True                        # 116>80 and SLA 0.01 ≤ 0.02
    assert s["deferrable_shifting_saving_$"] == 0.4 and s["deferrable_deadlines_respected"] is True


def test_summarize_pareto_blocks_sla_shedding():
    # the best arm beats baseline gp/$ but has WORSE SLA → NOT headline-safe
    s = bt.summarize(_state(real=100.0, dvfs=110.0, defr=104.0, both=120.0, base=80.0,
                            sla_both=0.09, sla_base=0.02))["pjm|expensive"]
    assert s["pareto_beats_baseline"] is True and s["pareto_sla_not_worse"] is False
    assert s["headline_safe"] is False                       # SLA-shedding blocked


def test_summarize_excludes_non_completed():
    st = _state(real=100.0, dvfs=110.0, defr=104.0, both=116.0, base=80.0)
    st["cells"]["pjm|expensive|real_price_dvfs_only"]["status"] = "TIMEOUT"   # drop one arm
    s = bt.summarize(st)["pjm|expensive"]
    assert "dvfs_lift_gp$" not in s                          # dvfs arm not COMPLETED → no dvfs lift
    assert "combined_lift_gp$" not in s                      # interaction needs all arms


# --- cell isolation: FAILED + TIMEOUT never crash the run --------------------
def test_failed_cell_is_recorded_not_raised():
    # market absent from _CTX → the worker raises KeyError → recorded FAILED (run continues)
    status, result, secs = bt.run_cell("no_such_market", [0, 1], "current_main_mpc_real_price",
                                       max_decisions=1, timeout=30)
    assert status == "FAILED" and "error" in result and secs >= 0


def test_timeout_cell_is_recorded_not_hung(monkeypatch):
    # a slow cell is killed at the hard timeout and recorded TIMEOUT (fork inherits the patched fn)
    def _slow(*a, **k):
        time.sleep(30)
    monkeypatch.setattr(bt, "evaluate_cell", _slow)
    bt._CTX["pjm"] = {"sentinel": True}
    t0 = time.monotonic()
    status, result, secs = bt.run_cell("pjm", [0, 1], "current_main_mpc_real_price",
                                       max_decisions=1, timeout=2)
    assert status == "TIMEOUT" and result is None
    assert time.monotonic() - t0 < 10                       # killed promptly, did NOT wait 30s


# --- resume skips completed/timeout cells ------------------------------------
def test_resume_skip_logic():
    state = {"cells": {"pjm|expensive|baseline_sla_aware": {"status": "COMPLETED"},
                       "pjm|expensive|real_price_dvfs_only": {"status": "TIMEOUT"},
                       "pjm|expensive|real_price_deferrable_only": {"status": "FAILED"}}}
    # COMPLETED and TIMEOUT are skipped on resume; FAILED is retried (status not in the skip set)
    def skip(key):
        return state["cells"].get(key, {}).get("status") in ("COMPLETED", "TIMEOUT")
    assert skip("pjm|expensive|baseline_sla_aware")
    assert skip("pjm|expensive|real_price_dvfs_only")
    assert not skip("pjm|expensive|real_price_deferrable_only")   # FAILED → retried
    assert not skip("pjm|expensive|current_main_mpc_real_price")  # never run → run it


def test_select_windows_quick_is_single():
    prices = {p: (0.02 if p % 4 == 0 else 0.25) for p in range(120)}
    wq = bt.select_windows(prices, 120, win_len=4, quick=True)
    assert set(wq) == {"expensive"} and len(wq["expensive"]) == 4
    wf = bt.select_windows(prices, 120, win_len=4, quick=False)
    assert {"cheap", "volatile", "expensive"} <= set(wf)
