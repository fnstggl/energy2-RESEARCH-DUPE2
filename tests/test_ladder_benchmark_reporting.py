"""The ladder benchmark runner's REPORTING contract (Phase E) — pure-function tests (no heavy compute).

These pin: the 8 arms are the right set with production_scheduler as the headline bar and oracle as a separate
diagnostic arm; every gp/$ comparison reports BOTH an absolute and a percent delta; the headline defaults to
production_scheduler (not fifo, not oracle); and the Pareto SLA clause is carried alongside each delta. The
`summarize` function is pure (it consumes already-computed cells), so this needs no market build.
"""

from __future__ import annotations

from scripts.run_ladder_benchmark import ARMS, summarize


def _cell(gp, sla):
    return {"status": "COMPLETED", "result": {"gp_per_dollar": gp, "sla_violation_rate": sla}}


def _state():
    # one window, all 8 arms present, with the headline optimiser beating both bars on gp/$ at lower SLA.
    arms = {
        "fifo": (180000.0, 0.46), "vllm_only": (411000.0, 0.10), "topology_aware": (181000.0, 0.46),
        "sla_aware": (295000.0, 0.23), "production_scheduler": (330000.0, 0.066),
        "aurelius_mpc_current_default": (619000.0, 0.02),
        "aurelius_mpc_hierarchical_search": (715000.0, 0.0), "oracle_diagnostic": (900000.0, 0.0),
    }
    return {"cells": {f"pjm|expensive|{a}": _cell(gp, sla) for a, (gp, sla) in arms.items()}}


def test_arms_set_and_order():
    assert ARMS == ("fifo", "vllm_only", "topology_aware", "sla_aware", "production_scheduler",
                    "aurelius_mpc_current_default", "aurelius_mpc_hierarchical_search", "oracle_diagnostic")
    # production_scheduler and the two MPC arms and the oracle are all SEPARATE arms.
    assert "production_scheduler" in ARMS
    assert "aurelius_mpc_hierarchical_search" in ARMS and "aurelius_mpc_current_default" in ARMS
    assert "oracle_diagnostic" in ARMS


def test_summary_reports_abs_and_pct_vs_production_scheduler():
    s = summarize(_state())["pjm|expensive"]
    h = s["aurelius_mpc_hierarchical_search"]["vs_production_scheduler"]
    assert h["abs_delta"] == 715000.0 - 330000.0            # absolute delta present
    assert h["pct_delta"] is not None                        # AND percent delta present
    assert abs(h["pct_delta"] - 100.0 * (715000.0 - 330000.0) / 330000.0) < 1e-2   # (reported rounded to 3dp)
    assert h["sla_not_worse"] is True                        # Pareto clause carried (0.0 <= 0.066)


def test_headline_default_is_production_scheduler_not_fifo_or_oracle():
    s = summarize(_state())["pjm|expensive"]
    # the optimiser's headline entry compares against production_scheduler (and sla_aware), never fifo.
    entry = s["aurelius_mpc_hierarchical_search"]
    assert "vs_production_scheduler" in entry
    assert "vs_sla_aware" in entry
    assert "vs_fifo" not in entry                            # fifo is NOT a headline comparator


def test_oracle_is_diagnostic_gap_only():
    s = summarize(_state())["pjm|expensive"]
    entry = s["aurelius_mpc_hierarchical_search"]
    # the oracle appears ONLY as a non-deployable upper-bound gap, never as the headline bar.
    assert "oracle_gap" in entry
    assert entry["oracle_gap"]["abs"] == 900000.0 - 715000.0
    # and gp_per_dollar carries every arm (oracle present as a row, not as the comparison target).
    assert s["gp_per_dollar"]["oracle_diagnostic"] == 900000.0


def test_secondary_bar_vs_sla_aware_has_abs_and_pct():
    s = summarize(_state())["pjm|expensive"]
    v = s["aurelius_mpc_hierarchical_search"]["vs_sla_aware"]
    assert v["abs_delta"] == 715000.0 - 295000.0 and v["pct_delta"] is not None


def test_current_default_arm_also_compared():
    s = summarize(_state())["pjm|expensive"]
    # both optimiser arms (current default AND hierarchical) get the full comparison set.
    cd = s["aurelius_mpc_current_default"]
    assert cd["vs_production_scheduler"]["abs_delta"] == 619000.0 - 330000.0
    assert cd["vs_production_scheduler"]["pct_delta"] is not None
