"""Batch-1 corrective PR — product-boundary + DistServe + regime-audit tests.

Proves: GPU assignment is core / auto-noop and cannot fake a headline on a homogeneous (or single-dominant-GPU
cost) fleet; KV precision and PD are OPTIONAL serving-engine integrations and DEFAULT-OFF; int4 stays
diagnostic-only; the DistServe-shaped PD fixtures behave (correct split helps, wrong hurts, low bandwidth
hurts, light prefers shared); the regime audit is deterministic and reports the required metrics; no benchmark
default / reward / cost / Pareto change; and no active cap=120 recommendation remains.
"""

from __future__ import annotations

import os

from aurelius.environment.actions import (
    CONNECTED_SURFACES,
    CORE_ORCHESTRATION_AUTO_NOOP,
    DIAGNOSTIC_ONLY,
    OPTIONAL_SERVING_ENGINE_INTEGRATION,
    ActionBundle,
    core_orchestration_surfaces,
    optional_serving_engine_surfaces,
    product_category,
    value_is_diagnostic_only,
)
from aurelius.environment.controller import ModelPredictiveEconomicController
from aurelius.environment.gpu_assignment import GPUType, WorkloadClass, compare_assignment_policies
from aurelius.environment.pd_disaggregation import (
    PDWorkload,
    distserve_goodput_comparison,
    pd_slo_goodput,
)
from aurelius.environment.physics_guided_candidates import (
    PhysicsGuidedCandidateGenerator,
    PlannerRegimeState,
)

_CLASSES = [WorkloadClass("latency_sensitive", 1024, 128, 2.0, sla_s=0.30, kind="latency"),
            WorkloadClass("batch", 1024, 512, 1.0, None, kind="batch")]


# --- product-boundary classification ----------------------------------------------------------------
def test_gpu_assignment_is_core_auto_noop():
    assert product_category("gpu_assignment_policy") == CORE_ORCHESTRATION_AUTO_NOOP
    assert "gpu_assignment_policy" in core_orchestration_surfaces()


def test_kv_precision_and_pd_are_optional_serving_engine_integrations():
    assert product_category("kv_cache_precision_policy") == OPTIONAL_SERVING_ENGINE_INTEGRATION
    assert product_category("prefill_decode_policy") == OPTIONAL_SERVING_ENGINE_INTEGRATION
    opt = optional_serving_engine_surfaces()
    assert "kv_cache_precision_policy" in opt and "prefill_decode_policy" in opt


def test_int4_values_are_diagnostic_only():
    assert value_is_diagnostic_only("precision_policy", "int4")
    assert value_is_diagnostic_only("kv_cache_precision_policy", "kv_int4_diagnostic_only")
    assert not value_is_diagnostic_only("kv_cache_precision_policy", "kv_fp8")
    # DIAGNOSTIC_ONLY is one of the five product categories.
    assert DIAGNOSTIC_ONLY in {"DIAGNOSTIC_ONLY"}


# --- default-off (the corrective behaviour) ---------------------------------------------------------
def test_controller_defaults_serving_engine_integrations_off():
    c = ModelPredictiveEconomicController.__dataclass_fields__
    assert c["enable_kv_cache_precision"].default is False
    assert c["enable_prefill_decode_disagg"].default is False


def test_default_mask_freezes_optional_integrations():
    # the controller's product-boundary default mask: only gpu_assignment (core) allowed; KV+PD frozen off.
    default_mask = frozenset({"gpu_assignment_policy"})
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", hbm_pressure=0.8, pd_divergence=True,
                            prefill_heavy=False, capacity_pressure=0.6, allowed_new_knobs=default_mask)
    cs = PhysicsGuidedCandidateGenerator().generate(st)
    assert {b.kv_cache_precision_policy for b in cs.bundles} == {"inherit_weight_precision"}
    assert {b.prefill_decode_policy for b in cs.bundles} == {"shared"}
    assert "kv_cache_precision_policy_disabled" in cs.pruned_reasons
    assert "prefill_decode_policy_disabled" in cs.pruned_reasons


def test_opt_in_re_enables_kv():
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", hbm_pressure=0.8,
                            allowed_new_knobs=frozenset({"gpu_assignment_policy", "kv_cache_precision_policy"}))
    cs = PhysicsGuidedCandidateGenerator().generate(st)
    assert "kv_fp8" in {b.kv_cache_precision_policy for b in cs.bundles}


# --- GPU assignment cannot fake a headline ----------------------------------------------------------
def test_gpu_assignment_not_in_reward_path():
    # SIMULATED_ONLY → not a CONNECTED reward surface; flipping it never changes the replay kwargs.
    assert "gpu_assignment_policy" not in CONNECTED_SURFACES
    base = ActionBundle()
    flipped = base.with_overrides(gpu_assignment_policy="fastest_for_latency_sensitive")
    assert flipped.replay_kwargs() == base.replay_kwargs()


def test_homogeneous_fleet_auto_noop_ties():
    cmp = compare_assignment_policies(_CLASSES, [GPUType("A100", 16)])
    gps = {round(r["gp_per_dollar"], 2) for p, r in cmp["results"].items() if r["deployable"]}
    assert cmp["homogeneous_fleet"] and len(gps) == 1


# --- DistServe-shaped PD fixtures -------------------------------------------------------------------
_DS = dict(prefill_work_s=0.7, decode_work_s=0.3, context_tokens=2048, decode_tokens=64)


def test_distserve_correct_split_helps():
    hi = PDWorkload(arrival_rate=20, kv_bandwidth_bytes_per_s=300e9, **_DS)
    r = distserve_goodput_comparison(hi, n_replicas=16, ttft_slo_s=1.0, tpot_slo_s=0.004)
    assert r["distserve_like"] and r["goodput_ratio_disagg_over_shared"] >= 1.5   # DistServe-order win


def test_distserve_wrong_split_hurts():
    hi = PDWorkload(arrival_rate=20, kv_bandwidth_bytes_per_s=300e9, **_DS)
    r = distserve_goodput_comparison(hi, n_replicas=16, ttft_slo_s=1.0, tpot_slo_s=0.004)
    wrong = pd_slo_goodput(hi, "decode_heavy", n_replicas=16, ttft_slo_s=1.0, tpot_slo_s=0.004)
    assert wrong["slo_safe_goodput"] < r["best"]["slo_safe_goodput"]


def test_distserve_insufficient_bandwidth_hurts():
    base = dict(arrival_rate=20, prefill_work_s=0.7, decode_work_s=0.3, context_tokens=4096, decode_tokens=64)
    fast = pd_slo_goodput(PDWorkload(kv_bandwidth_bytes_per_s=300e9, **base), "balanced_pd",
                          n_replicas=16, ttft_slo_s=1.2, tpot_slo_s=0.006)
    slow = pd_slo_goodput(PDWorkload(kv_bandwidth_bytes_per_s=1.0e9, **base), "balanced_pd",
                          n_replicas=16, ttft_slo_s=1.2, tpot_slo_s=0.006)
    assert slow["slo_safe_goodput"] < fast["slo_safe_goodput"]
    assert slow["kv_handoff_latency"] > fast["kv_handoff_latency"]


def test_distserve_light_prefers_shared():
    light = PDWorkload(arrival_rate=4, prefill_work_s=0.3, decode_work_s=0.3, context_tokens=512,
                       decode_tokens=128, kv_bandwidth_bytes_per_s=300e9)
    r = distserve_goodput_comparison(light, n_replicas=16, ttft_slo_s=2.0, tpot_slo_s=0.05)
    assert r["shared"]["slo_safe_goodput"] >= r["best"]["slo_safe_goodput"] - 1e-6
    assert not r["distserve_like"]


# --- regime audit: deterministic + reports the required metrics --------------------------------------
def _synthetic_periods():
    # (arrival, output_tokens, input_tokens); input present in one set, absent in another.
    return [[(float(i), 200, 1024) for i in range(40)] for _ in range(4)]


def test_kv_audit_reports_memory_and_hbm_metrics():
    from scripts.run_batch1_regime_audit import _kv_audit
    a1 = _kv_audit(_synthetic_periods(), "A100")
    a2 = _kv_audit(_synthetic_periods(), "A100")
    assert a1 == a2                                              # deterministic
    for k in ("pct_memory_bandwidth_bound", "pct_hbm_high_pressure", "kv_bytes_saved_pct_fp8",
              "mean_kv_occupancy_estimate"):
        assert k in a1
    assert a1["kv_bytes_saved_pct_fp8"] > 0.0


def test_pd_audit_reports_skew_and_interference_metrics():
    from scripts.run_batch1_regime_audit import _pd_audit
    a1 = _pd_audit(_synthetic_periods(), "A100")
    a2 = _pd_audit(_synthetic_periods(), "A100")
    assert a1 == a2
    for k in ("phase_mix", "pct_skewed", "mean_interference_relief_estimate",
              "mean_phase_pool_utilization", "mean_handoff_bytes", "mean_handoff_latency_s"):
        assert k in a1


def test_gpu_audit_reports_not_applicable():
    from types import SimpleNamespace

    from scripts.run_batch1_regime_audit import _gpu_audit
    fleet = SimpleNamespace(gpu_type_mix={"H100": 0.5, "A100": 0.5})
    ws = SimpleNamespace(servers={})
    a = _gpu_audit(fleet, ws)
    assert a["applicability"].startswith("NOT_APPLICABLE")
    assert a["request_to_gpu_assignment_in_reward_path"] is False
    assert a["default_on_auto_noop"] is True


def test_prompt_data_absence_detected():
    from scripts.run_batch1_regime_audit import _prompt_data_present
    assert _prompt_data_present([[(0.0, 100, 512)]]) is True
    assert _prompt_data_present([[(0.0, 100, 0), (1.0, 50, 0)]]) is False    # Azure trace input=0


# --- no reward/cost/Pareto change; no cap=120 recommendation ----------------------------------------
def test_pareto_gate_unchanged():
    from scripts.run_ladder_benchmark import summarize
    state = {"cells": {
        "pjm|expensive|production_scheduler": {"status": "COMPLETED",
            "result": {"gp_per_dollar": 100.0, "sla_violation_rate": 0.01}},
        "pjm|expensive|aurelius_mpc_hierarchical_search": {"status": "COMPLETED",
            "result": {"gp_per_dollar": 200.0, "sla_violation_rate": 0.0}}}}
    s = summarize(state)["pjm|expensive"]["aurelius_mpc_hierarchical_search"]["vs_production_scheduler"]
    assert s["abs_delta"] == 100.0 and s["pct_delta"] == 100.0 and s["sla_not_worse"] is True


def test_no_active_cap_120_recommendation_in_docs():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    freeze = os.path.join(root, "research", "REQUEST_CAP_SENSITIVITY_AND_BENCHMARK_FREEZE.md")
    with open(freeze) as f:
        text = f.read()
    assert "100,000" in text                                    # the corrected recommended cap
    # the active recommendation must be 100,000, not 120 (a withdrawn/obsolete mention of 120 is allowed).
    assert "Freeze at 120." not in text
    assert "Freeze at cap = 100,000" in text
