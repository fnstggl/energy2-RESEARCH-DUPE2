"""Batch-1 new action knobs — KV-cache precision, heterogeneous GPU assignment, prefill/decode disaggregation.

Covers the Phase-9 contract: the surfaces exist; defaults are bit-for-bit no-ops; each knob reaches reward
ONLY through physical channels (latency / GPU-seconds / memory / SLA), never a bonus; each can help AND hurt;
unsafe precision is excluded from the headline; heterogeneous assignment fakes no gain on a homogeneous fleet;
PD disaggregation pays a handoff cost; the planner generates the knobs only in their regimes; the ablation
deltas are absolute + percent; the Pareto gate is unchanged; replay is deterministic.
"""

from __future__ import annotations

import math

from aurelius.environment.action_registry import validate_action_bundle
from aurelius.environment.actions import (
    ACTION_SPECS,
    CONNECTED,
    CONNECTED_SURFACES,
    SIMULATED_ONLY,
    ActionBundle,
)
from aurelius.environment.gpu_assignment import (
    GPUType,
    WorkloadClass,
    compare_assignment_policies,
)
from aurelius.environment.kv_precision import (
    is_headline_safe_kv,
    kv_precision_memory_effect,
)
from aurelius.environment.pd_disaggregation import PDWorkload, pd_serving_point
from aurelius.environment.physics_guided_candidates import (
    PhysicsGuidedCandidateGenerator,
    PlannerRegimeState,
)
from aurelius.environment.roofline import ServingConfig, Workload, serving_point
from aurelius.environment.roofline_actions import (
    is_neutral_roofline_bundle,
    period_action_modulation,
    roofline_action_factors,
)

_RECS = [(float(i), 256, 512) for i in range(20)]
_WL_MEM = Workload(prompt_tokens=512, decode_tokens=512, context_len=4096)   # memory-bandwidth-bound decode
_WL_COMP = Workload(prompt_tokens=32, decode_tokens=4, context_len=32)        # compute-bound (tiny)


# --- surfaces exist + statuses ----------------------------------------------------------------------
def test_new_action_surfaces_exist():
    assert "kv_cache_precision_policy" in ACTION_SPECS
    assert "prefill_decode_policy" in ACTION_SPECS
    assert "gpu_assignment_policy" in ACTION_SPECS
    assert ACTION_SPECS["kv_cache_precision_policy"].status == CONNECTED
    assert ACTION_SPECS["prefill_decode_policy"].status == CONNECTED
    # heterogeneous assignment is NOT_APPLICABLE to the production cost path → SIMULATED_ONLY (fixture-only).
    assert ACTION_SPECS["gpu_assignment_policy"].status == SIMULATED_ONLY
    assert "kv_cache_precision_policy" in CONNECTED_SURFACES
    assert "prefill_decode_policy" in CONNECTED_SURFACES
    assert "gpu_assignment_policy" not in CONNECTED_SURFACES


def test_defaults_are_noops():
    b = ActionBundle()
    assert b.kv_cache_precision_policy == "inherit_weight_precision"
    assert b.prefill_decode_policy == "shared"
    assert b.gpu_assignment_policy == "homogeneous_default"
    assert validate_action_bundle(b)["ok"]
    # the default bundle is neutral → period modulation is skipped entirely (bit-for-bit unchanged).
    assert is_neutral_roofline_bundle(b)
    assert period_action_modulation(b, _RECS) is None


def test_options_are_valid_and_first_is_noop():
    for name in ("kv_cache_precision_policy", "prefill_decode_policy", "gpu_assignment_policy"):
        spec = ACTION_SPECS[name]
        assert spec.default == getattr(ActionBundle(), name)         # options[0] == the bundle default


# --- KV-cache precision: causal channel only --------------------------------------------------------
def test_kv_precision_separate_from_weights_changes_kv_bytes():
    # weights bf16, KV fp8 → KV bytes/token halve while weight precision is unchanged.
    base = serving_point(_WL_MEM, ServingConfig(gpu="A100", precision="bf16", kv_precision="inherit"))
    kv8 = serving_point(_WL_MEM, ServingConfig(gpu="A100", precision="bf16", kv_precision="fp8"))
    assert kv8["kv_bytes_per_token"] < base["kv_bytes_per_token"]
    assert math.isclose(kv8["kv_bytes_per_token"], base["kv_bytes_per_token"] / 2.0, rel_tol=1e-6)


def test_kv_precision_helps_memory_bound_decode():
    f = roofline_action_factors(ActionBundle(kv_cache_precision_policy="kv_fp8"), _WL_MEM, gpu="A100")
    assert f["decode_factor"] < 1.0          # faster decode in the bandwidth-bound regime
    assert f["kv_memory_saved_pct"] > 0.0
    assert f["quality_sla_risk"] == 0.0      # fp8 KV is headline-safe


def test_kv_precision_neutral_when_compute_bound():
    # KV-byte reduction barely moves a compute-bound decode (it binds the bandwidth term, not compute) — and
    # moves it FAR less than the memory-bound case, where the same knob is worth several percent.
    f_comp = roofline_action_factors(ActionBundle(kv_cache_precision_policy="kv_fp8"), _WL_COMP, gpu="A100")
    f_mem = roofline_action_factors(ActionBundle(kv_cache_precision_policy="kv_fp8"), _WL_MEM, gpu="A100")
    assert f_comp["decode_factor"] >= 0.99                       # ~neutral when not bandwidth-bound
    assert (1.0 - f_comp["decode_factor"]) < (1.0 - f_mem["decode_factor"])


def test_kv_precision_no_reward_bonus_at_default():
    f = roofline_action_factors(ActionBundle(), _WL_MEM, gpu="A100")
    for k in ("decode_factor", "prefill_factor", "gpu_seconds_factor", "completion_factor"):
        assert math.isclose(f[k], 1.0, rel_tol=1e-9)
    assert f["quality_sla_risk"] == 0.0


def test_kv_int4_unsafe_excluded_from_headline():
    assert not is_headline_safe_kv("kv_int4_diagnostic_only")
    assert is_headline_safe_kv("kv_fp8") and is_headline_safe_kv("kv_int8")
    f = roofline_action_factors(ActionBundle(kv_cache_precision_policy="kv_int4_diagnostic_only"),
                                _WL_MEM, gpu="A100")
    assert f["quality_sla_risk"] > 0.0       # carries an UNMODELLED quality risk → diagnostic only


def test_kv_precision_memory_effect_capacity_and_pressure():
    e = kv_precision_memory_effect("kv_fp8", gpu_type="A100", active_sequences=120, context_tokens=832)
    assert e.active_sequence_capacity_after > e.active_sequence_capacity_before   # more concurrency
    assert e.hbm_pressure_after < e.hbm_pressure_before                           # less HBM pressure
    assert e.kv_memory_saved_pct > 0.0
    noop = kv_precision_memory_effect("inherit_weight_precision", gpu_type="A100")
    assert noop.kv_memory_saved_pct == 0.0 and noop.capacity_gain_pct == 0.0      # no-op = zero deltas


# --- heterogeneous GPU assignment: no fake gain on a homogeneous fleet -------------------------------
_CLASSES = [WorkloadClass("latency_sensitive", 1024, 128, 2.0, sla_s=0.30, kind="latency"),
            WorkloadClass("batch", 1024, 512, 1.0, None, kind="batch"),
            WorkloadClass("memory_heavy", 4096, 256, 0.5, sla_s=8.0, kind="memory_heavy")]
_HETERO = [GPUType("H100", 4), GPUType("A10", 8), GPUType("H20", 4)]


def test_homogeneous_fleet_no_fake_benefit():
    cmp = compare_assignment_policies(_CLASSES, [GPUType("A100", 16)])
    gps = {round(r["gp_per_dollar"], 2) for p, r in cmp["results"].items() if r["deployable"]}
    assert cmp["homogeneous_fleet"] and len(gps) == 1            # one GPU type → every policy ties


def test_latency_sensitive_prefers_fast_gpu_under_tight_sla():
    cmp = compare_assignment_policies(_CLASSES, _HETERO)
    homo = cmp["results"]["homogeneous_default"]
    fast = cmp["results"]["fastest_for_latency_sensitive"]
    # cheap-everywhere violates the tight latency SLA; routing latency work to a fast GPU fixes it.
    assert homo["sla_violation_rate"] > 0.0
    assert fast["sla_violation_rate"] < homo["sla_violation_rate"]
    assert fast["gp_per_dollar"] > homo["gp_per_dollar"]


def test_memory_heavy_prefers_high_hbm():
    cmp = compare_assignment_policies(_CLASSES, _HETERO)
    mapping = cmp["results"]["memory_heavy_to_high_hbm"]["workload_to_gpu_mapping"]
    assert mapping["memory_heavy"] == "H20"                      # H20 has the most HBM in the fleet


def test_gpu_assignment_deterministic():
    a = compare_assignment_policies(_CLASSES, _HETERO)["results"]["balanced_heterogeneous"]
    b = compare_assignment_policies(_CLASSES, _HETERO)["results"]["balanced_heterogeneous"]
    assert a["workload_to_gpu_mapping"] == b["workload_to_gpu_mapping"]
    assert a["gp_per_dollar"] == b["gp_per_dollar"]


def test_oracle_assignment_is_non_deployable():
    cmp = compare_assignment_policies(_CLASSES, _HETERO)
    assert cmp["results"]["diagnostic_oracle_assignment"]["deployable"] is False
    assert cmp["best_deployable_policy"] != "diagnostic_oracle_assignment"


# --- PD disaggregation: has a handoff cost + can help and hurt --------------------------------------
def test_pd_shared_has_no_handoff_disaggregated_does():
    wl = PDWorkload(arrival_rate=8.0, prefill_work_s=0.4, decode_work_s=0.4, context_tokens=1024)
    shared = pd_serving_point(wl, "shared", n_replicas=12)
    disagg = pd_serving_point(wl, "prefill_heavy", n_replicas=12)
    assert shared.kv_handoff_latency == 0.0 and shared.kv_handoff_bytes == 0.0
    assert disagg.kv_handoff_latency > 0.0 and disagg.kv_handoff_bytes > 0.0   # disaggregation is not free


def test_pd_wrong_split_hurts():
    # prefill-heavy workload; a decode-heavy split saturates the prefill pool → completion explodes.
    ph = PDWorkload(arrival_rate=11.0, prefill_work_s=0.6, decode_work_s=0.22, context_tokens=2048)
    right = pd_serving_point(ph, "prefill_heavy", n_replicas=12)
    wrong = pd_serving_point(ph, "decode_heavy", n_replicas=12)
    assert wrong.mean_completion_s > 3.0 * right.mean_completion_s
    assert wrong.prefill_pool_utilization > 1.0                 # the wrong pool is saturated


def test_pd_prefill_heavy_helps_prefill_heavy_workload():
    ph = PDWorkload(arrival_rate=11.0, prefill_work_s=0.6, decode_work_s=0.22, context_tokens=2048)
    shared = pd_serving_point(ph, "shared", n_replicas=12)
    split = pd_serving_point(ph, "prefill_heavy", n_replicas=12)
    assert split.mean_completion_s < shared.mean_completion_s    # isolation avoids HoL interference
    assert split.idle_gpu_seconds_total >= 0.0


def test_pd_idle_seconds_reported_by_pool():
    wl = PDWorkload(arrival_rate=4.0, prefill_work_s=0.3, decode_work_s=0.3, context_tokens=512)
    r = pd_serving_point(wl, "p60_d40", n_replicas=12)
    assert r.idle_gpu_seconds_prefill >= 0.0 and r.idle_gpu_seconds_decode >= 0.0
    assert r.allocation_efficiency <= 1.0


# --- planner: regime-gated generation ----------------------------------------------------------------
def _gen(state):
    return PhysicsGuidedCandidateGenerator().generate(state)


def test_kv_precision_generated_only_in_memory_or_hbm_regime():
    mem = _gen(PlannerRegimeState(decode_regime="memory_bandwidth_bound", hbm_pressure=0.7))
    kv_opts = {b.kv_cache_precision_policy for b in mem.bundles}
    assert "kv_fp8" in kv_opts and "kv_int8" in kv_opts
    assert "kv_int4_diagnostic_only" not in kv_opts             # headline-safe: int4 KV opt-in only
    comp = _gen(PlannerRegimeState(decode_regime="compute_bound", hbm_pressure=0.1))
    assert {b.kv_cache_precision_policy for b in comp.bundles} == {"inherit_weight_precision"}
    assert "kv_precision_frozen" in comp.pruned_reasons


def test_pd_generated_only_when_divergent():
    div = _gen(PlannerRegimeState(decode_regime="memory_bandwidth_bound", pd_divergence=True,
                                  prefill_heavy=False, capacity_pressure=0.5))
    assert "p40_d60" in {b.prefill_decode_policy for b in div.bundles}   # decode-heavy → p40_d60
    nondiv = _gen(PlannerRegimeState(decode_regime="memory_bandwidth_bound", pd_divergence=False))
    assert {b.prefill_decode_policy for b in nondiv.bundles} == {"shared"}
    assert "prefill_decode_frozen" in nondiv.pruned_reasons


def test_gpu_assignment_frozen_off_on_production_fleet():
    cs = _gen(PlannerRegimeState(decode_regime="memory_bandwidth_bound", heterogeneous_fleet=False))
    assert {b.gpu_assignment_policy for b in cs.bundles} == {"homogeneous_default"}
    assert "gpu_assignment_frozen" in cs.pruned_reasons


def test_ablation_mask_freezes_disabled_knobs():
    # mask allows only KV → PD must stay at no-op even when divergent.
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", hbm_pressure=0.7, pd_divergence=True,
                            prefill_heavy=True, allowed_new_knobs=frozenset({"kv_cache_precision_policy"}))
    cs = _gen(st)
    assert {b.prefill_decode_policy for b in cs.bundles} == {"shared"}
    assert "kv_fp8" in {b.kv_cache_precision_policy for b in cs.bundles}


def test_anchors_preserved_with_new_knobs():
    cs = _gen(PlannerRegimeState(decode_regime="memory_bandwidth_bound", hbm_pressure=0.7))
    assert cs.anchor_flags["neutral"] and cs.anchor_flags["known_strong"]
    # the neutral bundle (all no-op) is always present.
    assert any(is_neutral_roofline_bundle(b) for b in cs.bundles)


# --- determinism + ablation reporting ----------------------------------------------------------------
def test_deterministic_modulation():
    b = ActionBundle(kv_cache_precision_policy="kv_fp8", precision_policy="fp8")
    a1 = period_action_modulation(b, _RECS, gpu="A100")
    a2 = period_action_modulation(b, _RECS, gpu="A100")
    assert a1 == a2


def test_ablation_pct_helper_reports_abs_and_pct():
    from scripts.run_batch1_ablation import _pct
    assert _pct(150.0, 100.0) == 50.0
    assert _pct(100.0, 0.0) is None      # guarded division


def test_pareto_gate_unchanged():
    # the ladder's Pareto clause (gp/$ up AND SLA not worse) is unchanged by this PR.
    from scripts.run_ladder_benchmark import summarize
    state = {"cells": {
        "pjm|expensive|production_scheduler": {"status": "COMPLETED",
            "result": {"gp_per_dollar": 100.0, "sla_violation_rate": 0.01}},
        "pjm|expensive|sla_aware": {"status": "COMPLETED",
            "result": {"gp_per_dollar": 90.0, "sla_violation_rate": 0.01}},
        "pjm|expensive|aurelius_mpc_hierarchical_search": {"status": "COMPLETED",
            "result": {"gp_per_dollar": 200.0, "sla_violation_rate": 0.0}}}}
    s = summarize(state)["pjm|expensive"]["aurelius_mpc_hierarchical_search"]["vs_production_scheduler"]
    assert s["abs_delta"] == 100.0 and s["pct_delta"] == 100.0 and s["sla_not_worse"] is True
