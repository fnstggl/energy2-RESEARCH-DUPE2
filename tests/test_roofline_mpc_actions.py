"""Controlled fixtures for the roofline-economic MPC actions (precision / spec-decode / clock /
co-location / prefill-decode allocation) + the adaptive search planner.

Each fixture proves an action helps / hurts / is neutral in the **physically-correct roofline regime**,
through the SAME ``roofline.serving_point`` physics the diagnostics use — never a bonus. The honesty
invariants (no direct reward bonus; neutral defaults reproduce; int4 carries a quality risk; co-location
credits no background goodput; pruning is bounded and reported) are locked here.
"""

from __future__ import annotations

from aurelius.environment.actions import ActionBundle
from aurelius.environment.roofline import ServingConfig, Workload, roofline_regime, serving_point
from aurelius.environment.roofline_actions import (
    PRECISION_QUALITY_RISK,
    action_serving_config,
    is_neutral_roofline_bundle,
    roofline_action_factors,
)
from aurelius.environment.search_planner import (
    FROZEN_OFF,
    AdaptiveSearchPlanner,
    roofline_pruned_options,
)

# --- regimes (constructed, not tuned) ---------------------------------------
# memory-bandwidth-bound decode: low batch, long context on a high-compute GPU → AI ≪ ridge.
MEM = (Workload(prompt_tokens=512, decode_tokens=256, context_len=2048), "H100", 8)
# compute-bound decode: very high batch on a bandwidth-rich / compute-light GPU → AI ≥ ridge.
COMP = (Workload(prompt_tokens=512, decode_tokens=64, context_len=512), "H20", 128)


def _b(**kw):
    return ActionBundle(**kw)


def test_regimes_are_what_we_claim():
    wl, gpu, batch = MEM
    assert roofline_regime("decode", ServingConfig(gpu=gpu, batch_size=batch), wl)["roofline_regime"] \
        == "memory_bandwidth_bound"
    wl, gpu, batch = COMP
    assert roofline_regime("decode", ServingConfig(gpu=gpu, batch_size=batch), wl)["roofline_regime"] \
        == "compute_bound"


def test_neutral_bundle_is_exactly_no_op():
    wl, gpu, batch = MEM
    f = roofline_action_factors(ActionBundle(), wl, gpu=gpu, batch_size=batch)
    assert is_neutral_roofline_bundle(ActionBundle())
    for k in ("prefill_factor", "decode_factor", "gpu_seconds_factor", "ttft_factor",
              "completion_factor", "power_factor"):
        assert abs(f[k] - 1.0) < 1e-9
    assert f["quality_sla_risk"] == 0.0


# --- 1,2: precision -----------------------------------------------------------
def test_fp8_helps_memory_bound_both_latency_and_cost():
    wl, gpu, batch = MEM
    f = roofline_action_factors(_b(precision_policy="fp8"), wl, gpu=gpu, batch_size=batch)
    assert f["decode_factor"] < 1.0 and f["gpu_seconds_factor"] < 1.0       # faster decode AND cheaper
    assert f["quality_sla_risk"] == 0.0                                     # fp8 ~lossless


def test_precision_helps_less_when_prefill_compute_bound():
    # prefill on long prompts at this batch is compute-bound → precision barely moves prefill time.
    wl, gpu, batch = MEM
    f = roofline_action_factors(_b(precision_policy="fp8"), wl, gpu=gpu, batch_size=batch)
    assert abs(f["prefill_factor"] - 1.0) < 0.2                              # prefill ~unchanged
    assert f["decode_factor"] < f["prefill_factor"]                         # the win is on decode


def test_int4_carries_quality_risk_unlike_fp8():
    assert PRECISION_QUALITY_RISK["int4"] > 0.0 and PRECISION_QUALITY_RISK["fp8"] == 0.0
    wl, gpu, batch = MEM
    f = roofline_action_factors(_b(precision_policy="int4"), wl, gpu=gpu, batch_size=batch)
    assert f["quality_sla_risk"] > 0.0                                      # not a free win


# --- 3,4: speculative decoding ------------------------------------------------
def test_spec_helps_memory_bound_latency_but_pays_a_compute_tax():
    wl, gpu, batch = MEM
    f = roofline_action_factors(_b(spec_decode_policy="medium"), wl, gpu=gpu, batch_size=batch)
    assert f["decode_factor"] < 1.0 and f["completion_factor"] < 1.0        # faster wall-clock
    # the draft+verify FLOPs are a real tax: GPU-seconds fall LESS than wall-clock does (and fp8, which
    # adds no FLOPs, is a strictly better cost deal at the same regime). Spec is a latency lever first.
    assert f["gpu_seconds_factor"] > f["decode_factor"]
    fp8 = roofline_action_factors(_b(precision_policy="fp8"), wl, gpu=gpu, batch_size=batch)
    assert f["gpu_seconds_factor"] > fp8["gpu_seconds_factor"]


def test_spec_hurts_or_neutral_compute_bound():
    wl, gpu, batch = COMP
    f = roofline_action_factors(_b(spec_decode_policy="aggressive"), wl, gpu=gpu, batch_size=batch)
    # compute-bound: the extra draft+verify FLOPs compete for the scarce resource → no latency win.
    assert f["decode_factor"] >= 1.0 - 1e-9


# --- 5,6,7: clock / DVFS ------------------------------------------------------
def test_low_clock_saves_power_memory_bound():
    wl, gpu, batch = MEM
    f = roofline_action_factors(_b(clock_policy="low"), wl, gpu=gpu, batch_size=batch)
    assert f["power_factor"] < 1.0                                          # less power draw
    assert f["decode_factor"] <= 1.0 + 1e-9                                 # memory-bound → ~no latency hit


def test_low_clock_hurts_latency_compute_bound():
    wl, gpu, batch = COMP
    f = roofline_action_factors(_b(clock_policy="low"), wl, gpu=gpu, batch_size=batch)
    assert f["decode_factor"] > 1.0                                         # compute scales with clock → slower
    assert f["power_factor"] < 1.0


def test_high_clock_helps_compute_bound_latency_at_higher_power():
    wl, gpu, batch = COMP
    f = roofline_action_factors(_b(clock_policy="high"), wl, gpu=gpu, batch_size=batch)
    assert f["decode_factor"] < 1.0 and f["power_factor"] > 1.0             # faster, but more energy


def test_high_clock_does_not_magically_help_memory_bound_throughput():
    wl, gpu, batch = MEM
    f = roofline_action_factors(_b(clock_policy="high"), wl, gpu=gpu, batch_size=batch)
    assert f["decode_factor"] >= 1.0 - 1e-9                                 # bandwidth-bound → clock doesn't help
    assert f["power_factor"] > 1.0                                          # …but still costs energy


# --- 8,9,10: co-location ------------------------------------------------------
def test_colocation_credits_no_background_goodput_without_a_trace():
    wl, gpu, batch = MEM
    # no background-work trace → background_work=False → ZERO useful background GPU-seconds credited.
    f = roofline_action_factors(_b(colocation_policy="aggressive"), wl, gpu=gpu, batch_size=batch,
                                background_work=False)
    assert f["coloc_useful_gpu_seconds"] == 0.0
    assert f["completion_factor"] >= 1.0                                    # only interference → hurts/neutral


def test_colocation_useful_only_with_background_and_sm_headroom():
    wl, gpu, batch = MEM                                                    # memory-bound → idle SMs exist
    f = roofline_action_factors(_b(colocation_policy="conservative"), wl, gpu=gpu, batch_size=batch,
                                background_work=True)
    assert f["coloc_useful_gpu_seconds"] > 0.0                              # real background work uses idle SMs


def test_colocation_hurts_more_when_compute_bound():
    wl, gpu, batch = COMP                                                   # no SM headroom
    f = roofline_action_factors(_b(colocation_policy="aggressive"), wl, gpu=gpu, batch_size=batch,
                                background_work=True)
    assert f["coloc_useful_gpu_seconds"] == 0.0                            # compute-bound → no idle SMs to use
    assert f["completion_factor"] > 1.0                                     # pure interference


# --- 11,12: batching ↔ precision / spec interactions -------------------------
def test_precision_benefit_depends_on_batch():
    wl, gpu = MEM[0], MEM[1]
    small = roofline_action_factors(_b(precision_policy="fp8"), wl, gpu=gpu, batch_size=2)["decode_factor"]
    large = roofline_action_factors(_b(precision_policy="fp8"), wl, gpu=gpu, batch_size=64)["decode_factor"]
    assert small != large                                                  # batching changes the precision win


def test_spec_benefit_depends_on_regime_set_by_batch():
    wl, gpu = COMP[0], COMP[1]
    # at low batch the same GPU/workload is memory-bound (spec can help); at high batch compute-bound.
    lo = roofline_action_factors(_b(spec_decode_policy="medium"), wl, gpu=gpu, batch_size=1)["decode_factor"]
    hi = roofline_action_factors(_b(spec_decode_policy="medium"), wl, gpu=gpu, batch_size=128)["decode_factor"]
    assert lo < 1.0 <= hi + 1e-9                                            # helps low-batch, not high-batch


# --- 13,15: unified planner pruning ------------------------------------------
def test_planner_prunes_differently_by_regime():
    mem = roofline_pruned_options(decode_regime="memory_bandwidth_bound")
    comp = roofline_pruned_options(decode_regime="compute_bound")
    assert "int4" in mem["precision_policy"] and "int4" not in comp["precision_policy"]
    assert "aggressive" in mem["spec_decode_policy"] and comp["spec_decode_policy"] == ("off",)
    assert "low" in mem["clock_policy"] and "high" in comp["clock_policy"]


def test_colocation_and_prefill_decode_are_frozen_off_with_reasons():
    surf = roofline_pruned_options(decode_regime="memory_bandwidth_bound")
    assert "colocation_policy" not in surf and "prefill_decode_policy" not in surf
    assert set(FROZEN_OFF) == {"colocation_policy", "prefill_decode_policy"}
    assert all(FROZEN_OFF[s][1] for s in FROZEN_OFF)                        # every freeze records a reason


def test_planner_reports_regret_and_is_bounded():
    # a synthetic interaction: fp8 + aggressive batching is best (coordinate descent can miss it).
    def score(b):
        s = 100.0 + (5 if b.precision_policy == "fp8" else 0) + (5 if b.batching_policy == "aggressive" else 0)
        return s + (30 if (b.precision_policy == "fp8" and b.batching_policy == "aggressive") else 0)

    surfaces = {"precision_policy": ("bf16", "fp8", "int4"),
                "batching_policy": ("conservative", "balanced", "aggressive"),
                "clock_policy": ("base", "low", "high"),
                "capacity_policy": ("reactive_lag1", "backlog_aware", "forecasted_mcs")}
    planner = AdaptiveSearchPlanner(exhaustive_max=20, beam_width=6, regret_audit_max=10000)
    # beam ALONE is cheaper than exhaustive enumeration
    best0, plan0 = planner.plan(score, surfaces=surfaces, frozen_reasons={}, regret_audit=False)
    assert best0.precision_policy == "fp8" and best0.batching_policy == "aggressive"  # found the interaction
    assert plan0.strategy == "beam_search" and plan0.candidates_evaluated < plan0.raw_candidate_count
    # with the regret audit ON we PAY EXTRA (re-run exhaustive) to MEASURE the loss — never hide it.
    best, plan = planner.plan(score, surfaces=surfaces, frozen_reasons={}, regret_audit=True)
    assert plan.regret_audited and plan.estimated_regret is not None and plan.estimated_regret <= 1e-9
    assert plan.candidates_evaluated >= plan.raw_candidate_count            # audit re-ran exhaustive
    # the per-decision report carries every required field (never a silent cap)
    d = plan.to_dict()
    for k in ("raw_candidate_count", "strategy", "candidates_evaluated", "best_reward",
              "estimated_regret", "best_bundle", "runtime_s"):
        assert k in d


def test_determinism():
    wl, gpu, batch = MEM
    a = roofline_action_factors(_b(precision_policy="fp8", spec_decode_policy="medium"), wl, gpu=gpu, batch_size=batch)
    b = roofline_action_factors(_b(precision_policy="fp8", spec_decode_policy="medium"), wl, gpu=gpu, batch_size=batch)
    assert a == b


def test_serving_point_is_the_single_physics_source():
    # the action config maps to a ServingConfig and the factors are pure serving_point ratios — proving
    # there is no separate magic path (the honesty property: one physics law, applied as a ratio).
    wl, gpu, batch = MEM
    cfg = action_serving_config(_b(precision_policy="fp8"), gpu=gpu, batch_size=batch)
    neutral = serving_point(wl, action_serving_config(ActionBundle(), gpu=gpu, batch_size=batch))
    act = serving_point(wl, cfg)
    f = roofline_action_factors(_b(precision_policy="fp8"), wl, gpu=gpu, batch_size=batch)
    assert abs(f["decode_factor"] - act["decode_gpu_seconds"] / neutral["decode_gpu_seconds"]) < 1e-9
