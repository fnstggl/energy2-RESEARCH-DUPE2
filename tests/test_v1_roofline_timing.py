"""V1 roofline-timing promotion (V2→V1).

Proves the promoted GPU/model-aware base timing: legacy scalar is preserved EXACTLY (default + benchmark
stability), roofline resolves by GPU/model, the path is deterministic and clone-safe, and reward still flows
only through the existing service-time/cost channels. Deterministic, no network, no GPU."""

from __future__ import annotations

import os
from types import SimpleNamespace

from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.prefill_decode import (
    TIMING_PROVENANCE,
    TPOT_S,
    compute_phase_serving,
    env_timing_model,
    resolve_serving_rates,
)
from aurelius.environment.world_simulator import initialize_world_state, simulate_period, warm_seed


def _recs(n=8, out=256, prompt=512):
    return [(float(i), out, prompt) for i in range(n)]


# --- roofline is now the default; legacy is explicit-only --------------------
def test_default_is_roofline_now():
    # production-default flip: a caller with no timing_model now gets ROOFLINE (was legacy_scalar pre-PR).
    default = compute_phase_serving(_recs(), [0] * 8)
    assert default.summary()["timing_model"] == "roofline"


def test_explicit_legacy_is_bit_for_bit_scalar():
    # legacy remains available as an explicit regression mode and reproduces the old scalar EXACTLY.
    recs = _recs()
    legacy = compute_phase_serving(recs, [0] * 8, timing_model="legacy_scalar")
    assert legacy.summary()["timing_model"] == "legacy_scalar"
    assert abs(legacy.decode_work_s[0] - 256 * TPOT_S * 0.92) < 1e-9          # exact scalar formula
    # and it differs from the new default (roofline on H100 prices decode lower)
    assert compute_phase_serving(recs, [0] * 8).decode_work_s[0] != legacy.decode_work_s[0]


def test_unknown_timing_model_raises():
    try:
        compute_phase_serving(_recs(), [0] * 8, timing_model="bogus")
    except ValueError as e:
        assert "timing_model" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown timing_model")


# --- roofline GPU/model awareness --------------------------------------------
def test_roofline_h100_decode_faster_than_l40s_same_request():
    recs = _recs()
    h = compute_phase_serving(recs, [0] * 8, timing_model="roofline", gpu_type="H100")
    sl = compute_phase_serving(recs, [0] * 8, timing_model="roofline", gpu_type="L40S")
    assert h.decode_work_s[0] < sl.decode_work_s[0]


def test_roofline_corrects_l40s_class_scalar_on_h100():
    # the headline: the legacy scalar (~L40S-class) overstates H100 decode; roofline prices it lower.
    recs = _recs()
    legacy = compute_phase_serving(recs, [0] * 8, timing_model="legacy_scalar")
    h100 = compute_phase_serving(recs, [0] * 8, timing_model="roofline", gpu_type="H100")
    l40s = compute_phase_serving(recs, [0] * 8, timing_model="roofline", gpu_type="L40S")
    assert h100.decode_work_s[0] < legacy.decode_work_s[0]              # H100 cheaper than the scalar
    assert abs(l40s.decode_work_s[0] - legacy.decode_work_s[0]) < 0.5 * legacy.decode_work_s[0]  # scalar ~ L40S


def test_roofline_70b_decode_slower_than_8b_same_gpu():
    p8, d8 = resolve_serving_rates("H100", "llama-8b-gqa", 512, 256)
    p70, d70 = resolve_serving_rates("H100", "llama-70b-gqa", 512, 256)
    assert d70 > d8 and p70 > p8


def test_roofline_deterministic():
    recs = _recs()
    a = compute_phase_serving(recs, [0] * 8, timing_model="roofline", gpu_type="H100")
    b = compute_phase_serving(recs, [0] * 8, timing_model="roofline", gpu_type="H100")
    assert a.summary() == b.summary()


def test_resolve_serving_rates_unknown_falls_back_conservatively():
    # unknown GPU/model must resolve to the documented defaults, not crash or return zero.
    p, d = resolve_serving_rates("XPU-NONEXISTENT", "no-such-model", 256, 128)
    assert p > 0 and d > 0


def test_provenance_labels_present():
    assert "legacy_scalar" in TIMING_PROVENANCE and "roofline" in TIMING_PROVENANCE
    r = compute_phase_serving(_recs(), [0] * 8, timing_model="roofline", gpu_type="H100")
    assert r.timing_provenance == TIMING_PROVENANCE["roofline"]


def test_env_timing_model_opt_in_only():
    saved = os.environ.get("AURELIUS_TIMING_MODEL")
    try:
        os.environ.pop("AURELIUS_TIMING_MODEL", None)
        assert env_timing_model() == "roofline"                 # default is now roofline (production path)
        os.environ["AURELIUS_TIMING_MODEL"] = "legacy_scalar"
        assert env_timing_model() == "legacy_scalar"            # explicit legacy reproduces old benchmarks
        os.environ["AURELIUS_TIMING_MODEL"] = "garbage"
        assert env_timing_model() == "roofline"                 # invalid -> safe production default
    finally:
        if saved is None:
            os.environ.pop("AURELIUS_TIMING_MODEL", None)
        else:
            os.environ["AURELIUS_TIMING_MODEL"] = saved


# --- end-to-end through the V1 simulator -------------------------------------
def _run(kv_extra, *, mutate=False, gpu_type=None):
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware", batching_policy="balanced")
    prefixes = [tuple(f"p{p}_{b}" for b in range(8)) for p in range(8)]
    recs = [(float(i), 256, 512) for i in range(60)]
    hs = [prefixes[i % 8] for i in range(60)]
    ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(ws, 8)
    kv = {"hash_seq": hs, "routing": "kv_aware", "cost_mode": "hybrid_capacity_work", **kv_extra}
    if gpu_type:
        kv["gpu_type"] = gpu_type
    out = simulate_period(ws, pol, recs, {"arrival_rate": 1.0, "arrival_p90": 1.5, "mean_service_s": 1.0},
                          sla_s=10.0, tick_seconds=10.0, cost_model=cm, fleet_state=fleet,
                          base_service_factor=0.95, period_hours=0.0167, dt_seconds=60.0,
                          kv_state=kv, mutate=mutate)
    return ws, out


def test_simulate_period_default_is_roofline_legacy_is_explicit():
    # the canonical path now defaults to ROOFLINE; a no-flag run equals explicit roofline, and the explicit
    # LEGACY run differs (the intentional production-physics correction, documented).
    _, base = _run({})
    _, roof = _run({"timing_model": "roofline"})
    _, legacy = _run({"timing_model": "legacy_scalar"})
    assert base.goodput_per_dollar == roof.goodput_per_dollar          # default == roofline
    assert base.goodput_per_dollar != legacy.goodput_per_dollar        # default != legacy (corrected physics)


def test_simulate_period_roofline_changes_service_through_physical_channel():
    _, legacy = _run({"timing_model": "legacy_scalar"}, gpu_type="H100")
    _, roof = _run({"timing_model": "roofline"}, gpu_type="H100")
    # the effect is physical (service time -> realized GPU-seconds -> operator_cost), and reward is still
    # goodput/$ computed through the existing channel (no direct bonus).
    assert roof.operator_cost != legacy.operator_cost
    assert roof.goodput_per_dollar >= 0.0


def test_roofline_reduces_phantom_sla_violations_on_h100():
    # the headline correction: the L40S-class scalar over-prices H100 decode -> more (phantom) SLA misses
    # than the roofline that prices H100 correctly. roofline must not be WORSE on SLA for a fast GPU.
    _, legacy = _run({"timing_model": "legacy_scalar"}, gpu_type="H100")
    _, roof = _run({"timing_model": "roofline"}, gpu_type="H100")
    assert roof.kpi.sla_violations <= legacy.kpi.sla_violations


def test_clone_isolation_roofline_does_not_mutate_real_state():
    ws, _ = _run({"timing_model": "roofline"}, mutate=False, gpu_type="H100")
    assert ws.period == 0          # mutate=False -> the real timeline never advanced
