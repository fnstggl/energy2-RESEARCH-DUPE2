"""Controlled fixtures for the PR #107 prefill/decode model + service-time-sensitive economics.

Every benefit flows through prefill work / realized GPU-seconds — never a reward bonus. These prove the
causal bridge: KV saves prefill → lower TTFT/realized work → (under a work-sensitive cost mode) lower
cost. They also prove the honest limits: decode-bound workloads barely monetize; no cost mode is free."""

from __future__ import annotations

from types import SimpleNamespace

from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.prefill_decode import (
    TPOT_S,
    TTFT_BASE_S,
    compute_phase_serving,
    effective_gpu_hours,
)
from aurelius.environment.world_simulator import initialize_world_state, simulate_period, warm_seed


def _recs(n, out, prompt):
    return [(float(i), out, prompt) for i in range(n)]


# --- prefill/decode separation (the Cause-A fix) ----------------------------

def test_prefix_hit_reduces_prefill_only_decode_unchanged():
    recs = _recs(4, out=100, prompt=512)
    cold = compute_phase_serving(recs, [0, 0, 0, 0])
    hit = compute_phase_serving(recs, [512, 512, 512, 512])     # full prompt cached
    assert hit.prefill_gpu_seconds < cold.prefill_gpu_seconds   # prefill falls
    assert hit.decode_gpu_seconds == cold.decode_gpu_seconds    # decode UNCHANGED (KV-insensitive)
    assert all(abs(d - 100 * TPOT_S * 0.92) < 1e-6 for d in hit.decode_work_s)  # balanced batch factor


def test_token_conservation_prefill_remaining():
    recs = _recs(3, out=50, prompt=300)
    r = compute_phase_serving(recs, [160, 0, 300])
    assert r.prefill_tokens_total == 900
    assert r.prefill_tokens_saved == 160 + 0 + 300
    assert r.prefill_tokens_remaining == (300 - 160) + 300 + 0
    assert r.decode_tokens_total == 150


def test_ttft_falls_with_prefix_hit():
    recs = _recs(2, out=80, prompt=2000)
    cold = compute_phase_serving(recs, [0, 0])
    hit = compute_phase_serving(recs, [2000, 2000])
    assert hit.summary()["ttft_p95"] < cold.summary()["ttft_p95"]
    assert hit.ttft_s[0] == TTFT_BASE_S                         # full hit → only the base prefill term


def test_decode_heavy_stays_decode_bound_despite_reuse():
    # long outputs, short prompts: decode dominates → KV reuse barely changes completion/realized work.
    recs = _recs(20, out=2000, prompt=128)
    cold = compute_phase_serving(recs, [0] * 20)
    hit = compute_phase_serving(recs, [128] * 20)              # full prompt cached
    assert cold.summary()["phase_bottleneck"] == "decode_bound"
    rel = (cold.realized_gpu_seconds - hit.realized_gpu_seconds) / cold.realized_gpu_seconds
    assert rel < 0.05                                          # <5% realized-work reduction (decode-bound)


def test_prefill_heavy_benefits_more_from_reuse():
    # long prompts, short outputs: prefill is a big share → reuse cuts realized work materially.
    recs = _recs(20, out=20, prompt=4000)
    cold = compute_phase_serving(recs, [0] * 20)
    hit = compute_phase_serving(recs, [4000] * 20)
    rel = (cold.realized_gpu_seconds - hit.realized_gpu_seconds) / cold.realized_gpu_seconds
    assert rel > 0.3                                          # prefill-heavy → large realized-work cut


def test_batching_changes_decode_work():
    recs = _recs(5, out=200, prompt=128)
    cons = compute_phase_serving(recs, [0] * 5, batching="conservative")
    aggr = compute_phase_serving(recs, [0] * 5, batching="aggressive")
    assert aggr.decode_gpu_seconds < cons.decode_gpu_seconds   # aggressive batching lowers per-token decode


# --- cost modes (the Cause-B fix) -------------------------------------------

def test_cost_mode_ordering_and_no_free_cost():
    prov_s, real_s = 1000.0, 300.0                            # realized << provisioned (slack)
    prov = effective_gpu_hours("provisioned_capacity", provisioned_gpu_seconds=prov_s, realized_gpu_seconds=real_s)
    hyb = effective_gpu_hours("hybrid_capacity_work", provisioned_gpu_seconds=prov_s, realized_gpu_seconds=real_s)
    real = effective_gpu_hours("realized_serving_work", provisioned_gpu_seconds=prov_s, realized_gpu_seconds=real_s)
    assert real <= hyb <= prov                                # realized cheapest, provisioned dearest
    assert real > 0.0 and hyb >= 0.5 * prov                   # never free; hybrid keeps a warm-idle floor


def test_realized_work_falls_when_prefill_falls():
    recs = _recs(10, out=30, prompt=3000)
    cold = compute_phase_serving(recs, [0] * 10)
    hit = compute_phase_serving(recs, [3000] * 10)
    prov_s = cold.realized_gpu_seconds * 1.5                  # a TIGHT period (provisioned ≈ realized)
    real_cold = effective_gpu_hours("realized_serving_work", provisioned_gpu_seconds=prov_s,
                                    realized_gpu_seconds=cold.realized_gpu_seconds)
    real_hit = effective_gpu_hours("realized_serving_work", provisioned_gpu_seconds=prov_s,
                                   realized_gpu_seconds=hit.realized_gpu_seconds)
    assert real_hit < real_cold                              # KV reuse → lower realized-work cost


def test_determinism():
    recs = _recs(8, out=64, prompt=512)
    assert compute_phase_serving(recs, [0] * 8).service_s == compute_phase_serving(recs, [0] * 8).service_s


# --- end-to-end through simulate_period -------------------------------------

def test_provisioned_mode_reproduces_no_gp_dollar_gain_realized_does():
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware", batching_policy="balanced")
    prefixes = [tuple(f"p{p}_{b}" for b in range(8)) for p in range(8)]
    recs = [(float(i), 30, 3000) for i in range(60)]          # prefill-heavy + reuse
    hs = [prefixes[i % 8] for i in range(60)]

    def run(mode):
        ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
        warm_seed(ws, 8)
        return simulate_period(ws, pol, recs, {"arrival_rate": 1.0, "arrival_p90": 1.5, "mean_service_s": 1.0},
                               sla_s=10.0, tick_seconds=10.0, cost_model=cm, fleet_state=fleet,
                               base_service_factor=0.95, period_hours=0.0167, dt_seconds=60.0,
                               kv_state={"hash_seq": hs, "routing": "kv_aware", "cost_mode": mode}, mutate=False)
    prov, real = run("provisioned_capacity"), run("realized_serving_work")
    assert real.goodput_per_dollar > prov.goodput_per_dollar  # realized-work monetizes the prefill saving
    assert prov.kv_diag["prefill_tokens_saved"] > 0           # the channel fired in both
    assert prov.kv_diag["phase_bottleneck"] in ("prefill_bound", "mixed", "decode_bound")
