#!/usr/bin/env python3
"""Bounded dt=60 Azure+Mooncake diagnostic for the V2 serving world model (Phase 11) + V1-vs-V2 (Phase X).

Runs configs A–G over a deterministic multi-hour window at dt=60s on a persistent V2 ClusterState, plus a
controlled V1-equivalent (legacy-scalar timing) vs V2 (roofline) comparison holding everything else fixed.
Deterministic, no network, no external deps. Emits JSON to stdout (and the scratchpad if a path is given).

  A legacy_scalar         B roofline_live          C +disaggregation_sweep   D +tiered_KV(full vs HBM-only)
  E +upgraded_batching    F +roofline_MPC_actions  G full + adaptive_search

Usage:  python scripts/run_dt60_full_serving_physics.py [n_periods] [out.json]
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.environment.v2.candidate_generator import generate_candidates  # noqa: E402
from aurelius.environment.v2.mpc_search import AdaptiveMPCSearchV2  # noqa: E402
from aurelius.environment.v2.world_simulator import (  # noqa: E402
    WorldSimulatorV2,
    simulate_period_v2,
)
from aurelius.environment.v2.world_state import build_fleet_v2, clone_state_v2  # noqa: E402

SLA_S = 5.0
PERIOD_S = 60.0
N_REPLICAS = 8


def _gen_window(n_periods: int, seed: int = 0):
    """Deterministic Azure-like per-period request batches + Mooncake-style hash families (reuse)."""
    rng = random.Random(seed)
    periods = []
    for p in range(n_periods):
        # diurnal-ish rate
        import math
        rate = 40 + int(30 * (1 + math.sin(2 * math.pi * p / max(1, n_periods))))
        reqs, hs = [], []
        for i in range(rate):
            prompt = rng.choice([64, 128, 256, 512, 1024])
            out = rng.choice([64, 128, 256, 512])
            reqs.append((i * (PERIOD_S / max(1, rate)), out, prompt))
            fam = rng.randint(0, 19)                       # 20 prefix families -> reuse
            hs.append([fam * 10 + j for j in range(rng.choice([1, 2, 3]))])
        periods.append((reqs, hs))
    return periods


def _agg(outcomes):
    """Aggregate per-period outcomes into headline metrics."""
    if not outcomes:
        return {}
    import statistics
    good = sum(o.metrics["sla_safe_goodput_tokens"] for o in outcomes)
    cost = sum(o.metrics["cost_usd"] for o in outcomes)
    comp95 = statistics.mean(o.serving["completion_p95"] for o in outcomes)
    ttft95 = statistics.mean(o.serving["ttft_p95"] for o in outcomes)
    viol = statistics.mean(o.sla_violation_rate for o in outcomes)
    energy = sum(o.metrics["energy_cost"] for o in outcomes)
    gpu_h = sum(o.metrics["billed_gpu_hours"] for o in outcomes)
    realized = sum(o.metrics["realized_gpu_seconds"] for o in outcomes)
    regimes = {}
    for o in outcomes:
        regimes[o.timing["regime"]] = regimes.get(o.timing["regime"], 0) + 1
    return {"gp_per_dollar": round(good / cost, 2) if cost else 0.0,
            "sla_safe_goodput_tokens": round(good, 0), "cost_usd": round(cost, 4),
            "energy_cost_usd": round(energy, 5), "billed_gpu_hours": round(gpu_h, 4),
            "realized_gpu_seconds": round(realized, 1),
            "completion_p95_mean_s": round(comp95, 4), "ttft_p95_mean_s": round(ttft95, 4),
            "sla_violation_rate_mean": round(viol, 4), "roofline_regime_mix": regimes}


def run_config(label, periods, action_fn, *, base_gpu="H100", bg=0.0, hbm_only=False):
    """Run all periods on a fresh persistent state with per-period action from ``action_fn``."""
    st = build_fleet_v2(n_replicas=N_REPLICAS, gpu_type=base_gpu, background_work_gpu_seconds=bg,
                        cap_hbm=512)
    if hbm_only:
        for r in st.replicas:
            r.kv.cap_cpu = r.kv.cap_remote = r.kv.cap_ssd = 0
    sim = WorldSimulatorV2()
    outcomes, decision_times, regrets = [], [], []
    for reqs, hs in periods:
        t0 = time.perf_counter()
        action, regret = action_fn(st, reqs, hs, sim)
        decision_times.append(time.perf_counter() - t0)
        outcomes.append(simulate_period_v2(st, reqs, hs, action, sla_s=SLA_S, period_s=PERIOD_S,
                                            cost_model=sim.cost_model, mutate=True))
        if regret is not None:
            regrets.append(regret)
    agg = _agg(outcomes)
    agg["runtime_per_decision_ms"] = round(1000 * sum(decision_times) / len(decision_times), 4)
    agg["mean_search_regret"] = round(sum(regrets) / len(regrets), 5) if regrets else None
    agg["config"] = label
    return agg


# --- action functions per config ------------------------------------------
def _fixed(action):
    return lambda st, reqs, hs, sim: ({**action}, None)


def _disagg_sweep(st, reqs, hs, sim):
    best, best_r = {"timing_model": "roofline"}, float("-inf")
    for mode, frac in [("shared_pool", None), ("disaggregated_static", 0.4),
                       ("disaggregated_static", 0.5), ("disaggregated_static", 0.6)]:
        a = {"timing_model": "roofline", "serving_mode": mode, "prefill_frac": frac}
        r = sim.evaluate(clone_state_v2(st), reqs, hs, a, sla_s=SLA_S, period_s=PERIOD_S, mutate=False).reward
        if r > best_r:
            best, best_r = a, r
    return best, None


def _mpc_actions(strategy):
    def fn(st, reqs, hs, sim):
        # probe the regime on a clone, generate regime-conditioned candidates, search
        probe = sim.evaluate(clone_state_v2(st), reqs, hs, {"timing_model": "roofline"}, sla_s=SLA_S,
                             period_s=PERIOD_S, mutate=False)
        regime = probe.timing["regime"]
        has_bg = st.background_work_gpu_seconds > 0
        cands = generate_candidates(regime if regime in ("compute", "memory") else "mixed",
                                    hbm_pressure=probe.timing["hbm_pressure"], has_background_work=has_bg)
        space = {}
        for c in cands.candidates:
            for k, v in c.items():
                space.setdefault(k, [])
                if v not in space[k]:
                    space[k].append(v)
        space["timing_model"] = ["roofline"]
        search = AdaptiveMPCSearchV2(space, exhaustive_threshold=400, beam_k=6)

        def ev(action):
            return sim.evaluate(clone_state_v2(st), reqs, hs, action, sla_s=SLA_S, period_s=PERIOD_S,
                                mutate=False).reward
        res = search.search(ev, strategy=strategy, audit=(strategy == "beam_search"))
        return res.selected, res.search_regret
    return fn


def main():
    n_periods = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    periods = _gen_window(n_periods)

    configs = {
        "A_legacy_scalar": run_config("A_legacy_scalar", periods, _fixed({"timing_model": "legacy_scalar"})),
        "B_roofline_live": run_config("B_roofline_live", periods, _fixed({"timing_model": "roofline"})),
        "C_disaggregation_sweep": run_config("C_disaggregation_sweep", periods, _disagg_sweep),
        "D_tiered_kv_full": run_config("D_tiered_kv_full", periods, _fixed({"timing_model": "roofline"})),
        "D_tiered_kv_hbm_only": run_config("D_tiered_kv_hbm_only", periods,
                                           _fixed({"timing_model": "roofline"}), hbm_only=True),
        "E_upgraded_batching": run_config("E_upgraded_batching", periods,
                                          _fixed({"timing_model": "roofline", "max_active_sequences": 128,
                                                  "chunked_prefill": True, "max_num_batched_tokens": 4096})),
        "F_roofline_mpc_actions": run_config("F_roofline_mpc_actions", periods, _mpc_actions("beam_search")),
        "G_full_adaptive_search": run_config("G_full_adaptive_search", periods, _mpc_actions("beam_search"),
                                             bg=150.0),
    }
    # V1-equivalent (legacy scalar) vs V2 (roofline), identical inputs/actions otherwise
    v1v2 = {"V1_legacy_scalar": configs["A_legacy_scalar"], "V2_roofline": configs["B_roofline_live"]}

    report = {"meta": {"n_periods": n_periods, "dt_seconds": PERIOD_S, "n_replicas": N_REPLICAS,
                       "sla_s": SLA_S, "gpu": "H100", "model": "llama-8b-gqa"},
              "configs": configs, "v1_vs_v2": v1v2}
    js = json.dumps(report, indent=2)
    print(js)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
