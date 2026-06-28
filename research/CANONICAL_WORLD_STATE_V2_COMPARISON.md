# Canonical World State V1 vs V2 — Honest Comparison (Phase X)

Both simulators run on **identical deterministic inputs** (same Azure-like request window, same
Mooncake-style prefix stream, same Alibaba-derived topology/economics, same dt=60s, same SLA, same Pareto
gate). The controlled V1-vs-V2 contrast holds everything fixed and flips **only the timing physics**:

- **V1-equivalent** = `timing_model="legacy_scalar"` — the scalar `PREFILL_S_PER_TOKEN=0.00015` /
  `TPOT_S=0.020` timing the canonical V1 world model uses (PR #107 fixtures prove the V2 legacy path
  reproduces V1's scalar service time by construction).
- **V2** = `timing_model="roofline"` — the live FLOP/bandwidth roofline (`roofline_external`, ported from
  InferSim/llm-analysis/LLM-Viewer in PR #110), resolved per (GPU, model, precision, batch, context).

Numbers below are from `scripts/run_dt60_full_serving_physics.py 120` (120 one-minute periods, 8× H100,
llama-8b-gqa, SLA 5 s), reproduced deterministically.

## Headline result

| metric | V1 (legacy scalar) | V2 (roofline) |
|--|--|--|
| gp/$ | **122.96** | **142,349.88** |
| SLA violation rate | **0.994** | **0.000** |
| completion p95 (mean) | **3287.9 s** | **1.074 s** |
| energy cost ($) | 0.8749 | 0.2495 |
| billed GPU-hours | 16.00 | 16.00 |
| roofline regime mix | mixed (n/a) | memory ×120 |

**The single biggest V1→V2 effect is a realism *correction*, not a tuning win.** V1's scalar `TPOT_S=0.020`
is an **L40S-class** decode constant (proved in `ROOFLINE_REUSE_DECISION.md`); applied to an **H100** fleet it
overstates decode time ~4×, so V1 predicts the fleet is catastrophically overloaded — completion p95 ≈ 3288 s
and 99.4 % of requests miss the 5 s SLA. V2's roofline correctly prices H100 decode at ~5 ms/token, so the
*same* workload on the *same* fleet completes in ~1 s with **zero** SLA violations. V1 was hallucinating SLA
failures that the hardware does not actually have. The energy line moves for the same reason: V2's lower
realized work → lower utilization → lower dynamic power (`cost_model` power scales with utilization).

This is exactly the deficit PR #110 flagged (a single fleet-wide timing constant cannot resolve the 4–40×
GPU×model spread). V2 closes it.

## The ten questions

**1. What does V2 model that V1 does not?**
FLOP/bandwidth roofline timing (per GPU/model/precision/batch/context) with arithmetic-intensity ridge-point
regime classification; tiered KV (GPU_HBM→CPU_DRAM→REMOTE_KV→SSD) with a remote-vs-recompute decision;
prefill/decode pool disaggregation with KV-handoff cost; token-budget continuous batching with a saturation
tail; precision/spec-decode/clock/co-location action surfaces; and an adaptive MPC search (beam/exhaustive)
with a regret audit. V1 has none of these.

**2. What does V1 still do better?**
V1 is the **mature, default, fully-integrated** canonical model: it carries the real Azure+Mooncake+Alibaba
two-clock environment, the calibrated cost/sensitivity machinery, the persistent warm/cold/migration replica
identity with PR #99–#107 physics, and the full validation/claim-gate apparatus wired into training/backtest.
V2 is a focused serving-physics layer; it does not (yet) replace V1's trace ingestion, forecasting, or the
production reporting path.

**3. Did V2 improve physical realism?** Yes, decisively — roofline timing, tiered KV, pools, batching, and
the regime classification are all physics V1 approximated with constants or omitted.

**4. Did V2 improve gp/$?** On this workload, gp/$ rises only because V2 *removes V1's phantom SLA
violations* (goodput was being thrown away by an over-pessimistic constant). Beyond that correction, the
*added* mechanisms (disaggregation, tiered KV) are **Pareto-neutral** here (see dt60 diagnostic): operator
cost is provisioned-capacity-dominated, so faster service cuts latency/energy, not cost. The one genuine
incremental gp/$ win is **co-location with real background work** (config G: +40 %, GPU-h 16→11.5).

**5. Did V2 worsen runtime?** Per-decision runtime is sub-millisecond for fixed-action configs; the MPC-search
configs cost ~0.2–1.1 s/decision (config F ≈ 1146 ms over a ~144-candidate memory-regime space, config G ≈
221 ms). At dt=60 s this is negligible. Not materially slower.

**6. Did V2 change action selection?** Yes — the roofline regime (memory-bound here) drives the candidate
generator to prioritise precision/spec/down-clock bundles; the MPC search selects energy-cutting actions
(config F energy 0.25→0.16) that V1's action surface could not express.

**7. Did V2 reveal any prior V1 artifact?** Yes — the central finding: **V1's scalar decode constant produces
phantom SLA violations on modern (H100-class) GPUs.** Any V1 headline computed on a fast-GPU fleet with the
0.020 s/token constant systematically under-counts SLA-safe goodput.

**8. Is V2 ready to replace V1?** **No.** V2 reproduces the controlled fixtures and is more physically
detailed, but it is a serving-physics layer, not the full canonical environment; it has not been run through
V1's trace-ingestion/training/reporting integration, and the comparison here uses a synthetic (if
trace-shaped) window, not the full Azure+Mooncake+v2026 pipeline.

**9. If not, what exact blocker remains?** (a) Wire V2 onto the real Azure/Mooncake/v2026 ingestion +
forecasting that V1 owns; (b) port V2's roofline timing back into V1's `prefill_decode` service path behind a
flag (so the canonical model gets the correction without a fork); (c) calibrate the analytical M/D/1 phase
queue against a per-iteration reference (Vidur) — its near-saturation behaviour is a cliff, not a curve.

**10. Which claims should use V1 vs V2?**
- **V1** for any production-reported headline today (it is the calibrated, claim-gated canonical model) —
  but its timing constant must be acknowledged as L40S-class (or the roofline correction applied).
- **V2** for *causal-fidelity* statements about serving physics (roofline regime, tiered KV, pools, batching,
  precision/spec/clock/co-location) and for the realism-correction finding above.

## Hard-rule compliance

V2 does **not** replace V1 (V1 unchanged, default, runnable). V2 shares only read-only primitives
(`roofline_external`, `cost_model`, fidelity tiers). No reward bonus, no action scalar, no roofline bonus —
every effect flows through TTFT/latency/queue/GPU-seconds/energy/SLA/cost. The Pareto gate is unchanged; the
legacy scalar is used as a **baseline**, not a fair comparator (it is, if anything, pessimistic — the opposite
of the optimistic-baseline trap). V2 becomes default only when the §9 blockers clear and a calibrated
comparison proves it more realistic and not materially slower.
