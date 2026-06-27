# Aurelius Next-Batch Connected Actions + Results (post-PR #99)

PR #99 connected KV-aware **routing** and proved the action-space diagnosis (one real action,
honestly connected, moved goodput/$). This batch connects the **next two** action surfaces the
canonical simulator can score causally today — **`capacity_multiplier`** (replica level) and
**`batching_policy`** (per-replica continuous batching) — and reruns the held-out evaluation.

The honesty bar is unchanged. A CONNECTED action must: change simulator output, change
reward/metrics, have a test proving the effect, have a fair baseline, have fidelity/provenance
documented, and remain behind the **Pareto-aware claim gate**. We do not add knobs; we do not
fake connections; we do not force a win.

> **Status of the headline: still honestly FALSE.** The MPC's goodput/$ edge on the full Azure
> week is real and large, but it is bought with a slightly higher SLA-violation rate, so the
> Pareto clause keeps the headline blocked — exactly as it should. The contribution of this
> batch is **two more real, tested, fairly-baselined levers** and a sharper, honest picture of
> *which* lever actually pays (batching) and *which* is a double-edged tool (capacity).

---

## 1. What was connected (and what was not)

| Action | Decision | Channel | Fidelity | Why |
|---|---|---|---|---|
| **`capacity_multiplier`** (0.75 / 1.0 / 1.5×) | **CONNECTED** | `run_unified_replay` | calibrated to existing sizing model | scales the per-tick sized replica count `c`; more replicas cut queue + SLA but cost strictly more GPU-hours |
| **`batching_policy`** (conservative / balanced / aggressive) | **CONNECTED** | `run_unified_replay` | **INFERRED** (public-prior magnitudes, sanity-banded) | per-replica concurrency (throughput↑, queue↓ at ~constant GPU-hours) traded against service inflation (latency↑, SLA risk↑) |
| routing, capacity, ordering, admission | already CONNECTED (PR #98/#99) | — | — | — |
| prewarming, placement/topology, migration, energy-shift, kv-placement | **DEFERRED** | — | — | each needs simulator state the loop lacks today (per-replica warm state, per-server rack map, cross-period placement/deferral state) — connecting now would be a fake knob |
| clock/DVFS, precision, spec-decode | **PLANNED (left as-is)** | — | — | no power/quality/acceptance model built — per instruction, not connected |

Full reasoning + missing-pieces table: `research/AURELIUS_NEXT_BATCH_ACTION_SIMULABILITY_AUDIT.md`.

### Mechanism (in `run_unified_replay`, all default to today's exact no-op)

- **`capacity_multiplier`** scales both the warmup `c` and the per-tick `cap.decide(st)` output.
  The GPU-hours denominator counts the scaled `c`, so more replicas ⇒ more cost. **No free
  capacity.**
- **`batch_concurrency`** raises the per-replica serving slots (`slots = round(c ×
  batch_concurrency)`) — continuous batching packs more requests onto the same physical replicas
  (GPU-hours unchanged). **`batch_service_factor`** multiplies each batched request's service
  time (shared compute ⇒ higher latency). `BATCHING_MODELS`: conservative `(1.0, 1.0)`, balanced
  `(2.0, 1.15)`, aggressive `(4.0, 1.5)`.

---

## 2. Direct-effect proof (the knobs are real, and not free)

Unit tests in `tests/test_unified_replay.py` (burst trace, deterministic):

| Test | Proves |
|---|---|
| `test_capacity_multiplier_buys_sla_with_more_gpu_hours_no_free_capacity` | 0.75→1.0→1.5× ⇒ GPU-hours **and** cost strictly ↑ **and** SLA violations strictly ↓ — a real Pareto trade |
| `test_batch_concurrency_cuts_queue_at_roughly_constant_gpu_hours` | concurrency 1→2→4× ⇒ violations ↓, SLA-safe goodput ↑, GPU-hours ≈ flat (≤ +5%) |
| `test_batch_service_inflation_is_not_free` | holding concurrency fixed, raising only the service factor ⇒ violations strictly ↑, SLA-safe goodput ↓ — a fake knob could not pay this |
| `test_capacity_multiplier_default_is_exact_noop`, `test_batching_default_is_exact_noop` | the connected defaults reproduce today's run bit-for-bit |

Empirical sweep (same burst trace, `sla_s=10`):

```
capacity_multiplier   0.75 → 1.0 → 1.5    gpu_h 1.10 → 1.51 → 2.10   viol 424 → 395 → 270
batch service factor  1.0  → 1.3 → 1.8    (concurrency fixed)        viol 401 → 420 → 912
batch concurrency     1.0  → 2.0 → 4.0    (service fixed)            viol 395 →  63 →   0   gpu_h ≈ 1.5 flat
```

---

## 3. Search method (Phase 9) — auditable, no knob silently excluded

Connecting both levers grows the connected bundle space to capacity(3)×ordering(2)×admission(2)×
routing(3)×capacity_multiplier(3)×batching(3) = **324 bundles** (> the 256 exhaustive budget).
`CandidateBundleGenerator.search` therefore switches from full enumeration to **coordinate
descent** from the no-op incumbent — it touches every connected dimension at ≈ 50–60 evaluations
(not 324), so no connected knob is dropped. The planner reports method, theoretical combinations,
candidates evaluated, frozen surfaces (with reasons), the **top-10 bundles among those
evaluated**, and a per-surface **ablation**. Freezing is explicit (`frozen` / `frozen_reasons`);
nothing is hand-picked.

---

## 4. Incremental per-action evaluation (held-out periods 126–168)

Each rung freezes the *other* new knob to its no-op and lets one move, so the marginal
contribution is isolated. Reported on **both** gp/$ and the SLA-violation rate (a gp/$ gain paid
for in SLA misses is never hidden). `pr99_core` is the control (both new knobs pinned to no-op;
only the PR-#99 connected set varies).

| Rung | gp/$ | Δ gp/$ vs core | SLA viol | ΔSLA | q_p95 | Pareto OK? | chosen mix |
|---|--:|--:|--:|--:|--:|:--:|---|
| `pr99_core` | 198,515 | — | 0.0938 | — | 162.6s | — | cap 1.0×, conservative |
| `+capacity_mult` | 186,350 | **−6.13%** | 0.1797 | +0.086 | 444.7s | **✗** | cap 0.75× (40/42) |
| `+batching` | 221,658 | **+11.66%** | 0.0173 | −0.076 | 4.0s | **✓** | balanced 37 / aggressive 5 |
| `full` | 272,235 | **+37.13%** | 0.0186 | −0.075 | 9.2s | **✓** | cap 0.75× + balanced |

**The honest findings:**

1. **Batching is the workhorse and a clean Pareto win.** Alone it adds **+11.66% gp/$ while
   *improving* SLA** (queue p95 collapses 162.6s → 4.0s): continuous-batching concurrency drains
   the queue at ~constant GPU-hours.
2. **`capacity_multiplier` alone is a *negative*** (−6.13% gp/$, SLA worse). With the tuned
   `risk_weight=0` config the planner greedily picks 0.75× to chase cheap goodput/$, but without
   batching to compensate, under-provisioning explodes the queue on the high-load tail and the
   goodput loss outweighs the cost saving. **This is the anti-fake-knob evidence in action: the
   lever can hurt, and the simulator shows it.**
3. **The two compound.** Only *with* batching's concurrency does leaner capacity become safe —
   the full bundle (+37.13%) far exceeds batching alone (+11.66%), because 0.75× capacity is now
   viable. The value is in the **interaction**, not either knob in isolation.

Artifact: `data/external/mpc_controller/action_increment_report.json`.

---

## 5. Final fair backtest (full Azure 2024 week, held-out)

Source: full `AzureLLMInferenceDataset2024/1week/conv` (168 hourly periods, 27.3 M requests),
disjoint train < val < eval splits; 42 held-out eval periods. Forecasters fit on train,
controller tuned on val (selected `horizon=1, risk_weight=0, confidence_min=0.1`), then run on
eval. The fair baselines now include operators who **already batch + route KV-aware**
(`aurelius_static_full`, `sla_aware_batched`) and one who **over-provisions** at fixed 1.5×
capacity (`sla_aware_capacity_1p5`) — so the MPC must win by adaptation, not by switching on a
lever a competent static operator would already use.

| Arm | gp/$ | SLA viol | queue p95 | GPU-hours | note |
|---|--:|--:|--:|--:|---|
| **mpc_controller** | **272,235** | 0.0186 | 9.2s | **91.2** | kv_aware ×42, capacity 0.75× ×42, batching balanced ×40 / aggressive ×2 |
| `aurelius_static_full` ⟵ **fair baseline** | 201,607 | 0.0141 | 8.7s | 125.8 | strong static: backlog + conformal + class-aware + kv_aware + **balanced batching** |
| `aurelius_canonical_kv_routing` | 195,413 | 0.0138 | 21.4s | 131.0 | PR-#99 fair baseline (no batching) |
| `sla_aware_kv_routing` | 191,649 | 0.0124 | 7.8s | 134.5 | |
| `sla_aware_batched` | 175,339 | 0.0137 | 0.0s | 144.9 | balanced batching at full (1.0×) capacity — queue fully drained, but pays GPU-hours |
| `aurelius_canonical` | 174,963 | 0.0200 | 22.2s | 141.5 | |
| `fifo_weak` *(weak)* | 174,537 | 0.1106 | 173.1s | 122.1 | never the fair baseline |
| `sla_aware` | 172,289 | 0.0176 | 8.6s | 145.2 | |
| `greedy_packing` *(weak)* | 168,180 | 0.1425 | 264.8s | 121.8 | |
| `sla_aware_capacity_1p5` | 136,386 | **0.0079** | 1.1s | **193.2** | fixed 1.5× over-provision: **best SLA, worst gp/$** — buying compliance with money |

**Claim gate (honest, Pareto-aware):**

```
fair_baseline           = aurelius_static_full   (strongest NON-weak by gp/$)
beats_fair_baseline     = True    (+35.03% gp/$)
pareto_sla_not_worse    = False   (mpc 0.0186 > fair 0.0141)
headline_claim_allowed  = False
```

**Reading it honestly.** Adding the fair batched baseline pulled the comparator up from 195,413
to 201,607 and the MPC's edge down from +39.3% (vs the non-batched baseline) to **+35.0%** — the
fairness correction the task requires. The edge is *real* and comes from the MPC running leaner
(91 GPU-hours vs 126: capacity 0.75× + batching concurrency), but it is **bought with a higher
violation rate** (0.0186 vs 0.0141), so the Pareto clause keeps the headline **False**. The
`sla_aware_capacity_1p5` row is the mirror image — the best SLA in the table (0.0079) at the
worst gp/$ (193 GPU-hours) — confirming `capacity_multiplier` is a genuine cost↔SLA trade, not a
free win in either direction. Artifact: `data/external/mpc_controller/evaluation_report.json`.

---

## 6. Honesty contract checklist

- [x] Each connected action **changes simulator output** (direct unit tests, §2).
- [x] Each changes **reward/metrics** (incremental eval moves gp/$ and SLA, §4).
- [x] **Tests prove the effect** + the no-op defaults + the not-free cost (§2).
- [x] **Fair baselines** exist for each lever (batched + over-provisioned operators, §5) — the
      MPC must beat an operator who already batches/provisions, not "discover" a switch.
- [x] **Fidelity/provenance documented**: capacity_multiplier calibrated to the existing sizing
      model; batching **INFERRED** (public-prior magnitudes, sanity-banded so aggressive raises
      violations).
- [x] **Pareto-aware claim gate** unchanged and still **honestly FALSE** on the headline.
- [x] **No fake knobs**: non-connected surfaces never touch replay kwargs
      (`test_action_surface.py`); deferred actions remain PLANNED/SIMULATED_ONLY.
- [x] **No connected knob silently excluded** from the search; freezing is explicit-with-reason.

*All numbers are SIMULATED — directional simulator evidence on public traces (Azure 2024 serving
week, Mooncake KV, v2026 fleet), not production telemetry.*
