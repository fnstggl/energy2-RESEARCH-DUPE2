# Sub-Hour MPC Action-Value Test

**Question (the only one this doc answers).** Does prewarm / migration / placement become *valuable*
when the receding-horizon MPC (PR #103) acts at **sub-hour** control intervals instead of hourly?

**One-line verdict (from the completed evidence).** **No — not down to 5-minute control.** Across
control intervals **dt ∈ {3600, 900, 300}s** and horizons **H ∈ {1, 2, 4, 8, 12, 24}** (2,160 held-out
decisions on the one-week Azure trace), the MPC selects **prewarm 0 times, migration 0 times, and
leaves placement topology-blind in all but 4 of 2,160 decisions**. Its only gp/$ lever remains
under-provisioning (`capacity_multiplier=0.75`) + KV-aware routing, and that edge is **never
Pareto-safe** (SLA is worse than the fair baseline at every dt and every H → claim gate `False`
everywhere). The one regime that could change this — **dt < the 300 s warm-idle timeout (dt = 60 s)**,
where a warmed pool first survives across control steps — **did not finish computing and is the single
decisive cell still open** (see *Smallest next diagnostic*). This run was stopped deliberately to
extract the answer rather than keep computing.

> Scope honesty: this is **SIMULATED** directional evidence on a calibrated world model, not production
> telemetry. No calibration was changed and no controller knob was added or tuned to force a result
> (one genuine bug was fixed — see below).

---

## What was run

`scripts/sweep_mpc_horizon.py` re-bins the **same** one-week Azure conv trace (27.3 M requests) at each
control interval `dt` (`period_seconds = dt`, `cycle_len = 86400/dt`), trains the forecaster ladder on
the pre-eval week, and runs the world-state MPC on a held-out tail — committing only the first action
each interval, re-planning every step. Because the request sample stride is global, the **arrival rate
is dt-invariant** (≈ 0.48 req/s median at every dt; verified), so the only thing that varies across
rows is *how often the controller acts*, not the load. Fair baseline = `aurelius_canonical_kv_routing`
(a strong static operator: backlog-aware capacity, conformal ordering, class-aware admission, KV-aware
routing — it does **not** prewarm/place/migrate).

- Eval window: a fixed **24 h real-time** slice (full diurnal, ramps included) at every dt, so the
  comparison is apples-to-apples. Decisions per dt are capped at **240** for tractability at fine dt,
  so dt=300 covers the most-recent 20 h (still a full diurnal). The cap and the real span are reported
  per row in the artifact.
- Artifact: `data/external/mpc_controller/mpc_subhour_action_value.json` (checkpointed per dt).
- Risk weight fixed at the established **0.3** (not tuned).

### Completed cells

| dt (control) | lookahead at H=24 | eval decisions | status |
|---|---|---|---|
| 3600 s (hourly) | 24 h | 24 × 6 = 144 | ✅ complete |
| 900 s (15-min) | 6 h | 96 × 6 = 576 | ✅ complete |
| 300 s (5-min) | 2 h | 240 × 6 = 1440 | ✅ complete |
| 60 s (1-min) | 24 min | (1440 × 6) | ❌ **not finished — decisive cell** |

---

## Results (completed dts)

Each row is the held-out MPC at that horizon. `gp/$` Δ is vs the fair baseline for that dt. **Gate** =
`beats_fair / pareto_sla_not_worse / headline_allowed`.

### dt = 3600 s — hourly (fair: gp/$ 94 339, SLA 0.0143, q_p95 9.50 s, GPU-h 31.6)

| H | look | gp/$ | Δ% | SLA | q_p95 | q_p99 | GPU-h | prewarm | place | migr | cap | route | rt/dec | gate |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 1 h | 94 746 | +0.4 | 0.0190 | 4.66 | 9.58 | 31.0 | off | blind | off | 0.75 | mixed | 0.12 s | T/F/F |
| 2 | 2 h | 94 549 | +0.2 | 0.0183 | 4.54 | 9.38 | 31.2 | off | blind×23,rack×1 | off | 0.75 | mixed | 0.19 s | T/F/F |
| 4 | 4 h | 96 086 | +1.9 | 0.0175 | 4.77 | 9.60 | 30.8 | off | blind×23,rack×1 | off | 0.75 | kv×21 | 0.31 s | T/F/F |
| 8 | 8 h | 96 710 | +2.5 | 0.0158 | 4.57 | 9.31 | 30.8 | off | blind×23,rack×1 | off | 0.75 | kv×24 | 0.58 s | T/F/F |
| 12 | 12 h | 96 644 | +2.4 | 0.0159 | 4.82 | 9.61 | 30.8 | off | blind×23,rack×1 | off | 0.75 | kv×24 | 0.80 s | T/F/F |
| 24 | 24 h | 97 349 | +3.2 | 0.0156 | 4.83 | 9.85 | 30.5 | off | blind | off | 0.75 | kv×24 | 1.56 s | T/F/F |

### dt = 900 s — 15-minute (fair: gp/$ 91 881, SLA 0.0152, q_p95 9.47 s, GPU-h 32.2)

| H | look | gp/$ | Δ% | SLA | GPU-h | prewarm | place | migr | gate |
|--|--|--|--|--|--|--|--|--|--|
| 1 | 15 m | 93 473 | +1.7 | 0.0177 | 31.4 | off | blind | off | T/F/F |
| 2 | 30 m | 93 653 | +1.9 | 0.0176 | 31.4 | off | blind | off | T/F/F |
| 4 | 60 m | 94 033 | +2.3 | 0.0169 | 31.4 | off | blind | off | T/F/F |
| 8 | 120 m | 94 451 | +2.8 | 0.0167 | 31.3 | off | blind | off | T/F/F |
| 12 | 180 m | 94 527 | +2.9 | 0.0165 | 31.3 | off | blind | off | T/F/F |
| 24 | 360 m | 94 562 | +2.9 | 0.0160 | 31.3 | off | blind | off | T/F/F |

### dt = 300 s — 5-minute (fair: gp/$ 89 900, SLA 0.0143, q_p95 9.54 s, GPU-h 28.1)

| H | look | gp/$ | Δ% | SLA | q_p95 | q_p99 | GPU-h | prewarm | place | migr | cap | route | rt/dec | gate |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 5 m | 92 732 | +3.2 | 0.0183 | 4.83 | 9.41 | 27.0 | off | blind | off | 0.75 | kv×191 | 0.13 s | T/F/F |
| 2 | 10 m | 92 967 | +3.4 | 0.0180 | 4.88 | 9.46 | 26.9 | off | blind | off | 0.75 | kv×205 | 0.19 s | T/F/F |
| 4 | 20 m | 93 492 | +4.0 | 0.0174 | 4.66 | 9.16 | 26.9 | off | blind | off | 0.75 | kv×227 | 0.32 s | T/F/F |
| 8 | 40 m | 93 930 | +4.5 | 0.0165 | 4.51 | 8.90 | 26.8 | off | blind | off | 0.75 | kv×240 | 0.59 s | T/F/F |
| 12 | 60 m | 93 918 | +4.5 | 0.0167 | 4.69 | 9.16 | 26.8 | off | blind | off | 0.75 | kv×240 | 0.83 s | T/F/F |
| 24 | 120 m | 93 942 | +4.5 | 0.0165 | 4.76 | 9.28 | 26.8 | off | blind | off | 0.75 | kv×240 | 1.67 s | T/F/F |

Batching mix: `conservative` in ≥ 98 % of decisions at every dt (a handful of `balanced`). Capacity:
`0.75` in **100 %** of decisions at every dt.

---

## The mechanism — why the completed range shows nothing

Warm persistence in the world simulator is **time-based**: an idle replica stays warm only while
`idle_steps × dt < WARM_IDLE_TIMEOUT_S` (calibrated **300 s**, `world_simulator.py:_advance`). For a
warmed/idle replica to **survive to the next control decision**, the step must be shorter than the
timeout, i.e. **dt < 300 s**. Every completed cell has **dt ≥ 300 s**, so in all of them the warm pool
**cools every step** — there is no cross-step state for prewarm or migration to pay off through, which
is exactly why the MPC's rollout never prefers them (no heuristic bonus props them up; they must earn
it through simulated future consequences, and at dt ≥ timeout there are none). dt = 300 s sits *on* the
boundary (after one idle step `idle_s = 300`, and the test is strict `< 300`, so it still cools). The
**first** interval at which a warmed pool persists across steps is **dt = 60 s** (idle 60/120/180/240 s
all `< 300` → survives up to 4 steps) — the cell that did not finish.

So the completed evidence is not a null result about the controller; it is a *consistent* result that
**"sub-hour" is not the relevant threshold — "sub-(warm-timeout)" is**, and the sweep stopped one rung
above it.

---

## Answers to the required questions (from completed evidence)

1. **Does prewarm become valuable?** No, down to 5-min control. Selected in **0 / 2 160** decisions.
   The benefit channel (a warm pool surviving to the next decision) does not exist at dt ≥ 300 s.
2. **Does migration become valuable?** No. Selected in **0 / 2 160** decisions. Same reason — plus its
   amortized move cost has nothing to pay back when relocated state doesn't persist a step.
3. **Does placement become more valuable?** No meaningful change. `topology_blind` in **2 156 / 2 160**
   decisions; the only departures are 4 single-period `rack_local` picks at hourly dt — noise, not a
   trend, and never `network_aware`.
4. **Is any improvement Pareto-safe?** **No.** Every dt × H beats fair on gp/$ (+0.2 % … +4.5 %) but
   has a **worse** SLA-violation rate (e.g. dt=300 H=24: 0.0165 vs fair 0.0143). `headline_allowed` is
   **False** in 100 % of rows.
5. **Is any result just buying SLA with more GPU-hours?** It is the *opposite*: the gp/$ edge is bought
   by **shedding SLA via under-provisioning** (`capacity_multiplier=0.75` in every decision), which
   *lowers* GPU-hours (e.g. dt=300: 26.8 vs fair 28.1). So the "win" is cheaper-not-better — precisely
   what the Pareto clause of the gate is designed to reject, and does.
6. **What dt/H gives the best Pareto-safe gp/$?** **None in the completed range** — no row is
   Pareto-safe. (Best raw gp/$ Δ is dt=300/H=8 at +4.5 %, but SLA 0.0165 > 0.0143 → not safe.)
7. **Is runtime acceptable?** Per-decision runtime is fine (0.12 s at H=1 → ~1.6 s at H=24, flat across
   dt). The **sweep** was slow for orchestration reasons, not per-decision cost — see below.

---

## Why the sweep was slow (and what was fixed)

- **Full-trace re-read per dt.** `build_mpc_inputs` streams all **27.3 M** trace rows once per dt
  (~70–120 s each) to re-bin at that resolution — 4 reads dominate fixed overhead.
- **Decisions × horizons × rollout.** Each decision is a coordinate-descent search over the connected
  bundle space × an H-step world rollout (`O(candidates × H)` world-steps; H=24 ≈ 1.6 s/decision). At
  fine dt the eval has many periods, so cells multiply: the **first run used an uncapped 24 h span →
  dt=60 alone was 1440 decisions × 6 horizons (~40 min)** and, worse, **wrote no checkpoint**, so a
  kill lost everything.
- **Fixes applied this PR (tooling only, no calibration change):** the sweep now **checkpoints each dt
  block as it completes** and **resumes** a matching config, flushes progress to stdout, and accepts
  `--max-eval-periods` to cap the finest dt. The three completed dts here were preserved precisely
  because of that checkpointing.

### The one real bug found and fixed

The held-out eval replay (`run_period_episode`) called `simulate_period(mutate=True)` **without**
threading `dt_seconds`, so it defaulted to 3600 s and advanced warm state **as if every step were an
hour, regardless of the control interval** — while the controller's *planning* rollout already passed
`dt_seconds=period_seconds`. Planning ≠ eval: at sub-hour dt the eval would never reflect warm
persistence, making any sub-hour test measure nothing. Fixed by threading `dt_seconds=period_seconds`
into the eval call. Regression test `test_eval_replay_threads_dt_into_warm_persistence` fails without
the fix (warm pool cools identically at 60 s and 3600 s) and passes with it (60 s preserves the pool,
3600 s cools it). This is a correctness fix, not a calibration change.

(Also added: `queue_delay_p99` to `EpisodeReport`/`PeriodOutcome`, matching the p99 already reported by
`serving_plane` / `optimizer_adapter` — a metric the test asks for, not a control knob.)

---

## Smallest next diagnostic (recommended, **not run**)

Run **exactly one cell: dt = 60 s**, the only interval with dt < the 300 s warm timeout, where a warmed
pool first persists across steps — i.e. the first regime in which prewarm/migration *can* have a
multi-period benefit. Keep it tiny and observable:

```
python -m scripts.sweep_mpc_horizon --dt-seconds 60 --horizons 1,4,12,24 \
    --eval-span-hours 6 --max-eval-periods 360 --risk-weight 0.3
```

≈ 360 decisions × 4 horizons over a 6 h diurnal slice (one ramp), ~8–12 min, checkpointed. Read its
`prewarm_mix` / `migration_mix` / `placement_mix` and the gate. Two outcomes, both informative:
- **They turn on at dt=60 and the gate goes True** → sub-(warm-timeout) control is what unlocks the
  deferred actions; the hourly machinery was correct but starved of cross-step persistence.
- **They stay off even at dt=60** → on this calibration/load the warm-seeded reactive baseline already
  keeps enough warm that prewarming/migration never pay their cost — a deeper conclusion than "wrong
  interval," and the honest stopping point.

A useful follow-up either way: add `world_static_best` / `prewarm_always` as extra gate arms so a
sub-hour win is judged against a *static stateful* operator, not only the no-stateful fair baseline.

---

## Files

- `scripts/sweep_mpc_horizon.py` — `--dt-seconds` (re-bin per control interval), `--eval-span-hours`,
  `--max-eval-periods`, per-dt checkpoint/resume.
- `aurelius/environment/training.py` — `build_mpc_inputs(control_dt_seconds=…)` re-bins the week at a
  sub-hour interval (`cycle_len = 86400/dt`); backward-compatible default (None → hourly).
- `aurelius/environment/controller.py` — **bug fix** (thread `dt_seconds` into the eval replay);
  `queue_delay_p99`.
- `aurelius/environment/world_simulator.py` — `queue_delay_p99` on `PeriodOutcome`.
- `tests/test_multi_period_mpc.py` — `test_eval_replay_threads_dt_into_warm_persistence` (regression
  guard for the bug) + `test_rebinning_changes_control_interval_not_hours`.
- `data/external/mpc_controller/mpc_subhour_action_value.json` — checkpointed results (3 dts).
