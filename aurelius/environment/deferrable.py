"""DeferrableWorkState — persistent shiftable (batch/offline/training) work for price-aware scheduling.

Serving requests are latency-bound; *deferrable* work (batch inference, embeddings, fine-tuning, maintenance)
can move in time to run when electricity is cheap, as long as deadlines are met. Aurelius has no real
deferrable-work trace (see research/ELECTRICITY_PRODUCTION_REALISM_AUDIT.md), so this is a **conservative,
SIMULATOR_INFERENCE** generator + a persistent pool the MPC schedules against. It must never create fake
savings:

  * work is CONSERVED — a delayed job stays in the pool until it runs or misses its deadline (never deleted);
  * a missed deadline is PENALISED (shifting cannot dodge work for free);
  * serving SLA DOMINATES — deferrable work runs only on spare capacity and adds its energy × price to cost;
  * with a FLAT price, price-aware shifting and run-asap produce the SAME cost (no fake shifting value).

Effects flow only through timing → GPU-seconds → energy_kWh → × electricity_price → cost. No reward bonus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WORKLOAD_TYPES = ("batch_inference", "embeddings", "fine_tuning", "maintenance")
STATUS = ("waiting", "running", "completed", "missed_deadline")


@dataclass
class DeferrableJob:
    """One shiftable job. Periods are control-step indices (deadlines are integer periods)."""
    job_id: str
    workload_type: str
    arrival_period: int
    deadline_period: int
    est_gpu_seconds: float
    est_energy_kwh: float
    priority: int = 1
    region_eligibility: tuple = ()             # regions the job may run in ("" = any)
    time_shiftable: bool = True
    region_shiftable: bool = False             # region shift is OUT OF SCOPE this PR (no multi-region fleet)
    status: str = "waiting"
    ran_period: int = -1
    price_paid: float = 0.0                    # $/kWh actually paid when it ran
    missed_deadline_penalty: float = 0.0       # $ charged if the deadline is missed
    provenance: str = "SIMULATOR_INFERENCE"

    @property
    def latest_safe_start(self) -> int:
        return self.deadline_period             # est_gpu_seconds ≤ one period → must START by the deadline


@dataclass
class DeferrableWorkState:
    """Persistent deferrable pool + the cost/conservation ledger (clones via deepcopy on CanonicalWorldState)."""
    jobs: list = field(default_factory=list)            # DeferrableJob, persists across periods
    completed: int = 0
    missed: int = 0
    shifted: int = 0                                     # jobs that ran later than arrival (time-shifted)
    energy_kwh_run: float = 0.0
    electricity_cost: float = 0.0                        # cumulative $ for deferrable energy
    missed_penalty_cost: float = 0.0

    def waiting(self) -> list:
        return [j for j in self.jobs if j.status == "waiting"]

    def to_dict(self) -> dict:
        return {"n_jobs": len(self.jobs), "waiting": len(self.waiting()), "completed": self.completed,
                "missed": self.missed, "shifted": self.shifted, "energy_kwh_run": round(self.energy_kwh_run, 4),
                "electricity_cost": round(self.electricity_cost, 5),
                "missed_penalty_cost": round(self.missed_penalty_cost, 5)}


def generate_deferrable_pool(n_jobs: int, *, horizon_periods: int, base_energy_kwh: float = 0.5,
                             seed: int = 0) -> DeferrableWorkState:
    """Conservative deterministic synthetic deferrable pool (SIMULATOR_INFERENCE).

    Modest per-job energy and realistic deadline slack (each job can shift a few periods). Deterministic
    (seeded by index — no RNG that would break replay). Not tuned to create savings: energies and slack are
    uniform and conservative, and the missed-deadline penalty is set to the cost of running at the WORST
    (p100-ish) price so dodging a deadline is never cheaper than running.
    """
    jobs = []
    for i in range(n_jobs):
        arrival = (i * 3) % max(1, horizon_periods // 2)        # spread arrivals over the first half
        slack = 2 + (i % 4)                                     # 2..5 periods of deadline slack
        wl = WORKLOAD_TYPES[i % len(WORKLOAD_TYPES)]
        energy = base_energy_kwh * (1.0 + 0.5 * (i % 3))        # 0.5 / 0.75 / 1.0 × base
        jobs.append(DeferrableJob(
            job_id=f"def{i}", workload_type=wl, arrival_period=arrival,
            deadline_period=arrival + slack, est_gpu_seconds=energy * 3600.0 / 0.7,  # ~0.7 kW draw
            est_energy_kwh=round(energy, 4), priority=1 + (i % 3),
            missed_deadline_penalty=round(energy * 5.0, 4),     # ≥ worst-case run cost → no free dodging
            provenance="SIMULATOR_INFERENCE"))
    return DeferrableWorkState(jobs=jobs)


def schedule_deferrable(state: DeferrableWorkState, *, period: int, prices: dict, policy: str,
                        gpu_seconds_available: float) -> dict:
    """Advance the deferrable pool ONE period under ``policy`` (mutates ``state``). Returns the period delta.

    Policies (deadline safety dominates price in ALL of them — a job at its deadline runs regardless of price):
      * ``off``         — run nothing (work waits; deadlines still enforced → may miss + be penalised).
      * ``asap``        — run every eligible waiting job now (price-blind), capacity permitting.
      * ``price_aware`` — run a job now iff FORCED (deadline reached) OR now is the cheapest remaining period in
                          its deadline window (no strictly-cheaper period ahead). Causal: day-ahead prices for
                          the deadline window are known. With FLAT prices nothing is ever cheaper ahead → runs
                          immediately ≡ asap → identical cost (no fake shifting value).

    Work is conserved: a job past its deadline that never ran is marked missed and penalised, never deleted.
    """
    ran = energy = cost = 0.0
    shifted = missed = completed = 0
    budget = gpu_seconds_available
    now_price = prices.get(period, 0.06)
    for j in state.jobs:
        if j.status != "waiting":
            continue
        if period > j.deadline_period:                          # deadline blew past unmet → penalise, conserve
            j.status = "missed_deadline"
            state.missed += 1
            state.missed_penalty_cost += j.missed_deadline_penalty
            missed += 1
            continue
        if period < j.arrival_period:
            continue
        forced = period >= j.deadline_period                    # last chance to start
        cheaper_ahead = any(prices.get(q, now_price) < now_price
                            for q in range(period + 1, j.deadline_period + 1))
        if policy == "off":
            run = False
        elif policy == "asap":
            run = True
        else:                                                   # price_aware
            run = forced or not cheaper_ahead                   # run when now is the cheapest remaining option
        if run and budget >= j.est_gpu_seconds:
            j.status = "completed"
            j.ran_period = period
            j.price_paid = now_price
            budget -= j.est_gpu_seconds
            c = j.est_energy_kwh * now_price
            ran += j.est_gpu_seconds
            energy += j.est_energy_kwh
            cost += c
            state.completed += 1
            state.energy_kwh_run += j.est_energy_kwh
            state.electricity_cost += c
            completed += 1
            if period > j.arrival_period:
                state.shifted += 1
                shifted += 1
    return {"ran_gpu_seconds": round(ran, 1), "energy_kwh": round(energy, 4),
            "electricity_cost": round(cost, 5), "completed": completed, "shifted": shifted, "missed": missed}


def run_deferrable_episode(state: DeferrableWorkState, *, periods, prices, spare_by_period,
                           policy="price_aware") -> dict:
    """Drive the deferrable pool across ``periods`` under ``policy`` (mutates ``state``). Serving dominates:
    ``spare_by_period[p]`` is the GPU-seconds left for deferrable work after serving (0 ⇒ no room → defer).

    A final sweep marks any job still waiting past its deadline as missed (work is conserved). Returns the
    aggregate ledger.
    """
    for p in sorted(periods):
        schedule_deferrable(state, period=p, prices=prices, policy=policy,
                            gpu_seconds_available=spare_by_period.get(p, 0.0))
    last = max(periods) if periods else 0
    for j in state.jobs:                                # final sweep: unmet deadlines are missed + penalised
        if j.status == "waiting" and last >= j.deadline_period:
            j.status = "missed_deadline"
            state.missed += 1
            state.missed_penalty_cost += j.missed_deadline_penalty
    avg_price = (sum(j.price_paid for j in state.jobs if j.status == "completed") /
                 max(1, state.completed)) if state.completed else 0.0
    return {**state.to_dict(), "policy": policy, "avg_price_paid": round(avg_price, 5)}
