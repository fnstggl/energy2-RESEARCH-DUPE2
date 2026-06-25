"""Unified replay engine (Phase 1b-A) — one closed-loop, one trace, one state.

This is the piece the combination search (:mod:`aurelius.optimizer.joint`) is NOT:
a **single discrete-event loop** in which every serving surface acts on **one
evolving cluster state**, so that one surface's decision MUTATES the inputs the
next surface sees. That closed loop is the structural prerequisite for
compounding — and, just as importantly, it is the instrument that lets us prove
*why* compounding does or does not appear on a given trace.

Open-loop vs closed-loop (the whole point)
-------------------------------------------
``joint.combination_search`` is **open-loop**: it pre-computes each lever's
schedule from the *entire* raw trace, offline, then prices the combination. The
capacity schedule for tick ``t`` cannot see the backlog that this tick's ordering
or admission decisions actually produced — because every schedule was frozen
before the simulation ran. Levers therefore cannot interact; the search can only
*add up* effects that were each measured in isolation. That is why "nothing
drives it in a loop yet" was the correct critique.

This engine is **closed-loop**: a single event heap carries arrivals,
completions and tick-boundary control events over one evolving state
(``_State``). At each tick the control plane observes the **live** queue depth,
in-flight work and realised history — the backlog that admission and ordering
*just* shaped — and only then sizes capacity for the next window. Admission
deferral shrinks the backlog the capacity controller sees; ordering changes how
fast the backlog drains; capacity changes how much of it clears. The three
levers finally share a state, so any interaction (compounding *or* substitutive)
can actually express itself.

Workload classes (the data lever)
---------------------------------
Each job carries a class label. ``latency_critical`` jobs are gated by the tight
SLA and are **never** deferred (flow control never gates latency-critical load —
matching ``frontier/admission.py``). ``best_effort`` jobs have a relaxed SLA and
**can** be deferred under live overload. On a single-class trace (every public
LLM-serving trace today) admission has nothing legal to defer, so it is inert —
this engine *measures* that, and the A/B against a class-augmented trace isolates
the no-compounding cause to **data structure**, not the loop.

Physics + pricing are shared with the rest of the system: service time is
``_service_time_s`` (TTFT + tokens·TPOT); capacity sizing uses the same Erlang-C
gate (``_min_safe_c``); the cost denominator is the pure on-demand
``sum(c[t])·tick_hr·GPU_HOUR_USD`` (no spot, no oracle). Deterministic — no
randomness, idx-stable tie-breaking. Directional simulator only — not production
savings (``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from ..benchmarks.forecasted_mcs import _min_safe_c
from ..benchmarks.srtf_serving_backtest import (
    GPU_HOUR_USD,
    _service_time_s,
)

# Workload classes.
CLASS_LATENCY = "latency_critical"
CLASS_BEST_EFFORT = "best_effort"

# Event-type ordering at equal timestamps: arrivals land, completions free
# servers, THEN the tick control plane observes the resulting state.
_EV_ARRIVAL = 0
_EV_COMPLETION = 1
_EV_TICK = 2


# ---------------------------------------------------------------------------
# Job + evolving state
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """One request flowing through the closed loop (warped/sim-time arrival)."""

    idx: int
    arrival_s: float
    actual_tokens: int
    predicted_tokens: float
    service_s: float
    cls: str = CLASS_LATENCY
    # runtime state (mutated by the loop)
    admit_s: float = -1.0
    start_s: float = -1.0
    done_s: float = -1.0
    deferred_ticks: int = 0


@dataclass
class _State:
    """The single evolving cluster state every surface reads and writes."""

    now: float = 0.0
    tick: int = 0
    c: int = 1                       # active server count this window
    busy: int = 0                    # servers currently serving
    wait_queue: list = field(default_factory=list)      # admitted, not dispatched
    defer_buffer: list = field(default_factory=list)     # best-effort, deferred
    # realised causal history (index t holds tick t's revealed values).
    # Sizing history is LATENCY-CRITICAL ONLY: the SLA tier is what on-demand
    # capacity is provisioned for; best-effort is backfill, never sized-for.
    hist_arrivals: list = field(default_factory=list)
    hist_service_s: list = field(default_factory=list)
    # accumulators within the current tick window
    _tick_arrivals: int = 0          # latency-critical arrivals this tick
    _tick_service_sum: float = 0.0
    _tick_service_n: int = 0

    def backlog(self) -> int:
        """Live admitted-but-unserved depth (all classes)."""
        return len(self.wait_queue)

    def lc_backlog(self) -> int:
        """Live LATENCY-CRITICAL backlog — the signal capacity reacts to."""
        return sum(1 for j in self.wait_queue if j.cls == CLASS_LATENCY)


# ---------------------------------------------------------------------------
# Control-plane surfaces (pluggable, deployable, causal)
# ---------------------------------------------------------------------------

class CapacityController:
    """Sizes ``c`` for the upcoming window from the LIVE state + causal history.

    Modes:
      * ``reactive_lag1``  — Erlang-C on last tick's realised arrivals/service.
      * ``forecasted_mcs`` — Erlang-C on an EWMA forecast of arrivals/service.
      * ``backlog_aware``  — forecast PLUS the live observed backlog (only
        expressible in a closed loop): provision enough to clear the backlog the
        other levers just produced within the SLA, and shed when idle.
    """

    def __init__(
        self,
        mode: str,
        *,
        tick_seconds: float,
        sla_s: float,
        mcs_gate: float = 9.5,
        warmup_c: int = 4,
        ewma_alpha: float = 0.5,
        drain_horizon_ticks: float = 3.0,
    ) -> None:
        self.mode = mode
        self.tick_seconds = tick_seconds
        self.sla_s = sla_s
        self.mcs_gate = mcs_gate
        self.warmup_c = warmup_c
        self.ewma_alpha = ewma_alpha
        self.drain_horizon_ticks = drain_horizon_ticks
        self._ewma_count: float | None = None
        self._ewma_svc: float | None = None

    def _update_ewma(self, count: int, mean_svc: float | None) -> None:
        a = self.ewma_alpha
        self._ewma_count = float(count) if self._ewma_count is None else (
            a * count + (1.0 - a) * self._ewma_count)
        if mean_svc is not None:
            self._ewma_svc = mean_svc if self._ewma_svc is None else (
                a * mean_svc + (1.0 - a) * self._ewma_svc)

    def decide(self, st: _State) -> int:
        # No history yet → deployable cold-start guess.
        if not st.hist_arrivals:
            return max(1, self.warmup_c)

        if self.mode == "reactive_lag1":
            arr_hat = float(st.hist_arrivals[-1])
            svc_hat = st.hist_service_s[-1] if st.hist_service_s[-1] > 0 else _service_time_s(1)
        else:  # forecasted_mcs / backlog_aware share the EWMA forecast
            arr_hat = self._ewma_count if self._ewma_count is not None else float(st.hist_arrivals[-1])
            svc_hat = self._ewma_svc if self._ewma_svc is not None else _service_time_s(1)

        if arr_hat <= 0.0 and st.backlog() == 0 and st.busy == 0:
            return 1
        lam = max(arr_hat, 0.0) / self.tick_seconds
        sla_wait = max(0.0, self.sla_s - svc_hat)
        c = _min_safe_c(lam, svc_hat, sla_wait, self.mcs_gate) if arr_hat > 0 else 1

        if self.mode == "backlog_aware":
            # Closed-loop term: add just enough replicas to DRAIN the live
            # LATENCY-CRITICAL backlog over a few ticks (not instantly — that
            # over-provisions and fights admission). Best-effort backlog is
            # backfill and never triggers scale-up, so deferring it (admission)
            # genuinely lets capacity stay lean instead of chasing it. This is the
            # standard multi-tier model and the reason the levers can compound.
            lc_back = st.lc_backlog()
            if lc_back > 0 and svc_hat > 0:
                horizon_s = max(1e-9, self.drain_horizon_ticks * self.tick_seconds)
                extra = lc_back * svc_hat / horizon_s
                c += int(-(-extra // 1))  # ceil
            c = max(c, st.busy)  # never below what in-flight work needs
        return max(1, c)


class AdmissionController:
    """Class-aware peak-shave flow control (deployable-correct).

    Decides only on **best-effort** load at tick boundaries; latency-critical is
    never gated. Under live overload (latency-critical backlog present or servers
    saturated) best-effort is deferred to a later tick; otherwise admitted. No job
    is dropped — deferral only moves timing/cost, which the SLA prices honestly.
    ``mode="off"`` admits everything immediately (no flow control).
    """

    def __init__(self, mode: str, *, overload_backlog: int = 2) -> None:
        self.mode = mode
        self.overload_backlog = overload_backlog

    def admit_on_arrival(self, st: _State, job: Job) -> bool:
        """Off, or latency-critical → admit immediately (bypass the tick gate)."""
        if self.mode == "off":
            return True
        return job.cls == CLASS_LATENCY

    def decide_deferred(self, st: _State) -> tuple[list, list]:
        """At a tick: split (buffered best-effort + carried) into (admit, defer)."""
        if self.mode == "off":
            return st.defer_buffer, []
        pool = st.defer_buffer
        # Defer batch ONLY while the latency-critical tier is genuinely backlogged
        # (a real burst). In valleys (no LC backlog) the buffer drains, so batch is
        # time-shifted into the troughs rather than starved — admitting it then is
        # free because class-priority dispatch already serves any LC work first.
        overloaded = st.lc_backlog() >= self.overload_backlog
        if overloaded:
            for j in pool:
                j.deferred_ticks += 1
            return [], list(pool)
        return list(pool), []


# ---------------------------------------------------------------------------
# Ordering — dispatch key from the wait queue
# ---------------------------------------------------------------------------

def _dispatch_index(st: _State, ordering: str) -> int:
    """Index into ``st.wait_queue`` of the next job to serve.

    Strict class priority: ``latency_critical`` is always served before
    ``best_effort`` (batch backfills) — standard multi-tier behaviour. Within the
    chosen class the ordering discipline (FIFO or abs-conformal SRPT) decides.
    """
    q = st.wait_queue
    pool = [i for i, j in enumerate(q) if j.cls == CLASS_LATENCY]
    if not pool:
        pool = list(range(len(q)))
    if ordering == "fifo":
        best, bk = pool[0], (q[pool[0]].admit_s, q[pool[0]].idx)
        for i in pool[1:]:
            k = (q[i].admit_s, q[i].idx)
            if k < bk:
                best, bk = i, k
        return best
    # abs_conformal SRPT on the causal predicted-token prior, aged by wait time.
    best, bk = pool[0], None
    for i in pool:
        j = q[i]
        aged = j.predicted_tokens / (1.0 + 0.05 * max(0.0, st.now - j.admit_s))
        k = (aged, j.idx)
        if bk is None or k < bk:
            best, bk = i, k
    return best


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class UnifiedKPI:
    """KPIs from one closed-loop run — comparable to forecasted_mcs.PolicyKPI."""

    capacity: str
    ordering: str
    admission: str
    levers_on: tuple
    n_ticks: int
    c_mean: float
    c_min: int
    c_max: int
    gpu_hours: float
    cost_usd: float
    sla_safe_goodput: float
    goodput_per_dollar: float
    n_total: int
    n_sla_safe: int
    sla_violations: int
    n_deferred: int
    backlog_peak: int
    c_trace: tuple = ()      # per-tick c (proof the loop actually coupled)

    @property
    def label(self) -> str:
        return "+".join(self.levers_on) if self.levers_on else "base"

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "ordering": self.ordering,
            "admission": self.admission,
            "levers_on": list(self.levers_on),
            "label": self.label,
            "n_ticks": self.n_ticks,
            "c_mean": round(self.c_mean, 4),
            "c_min": self.c_min,
            "c_max": self.c_max,
            "gpu_hours": round(self.gpu_hours, 4),
            "cost_usd": round(self.cost_usd, 4),
            "sla_safe_goodput": round(self.sla_safe_goodput, 2),
            "goodput_per_dollar": round(self.goodput_per_dollar, 2),
            "n_total": self.n_total,
            "n_sla_safe": self.n_sla_safe,
            "sla_violations": self.sla_violations,
            "n_deferred": self.n_deferred,
            "backlog_peak": self.backlog_peak,
        }


# ---------------------------------------------------------------------------
# The closed loop
# ---------------------------------------------------------------------------

def run_unified_replay(
    jobs: list,
    *,
    tick_seconds: float,
    sla_s: float,
    capacity: str = "reactive_lag1",
    ordering: str = "fifo",
    admission: str = "off",
    mcs_gate: float = 9.5,
    warmup_c: int = 4,
    best_effort_sla_s: float = 300.0,
) -> UnifiedKPI:
    """Run one configuration of all serving surfaces in ONE event loop.

    ``jobs`` are :class:`Job` with **sim-time** ``arrival_s`` (already warped).
    Returns a :class:`UnifiedKPI`. The capacity controller observes the live
    backlog produced by this run's ordering+admission — the structural difference
    from the open-loop combination search.
    """
    if not jobs:
        return UnifiedKPI(capacity, ordering, admission, (), 0, 0.0, 0, 0,
                          0.0, 1e-9, 0.0, 0.0, 0, 0, 0, 0, 0, ())

    jobs = sorted(jobs, key=lambda j: (j.arrival_s, j.idx))
    t_max = jobs[-1].arrival_s
    n_ticks = max(1, int(t_max / tick_seconds) + 1)

    st = _State(c=max(1, warmup_c))
    cap = CapacityController(capacity, tick_seconds=tick_seconds, sla_s=sla_s,
                             mcs_gate=mcs_gate, warmup_c=warmup_c)
    adm = AdmissionController(admission)

    # servers: sid -> (job, version) or None; version guards stale completions.
    servers: dict = {}
    s_ver: dict = {}

    events: list = []
    seq = [0]

    def _en() -> int:
        seq[0] += 1
        return seq[0]

    for j in jobs:
        heapq.heappush(events, (j.arrival_s, _EV_ARRIVAL, _en(), j))
    for k in range(n_ticks):
        heapq.heappush(events, (k * tick_seconds, _EV_TICK, _en(), k))

    c_per_tick: list = [0] * n_ticks
    backlog_peak = 0
    response: dict = {}

    def _free_sid() -> int | None:
        for s in range(st.c):
            if servers.get(s) is None:
                return s
        return None

    def _start(sid: int, job: Job, t: float) -> None:
        job.start_s = t
        servers[sid] = job
        s_ver[sid] = s_ver.get(sid, 0) + 1
        st.busy += 1
        heapq.heappush(events, (t + job.service_s, _EV_COMPLETION, _en(), (sid, s_ver[sid], job)))

    def _dispatch() -> None:
        nonlocal backlog_peak
        while st.wait_queue:
            sid = _free_sid()
            if sid is None:
                break
            i = _dispatch_index(st, ordering)
            job = st.wait_queue.pop(i)
            _start(sid, job, st.now)
        backlog_peak = max(backlog_peak, st.backlog())

    while events:
        t, ety, _, payload = heapq.heappop(events)
        st.now = t

        if ety == _EV_ARRIVAL:
            job = payload
            if job.cls == CLASS_LATENCY:   # sizing history is SLA-tier only
                st._tick_arrivals += 1
                st._tick_service_sum += job.service_s
                st._tick_service_n += 1
            if adm.admit_on_arrival(st, job):
                job.admit_s = t
                st.wait_queue.append(job)
                _dispatch()
            else:
                st.defer_buffer.append(job)

        elif ety == _EV_COMPLETION:
            sid, ver, job = payload
            if s_ver.get(sid) != ver:
                continue
            job.done_s = t
            response[job.idx] = t - job.arrival_s
            servers[sid] = None
            st.busy -= 1
            _dispatch()

        else:  # _EV_TICK — the control plane observes the live state
            k = payload
            st.tick = k
            # 1) admission decides on deferred/best-effort pool from live overload
            admit, defer = adm.decide_deferred(st)
            for job in admit:
                job.admit_s = t
                st.wait_queue.append(job)
            st.defer_buffer = defer
            # 2) capacity sizes the next window from the LIVE backlog + history
            st.c = cap.decide(st)
            c_per_tick[k] = st.c
            # 3) fold realised tick history for the next forecast
            mean_svc = (st._tick_service_sum / st._tick_service_n) if st._tick_service_n else None
            st.hist_arrivals.append(st._tick_arrivals)
            st.hist_service_s.append(mean_svc if mean_svc is not None else 0.0)
            cap._update_ewma(st._tick_arrivals, mean_svc)
            st._tick_arrivals = 0
            st._tick_service_sum = 0.0
            st._tick_service_n = 0
            # 4) dispatch under the (possibly new) capacity
            _dispatch()

    # Flush any never-admitted deferred best-effort at the final capacity so they
    # complete (counted honestly against their relaxed SLA).
    if st.defer_buffer:
        st.c = max(st.c, 1)
        for job in sorted(st.defer_buffer, key=lambda j: j.idx):
            job.admit_s = st.now
            st.wait_queue.append(job)
        st.defer_buffer = []
        guard = 0
        while st.wait_queue and guard < len(jobs) + 10:
            guard += 1
            sid = _free_sid()
            if sid is None:
                # advance to the earliest completion to free a server
                ev = heapq.heappop(events) if events else None
                if ev is None:
                    # no servers will free (shouldn't happen) — serve sequentially
                    job = st.wait_queue.pop(0)
                    st.now += job.service_s
                    job.start_s = job.done_s = st.now
                    response[job.idx] = st.now - job.arrival_s
                    continue
                t, ety, _, payload = ev
                st.now = t
                if ety == _EV_COMPLETION:
                    sid2, ver, job = payload
                    if s_ver.get(sid2) == ver:
                        job.done_s = t
                        response[job.idx] = t - job.arrival_s
                        servers[sid2] = None
                        st.busy -= 1
                continue
            i = _dispatch_index(st, ordering)
            job = st.wait_queue.pop(i)
            _start(sid, job, st.now)
        # drain remaining completions
        while events:
            t, ety, _, payload = heapq.heappop(events)
            if ety == _EV_COMPLETION:
                sid2, ver, job = payload
                if s_ver.get(sid2) == ver:
                    job.done_s = t
                    response[job.idx] = t - job.arrival_s
                    servers[sid2] = None

    # KPIs.
    gpu_hours = sum(c_per_tick) * tick_seconds / 3600.0
    cost = max(gpu_hours * GPU_HOUR_USD, 1e-9)

    def _sla_for(job: Job) -> float:
        return sla_s if job.cls == CLASS_LATENCY else best_effort_sla_s

    goodput = 0.0
    n_sla_safe = 0
    for job in jobs:
        r = response.get(job.idx)
        if r is not None and r <= _sla_for(job):
            goodput += job.actual_tokens
            n_sla_safe += 1
    n_total = len(jobs)
    n_deferred = sum(1 for j in jobs if j.deferred_ticks > 0)

    levers = tuple(x for x, on in (
        ("C", capacity == "forecasted_mcs" or capacity == "backlog_aware"),
        ("O", ordering == "abs_conformal"),
        ("A", admission != "off"),
    ) if on)

    return UnifiedKPI(
        capacity=capacity, ordering=ordering, admission=admission, levers_on=levers,
        n_ticks=n_ticks, c_mean=(sum(c_per_tick) / n_ticks) if n_ticks else 0.0,
        c_min=min(c_per_tick) if c_per_tick else 0,
        c_max=max(c_per_tick) if c_per_tick else 0,
        gpu_hours=gpu_hours, cost_usd=cost, sla_safe_goodput=goodput,
        goodput_per_dollar=goodput / cost, n_total=n_total, n_sla_safe=n_sla_safe,
        sla_violations=n_total - n_sla_safe, n_deferred=n_deferred,
        backlog_peak=backlog_peak, c_trace=tuple(c_per_tick),
    )


# ---------------------------------------------------------------------------
# Combination runner — the full closed-loop lattice + interaction verdict
# ---------------------------------------------------------------------------

# Lever options (off → on).
_CAP_OFF = "reactive_lag1"
_CAP_ON = "backlog_aware"
_ORD_OFF = "fifo"
_ORD_ON = "abs_conformal"
_ADM_OFF = "off"
_ADM_ON = "class_aware"


def _jobs_hash(jobs: list) -> str:
    """Deterministic content hash of the job stream (reproducibility stamp)."""
    import hashlib
    h = hashlib.sha256()
    for j in sorted(jobs, key=lambda j: (j.arrival_s, j.idx)):
        h.update(f"{j.arrival_s:.6f}|{j.actual_tokens}|{j.cls}\n".encode())
    return h.hexdigest()[:16]


@dataclass
class UnifiedLatticeResult:
    """Full 2×2×2 closed-loop lattice + the measured interaction verdict.

    The productized "joint loop" result: every lever combination run through ONE
    discrete-event loop on ONE evolving state, scored by ONE objective
    (:class:`ObjectiveLayer`), with the honest verdict on whether combining the
    levers COMPOUNDS (beats the best single lever) or is SUBSTITUTIVE.
    """

    trace_id: str
    jobs_hash: str
    n_jobs: int
    n_latency_critical: int
    n_best_effort: int
    sla_s: float
    tick_seconds: float
    denominator: str
    cells: list                 # list[UnifiedKPI]
    base_gpd: float
    best_single_label: str
    best_single_gpd: float
    best_overall_label: str
    best_overall_gpd: float
    best_multi_label: str
    best_multi_gpd: float
    compounding: bool
    interaction: str            # "compounding" | "substitutive" | "neutral"
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "jobs_hash": self.jobs_hash,
            "n_jobs": self.n_jobs,
            "n_latency_critical": self.n_latency_critical,
            "n_best_effort": self.n_best_effort,
            "sla_s": self.sla_s,
            "tick_seconds": self.tick_seconds,
            "denominator": self.denominator,
            "cells": [c.to_dict() for c in self.cells],
            "base_gpd": round(self.base_gpd, 4),
            "best_single": {"label": self.best_single_label,
                            "goodput_per_dollar": round(self.best_single_gpd, 4)},
            "best_multi": {"label": self.best_multi_label,
                           "goodput_per_dollar": round(self.best_multi_gpd, 4)},
            "best_overall": {"label": self.best_overall_label,
                             "goodput_per_dollar": round(self.best_overall_gpd, 4)},
            "compounding": self.compounding,
            "interaction": self.interaction,
            "notes": list(self.notes),
        }


def run_unified_combination(
    jobs: list,
    *,
    tick_seconds: float,
    sla_s: float,
    trace_id: str = "trace",
    mcs_gate: float = 9.5,
    notes=(),
) -> UnifiedLatticeResult:
    """Run the full closed-loop lever lattice on one job stream and measure it.

    This is the unified-engine analogue of ``joint.combination_search`` — but every
    cell is a genuine closed-loop run where capacity reacts to the live backlog
    that ordering+admission shaped (not a precomputed offline schedule). Scored by
    :class:`ObjectiveLayer`; priced on the pure on-demand denominator.
    """
    from .layers import ObjectiveLayer

    obj = ObjectiveLayer()
    cells: list = []
    import itertools
    for cap, order, adm in itertools.product(
        (_CAP_OFF, _CAP_ON), (_ORD_OFF, _ORD_ON), (_ADM_OFF, _ADM_ON)
    ):
        cells.append(run_unified_replay(
            jobs, tick_seconds=tick_seconds, sla_s=sla_s,
            capacity=cap, ordering=order, admission=adm, mcs_gate=mcs_gate,
        ))

    by_label = {c.label: c for c in cells}
    ranked = obj.compare({c.label: c.goodput_per_dollar for c in cells})
    best_overall = by_label[ranked[0][0]]
    base = next(c for c in cells if not c.levers_on)
    singles = [c for c in cells if len(c.levers_on) == 1]
    multis = [c for c in cells if len(c.levers_on) >= 2]
    best_single = max(singles, key=lambda c: c.goodput_per_dollar)
    best_multi = max(multis, key=lambda c: c.goodput_per_dollar)

    margin = best_multi.goodput_per_dollar - best_single.goodput_per_dollar
    rel = margin / best_single.goodput_per_dollar if best_single.goodput_per_dollar else 0.0
    if rel > 0.005:
        interaction, compounding = "compounding", True
    elif rel < -0.005:
        interaction, compounding = "substitutive", False
    else:
        interaction, compounding = "neutral", False

    return UnifiedLatticeResult(
        trace_id=trace_id, jobs_hash=_jobs_hash(jobs), n_jobs=len(jobs),
        n_latency_critical=sum(1 for j in jobs if j.cls == CLASS_LATENCY),
        n_best_effort=sum(1 for j in jobs if j.cls == CLASS_BEST_EFFORT),
        sla_s=sla_s, tick_seconds=tick_seconds, denominator="on_demand",
        cells=cells, base_gpd=base.goodput_per_dollar,
        best_single_label=best_single.label, best_single_gpd=best_single.goodput_per_dollar,
        best_overall_label=best_overall.label, best_overall_gpd=best_overall.goodput_per_dollar,
        best_multi_label=best_multi.label, best_multi_gpd=best_multi.goodput_per_dollar,
        compounding=compounding, interaction=interaction, notes=tuple(notes),
    )


__all__ = [
    "CLASS_LATENCY", "CLASS_BEST_EFFORT", "Job", "CapacityController",
    "AdmissionController", "UnifiedKPI", "run_unified_replay",
    "UnifiedLatticeResult", "run_unified_combination",
]
