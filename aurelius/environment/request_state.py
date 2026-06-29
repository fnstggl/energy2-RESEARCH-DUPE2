"""RequestState + RequestLifecycleState — canonical per-request lifecycle, promoted from the replay.

Today per-request state lives only inside `unified_replay.run_unified_replay` as ephemeral `Job` records
(arrival_s / admit_s / start_s / done_s) that vanish when the period ends — there is no persistent canonical
request lifecycle (see `CANONICAL_STATE_COVERAGE_AUDIT.md`). This module **promotes** that lifecycle into a
clone-safe canonical record + the consolidated queue summary it enables. It does NOT re-implement the replay's
scheduling/queue logic (the replay heap stays authoritative); it records the outcome.

Provenance: request *identity* (arrival, prompt/output tokens, class) is TRACE_DERIVED (Azure/Mooncake); the
fine-grained lifecycle *timestamps* (queue-entry / dispatch / prefill / decode) are SIMULATOR_INFERENCE when not
in the trace — labelled, never fabricated as measured. The conservation invariant
`arrived = queued + running + completed + dropped` is enforced (`state_validation.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

LIFECYCLE = ("arrived", "queued", "admitted", "dispatched", "prefill", "decode",
             "completed", "dropped", "missed_sla")


@dataclass
class RequestPlacement:
    replica_id: str = ""
    gpu_id: str = ""
    server_id: str = ""
    rack_id: str = ""


@dataclass
class RequestState:
    """One request's persistent lifecycle. Persists until completed/dropped; clone-safe (plain dataclass)."""
    request_id: str
    arrival_time: float
    period: int
    workload_type: str = "latency_critical"
    prompt_length: int = 0
    output_length: int = 0
    sla_target_s: float = 0.0
    deadline: float = 0.0
    priority: int = 1
    model_id: str = ""
    region: str = ""
    placement: RequestPlacement = field(default_factory=RequestPlacement)
    # lifecycle timestamps (SIMULATOR_INFERENCE unless trace-derived; -1 = not yet reached)
    queue_entry_time: float = -1.0
    admitted_time: float = -1.0
    dispatch_time: float = -1.0
    completion_time: float = -1.0
    status: str = "arrived"
    estimated_runtime_s: float = 0.0
    remaining_runtime_s: float = 0.0
    provenance: str = "TRACE_DERIVED identity; SIMULATOR_INFERENCE lifecycle"

    @property
    def latest_safe_start(self) -> float:
        return self.deadline - self.estimated_runtime_s

    def to_dict(self) -> dict:
        return {"request_id": self.request_id, "period": self.period, "status": self.status,
                "arrival_time": round(self.arrival_time, 4), "prompt_length": self.prompt_length,
                "output_length": self.output_length, "sla_target_s": round(self.sla_target_s, 4),
                "completion_time": round(self.completion_time, 4),
                "placement": vars(self.placement), "provenance": self.provenance}


@dataclass
class RequestLifecycleState:
    """Persistent pool of RequestState. Promoted from the per-period requests + the realised serving outcome.
    Conservation: arrived == queued + running + completed + dropped (checked in state_validation)."""
    requests: dict = field(default_factory=dict)        # request_id -> RequestState
    arrived: int = 0
    completed: int = 0
    dropped: int = 0
    missed_sla: int = 0

    def ingest_period(self, period: int, recs: list, *, sla_s: float, completed_ids: set | None = None,
                      sla_safe_frac: float = 1.0, period_t0: float | None = None) -> int:
        """Promote one period's requests (recs = [(arrival, output_tokens, prompt_tokens?), ...]) into canonical
        RequestState records with their realised terminal status. `completed_ids` (if given) is exact; otherwise
        the first `sla_safe_frac` of requests are marked completed and the rest missed_sla (SIMULATOR_INFERENCE,
        aggregate-consistent with the period's SLA-safe goodput). Returns the count ingested."""
        recs = sorted(recs, key=lambda r: r[0])
        t0 = period_t0 if period_t0 is not None else (recs[0][0] if recs else 0.0)
        n = len(recs)
        n_safe = (len(completed_ids) if completed_ids is not None
                  else int(round(max(0.0, min(1.0, sla_safe_frac)) * n)))
        for i, r in enumerate(recs):
            rid = f"p{period}_r{i}"
            out_tok = int(r[1])
            prompt = int(r[2]) if len(r) > 2 else out_tok
            safe = (rid in completed_ids) if completed_ids is not None else (i < n_safe)
            rs = RequestState(
                request_id=rid, arrival_time=float(r[0]), period=int(period), prompt_length=prompt,
                output_length=out_tok, sla_target_s=float(sla_s), deadline=float(r[0]) + float(sla_s),
                queue_entry_time=float(r[0]), admitted_time=float(r[0]), dispatch_time=float(r[0] - t0),
                status="completed" if safe else "missed_sla")
            rs.completion_time = rs.dispatch_time + (sla_s * (0.5 if safe else 1.5))   # inferred, label says so
            self.requests[rid] = rs
            self.arrived += 1
            if safe:
                self.completed += 1
            else:
                self.missed_sla += 1
                self.dropped += 1                        # a missed-SLA request leaves the pool (terminal)
        return n

    # -- conservation + queue consolidation ----------------------------------
    def running(self) -> int:
        return sum(1 for r in self.requests.values() if r.status in ("queued", "admitted", "dispatched",
                                                                      "prefill", "decode", "arrived"))

    def conserved(self) -> bool:
        """arrived == running + completed + dropped (every arrived request is in exactly one terminal/active set)."""
        return self.arrived == self.running() + self.completed + self.dropped

    def queue_summary(self) -> dict:
        """Consolidated QueueState view built from RequestState (resolves the world_state.QueueState placeholder):
        backlog, class mix, SLA-slack distribution — single authoritative summary, derived not duplicated."""
        active = [r for r in self.requests.values() if r.status in ("arrived", "queued", "admitted")]
        cls_mix: dict = {}
        for r in self.requests.values():
            cls_mix[r.workload_type] = cls_mix.get(r.workload_type, 0) + 1
        return {"backlog": len(active), "arrived": self.arrived, "completed": self.completed,
                "missed_sla": self.missed_sla, "dropped": self.dropped, "class_mix": cls_mix,
                "completion_rate": round(self.completed / max(1, self.arrived), 4)}

    def to_dict(self, *, max_requests: int = 50) -> dict:
        return {"n_requests": len(self.requests), "arrived": self.arrived, "completed": self.completed,
                "dropped": self.dropped, "missed_sla": self.missed_sla, "conserved": self.conserved(),
                "queue_summary": self.queue_summary(),
                "sample": [r.to_dict() for r in list(self.requests.values())[:max_requests]]}


@dataclass
class RooflineRecord:
    """A persisted snapshot of the per-period roofline regime (promoted from PeriodOutcome.roofline_diag).
    Diagnostic + planning state — NOT a reward term. Folds the decode-phase classification in."""
    period: int
    gpu_type: str = ""
    precision: str = "bf16"
    decode_regime: str = ""                  # memory_bandwidth_bound | compute_bound
    phase_bottleneck: str = ""               # decode_phase_bound | prefill_phase_bound | mixed
    arithmetic_intensity: float = 0.0
    ridge_point: float = 0.0
    power_w: float = 0.0
    timing_model: str = "legacy_scalar"
    provenance: str = "SIMULATOR_INFERENCE (roofline)"

    @staticmethod
    def from_diag(period: int, diag: dict | None, *, gpu_type: str = "", power_w: float = 0.0) -> "RooflineRecord":
        diag = diag or {}
        return RooflineRecord(
            period=int(period), gpu_type=gpu_type or diag.get("gpu_type", ""),
            precision=diag.get("precision", "bf16"),
            decode_regime=diag.get("decode_regime", diag.get("roofline_regime", "")),
            phase_bottleneck=diag.get("phase_bottleneck", ""),
            arithmetic_intensity=float(diag.get("arithmetic_intensity", 0.0)),
            ridge_point=float(diag.get("ridge_point", 0.0)), power_w=float(power_w),
            timing_model=diag.get("timing_model", "legacy_scalar"))


__all__ = ["RequestState", "RequestLifecycleState", "RequestPlacement", "RooflineRecord", "LIFECYCLE"]
