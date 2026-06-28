"""PrefillDecodeSchedulerV2 — pools + phase queues + continuous batching (V2).

Ports the Splitwise/DistServe prefill/decode disaggregation pattern and the vLLM/Orca/Sarathi continuous-
batching pattern into an **analytical period scheduler** (occupancy + M/D/1 phase queues, not a per-iteration
event loop — labelled SIMULATOR_INFERENCE; sufficient for causal direction and economic consequence).

Serving modes:
  * ``shared_pool``          — every replica serves both phases (≈ V1 behaviour).
  * ``disaggregated_static`` — replicas split into a prefill pool and a decode pool by ``prefill_frac``; a KV
    handoff (bytes/bandwidth) is paid between phases.
  * ``disaggregated_sweep``  — the simulator evaluates several ``prefill_frac`` values and keeps the best.

Continuous batching: an effective batch is formed under a token budget (``max_num_batched_tokens``), an
active-sequence cap (``max_active_sequences``), and an HBM/KV-memory limit. Bigger batches amortise decode
weight traffic (the roofline `/batch`) but, past a saturation point or under HBM pressure, pay a queue/tail
penalty — so aggressive batching helps throughput and hurts tail latency under tight SLA. Chunked prefill
(Sarathi) reduces decode stalls by capping per-iteration prefill tokens, modelled as a reduction of the
prefill-induced decode-queue inflation.

Causal law: every term flows into TTFT / completion latency / queue wait / GPU-seconds / SLA — never reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SERVING_MODES = ("shared_pool", "disaggregated_static", "disaggregated_sweep")
ALLOCATIONS = {"shared": None, "p40_d60": 0.4, "p50_d50": 0.5, "p60_d40": 0.6,
               "p20_d80": 0.2, "p80_d20": 0.8}


@dataclass
class SchedRequest:
    arrival_s: float
    prompt_tokens: int
    output_tokens: int
    saved_prefill_tokens: int = 0       # from the tiered KV decision
    transfer_latency_s: float = 0.0     # KV transfer (tier hit) added to TTFT
    recompute: bool = False


@dataclass
class ServingResultV2:
    n: int
    ttft_s: list = field(default_factory=list)
    completion_s: list = field(default_factory=list)
    prefill_queue_wait_s: list = field(default_factory=list)
    decode_queue_wait_s: list = field(default_factory=list)
    prefill_gpu_seconds: float = 0.0
    decode_gpu_seconds: float = 0.0
    prefill_idle_gpu_seconds: float = 0.0
    decode_idle_gpu_seconds: float = 0.0
    kv_handoff_bytes: int = 0
    kv_handoff_latency_s: float = 0.0
    effective_batch_size: float = 0.0
    active_decode_sequences: float = 0.0
    decode_steps: int = 0
    prefill_chunks: int = 0
    admitted: int = 0
    dropped: int = 0
    phase_interference: float = 0.0
    allocation_efficiency: float = 0.0
    batching_regime: str = "unsaturated"
    sla_slack_s: float = 0.0

    def _pct(self, xs, q):
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(len(s) * q))]

    @property
    def realized_gpu_seconds(self) -> float:
        return self.prefill_gpu_seconds + self.decode_gpu_seconds

    def summary(self) -> dict:
        return {"n": self.n, "ttft_p50": round(self._pct(self.ttft_s, 0.5), 5),
                "ttft_p95": round(self._pct(self.ttft_s, 0.95), 5),
                "ttft_p99": round(self._pct(self.ttft_s, 0.99), 5),
                "completion_p50": round(self._pct(self.completion_s, 0.5), 5),
                "completion_p95": round(self._pct(self.completion_s, 0.95), 5),
                "completion_p99": round(self._pct(self.completion_s, 0.99), 5),
                "prefill_gpu_seconds": round(self.prefill_gpu_seconds, 4),
                "decode_gpu_seconds": round(self.decode_gpu_seconds, 4),
                "realized_gpu_seconds": round(self.realized_gpu_seconds, 4),
                "kv_handoff_bytes": self.kv_handoff_bytes,
                "kv_handoff_latency_s": round(self.kv_handoff_latency_s, 5),
                "effective_batch_size": round(self.effective_batch_size, 3),
                "active_decode_sequences": round(self.active_decode_sequences, 3),
                "decode_steps": self.decode_steps, "prefill_chunks": self.prefill_chunks,
                "admitted": self.admitted, "dropped": self.dropped,
                "phase_interference": round(self.phase_interference, 4),
                "allocation_efficiency": round(self.allocation_efficiency, 4),
                "batching_regime": self.batching_regime}


def _md1_wait(lam: float, service_s: float, servers: int) -> float:
    """M/D/1-per-server queue wait approximation (deterministic service). ρ=λ·D/c; Wq=ρ·D/(2(1−ρ))."""
    c = max(1, servers)
    if service_s <= 0:
        return 0.0
    rho = (lam * service_s) / c
    if rho >= 0.999:
        rho = 0.999
    return (rho * service_s) / (2.0 * (1.0 - rho))


@dataclass
class PrefillDecodeSchedulerV2:
    max_num_batched_tokens: int = 2048
    max_active_sequences: int = 64
    chunked_prefill: bool = True
    saturation_seqs: int = 48
    handoff_bw_gbps: float = 50.0       # KV handoff bandwidth (RDMA), disaggregated only

    def simulate(self, reqs, *, timing_model, n_replicas: int, serving_mode: str = "shared_pool",
                 prefill_frac: float | None = None, sla_s: float = 5.0, period_s: float = 60.0,
                 precision: str = "bf16", spec_decode: str = "off", clock: str = "base",
                 kv_bytes_per_token: int = 131072) -> ServingResultV2:
        """Run one period of ``reqs`` (sorted by arrival) under the action. Causal, deterministic."""
        reqs = sorted(reqs, key=lambda r: r.arrival_s)
        res = ServingResultV2(n=len(reqs))
        if not reqs:
            return res
        disagg = serving_mode != "shared_pool" and prefill_frac is not None
        if disagg:
            n_pf = max(1, int(round(n_replicas * prefill_frac)))
            n_dc = max(1, n_replicas - n_pf)
        else:
            n_pf = n_dc = n_replicas

        # effective batch under token budget + active-seq cap + HBM pressure
        mean_tokens = sum(r.prompt_tokens + r.output_tokens for r in reqs) / len(reqs)
        budget_batch = self.max_num_batched_tokens / max(1.0, mean_tokens)
        eff_batch = max(1.0, min(budget_batch, float(self.max_active_sequences)))
        res.effective_batch_size = eff_batch

        # first pass: per-request work via roofline at this batch
        prefill_work, decode_work = [], []
        sum_out = 0
        for r in reqs:
            remaining = max(0, r.prompt_tokens - r.saved_prefill_tokens)
            ctx = r.prompt_tokens + r.output_tokens // 2
            t = timing_model.estimate(prompt_tokens=r.prompt_tokens, output_tokens=r.output_tokens,
                                       prefill_tokens_remaining=remaining, context_tokens=ctx,
                                       batch=int(round(eff_batch)), active_sequences=int(round(eff_batch)),
                                       precision=precision, spec_decode=spec_decode, clock=clock)
            prefill_work.append(t.prefill_time_s)
            decode_work.append(t.decode_time_s)
            sum_out += r.output_tokens
            res.hbm_pressure = getattr(res, "hbm_pressure", 0.0)
        last_t = t  # keep last diagnostics

        # phase queues (M/D/1 per pool). λ = arrivals/period.
        lam = len(reqs) / max(period_s, 1e-9)
        mean_pf = sum(prefill_work) / len(reqs)
        mean_dc = sum(decode_work) / len(reqs)
        pf_wait = _md1_wait(lam, mean_pf, n_pf)
        dc_wait = _md1_wait(lam, mean_dc, n_dc)

        # batching saturation: past saturation_seqs or HBM pressure, tail penalty inflates decode queue
        active_seqs = (sum_out * mean_dc) / max(period_s, 1e-9)
        res.active_decode_sequences = active_seqs
        hbm_press = last_t.hbm_pressure
        sat = 1.0
        if active_seqs > self.saturation_seqs or hbm_press > 1.0:
            over = max(active_seqs / self.saturation_seqs, hbm_press)
            sat = 1.0 + 0.25 * (over - 1.0)
            res.batching_regime = "saturated"
        dc_wait *= sat

        # chunked prefill reduces prefill-induced decode stalls (interference) in shared pool
        interference = 0.0
        if serving_mode == "shared_pool":
            interference = min(0.5, mean_pf / max(mean_dc, 1e-9) * 0.1)
            if self.chunked_prefill:
                interference *= 0.4
                res.prefill_chunks = sum(max(1, (max(0, r.prompt_tokens - r.saved_prefill_tokens)
                                                 ) // max(1, self.max_num_batched_tokens) + 1) for r in reqs)
            dc_wait *= (1.0 + interference)
        res.phase_interference = interference

        # KV handoff (disaggregated): transfer the prompt KV prefill→decode pool
        handoff_lat = 0.0
        if disagg:
            for r in reqs:
                hb = max(0, r.prompt_tokens) * kv_bytes_per_token
                res.kv_handoff_bytes += hb
                handoff_lat += hb / (self.handoff_bw_gbps * 1e9)
            res.kv_handoff_latency_s = handoff_lat
        mean_handoff = handoff_lat / len(reqs) if disagg else 0.0

        # assemble per-request TTFT / completion
        sla_safe = 0
        for i, r in enumerate(reqs):
            ttft = pf_wait + prefill_work[i] + r.transfer_latency_s + mean_handoff
            completion = ttft + dc_wait + decode_work[i]
            res.ttft_s.append(round(ttft, 6))
            res.completion_s.append(round(completion, 6))
            res.prefill_queue_wait_s.append(round(pf_wait, 6))
            res.decode_queue_wait_s.append(round(dc_wait, 6))
            res.prefill_gpu_seconds += prefill_work[i]
            res.decode_gpu_seconds += decode_work[i]
            if completion <= sla_s:
                sla_safe += 1
        res.admitted = len(reqs)
        res.decode_steps = sum_out
        res.sla_slack_s = round(sla_s - res._pct(res.completion_s, 0.95), 5)

        # idle GPU-seconds: provisioned pool capacity minus busy
        prov_pf = n_pf * period_s
        prov_dc = n_dc * period_s
        if disagg:
            res.prefill_idle_gpu_seconds = max(0.0, prov_pf - res.prefill_gpu_seconds)
            res.decode_idle_gpu_seconds = max(0.0, prov_dc - res.decode_gpu_seconds)
            # allocation efficiency: how balanced are the two pools' utilisation
            u_pf = res.prefill_gpu_seconds / max(prov_pf, 1e-9)
            u_dc = res.decode_gpu_seconds / max(prov_dc, 1e-9)
            res.allocation_efficiency = 1.0 - abs(u_pf - u_dc)
        else:
            busy = res.realized_gpu_seconds
            res.prefill_idle_gpu_seconds = max(0.0, n_replicas * period_s - busy)
            res.allocation_efficiency = 1.0
        res.summary_sla_safe = sla_safe
        return res


__all__ = ["PrefillDecodeSchedulerV2", "SchedRequest", "ServingResultV2", "SERVING_MODES", "ALLOCATIONS"]
