"""Prefill/decode serving model + service-time-sensitive economics (PR #107).

PR #106 made a KV prefix hit lower a per-request service factor, but (a) the live service time had only
a CONSTANT prefill term (`TTFT_BASE_S`), so a hit had no prompt-prefill to cut and instead scaled the
whole service — including decode, which is physically wrong; and (b) cost was a capacity integral pinned
to the floor, so faster service never cut cost (see PREFILL_DECODE_ECONOMICS_GAP_AUDIT.md). This module
fixes both, honestly:

- **Prefill is prompt-driven and KV-reducible; decode is output-driven and KV-insensitive.**
  `prefill_work_s = TTFT_BASE_S + (prompt − prefix_hit)·PREFILL_S_PER_TOKEN + model_cold_s`;
  `decode_work_s = out·TPOT_S·batch_factor`. A KV hit cuts **prefill only** → lower **TTFT**; completion
  stays decode-bound. Long outputs remain decode-bound however much prefill is saved.
- **Realized GPU-seconds + cost modes.** `realized_gpu_seconds = Σ(prefill_work + decode_work)`. Cost can
  follow provisioned capacity (reproduces #106), realized work (upper-bound counterfactual), or a hybrid
  (provisioned floor + service reduces active-replica-seconds, bounded). **No mode is free**; warm/idle
  capacity always costs.

Every effect flows through service time / GPU-seconds; nothing here touches reward directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# public bands (see PREFILL_DECODE_SERVING_PHYSICS_AUDIT.md). Existing constants kept identical.
TTFT_BASE_S = 0.150                 # min prefill / first-token overhead (BENCHMARK_DERIVED, existing)
TPOT_S = 0.020                      # per-output-token decode, ~50 tok/s/seq (BENCHMARK_DERIVED, existing)
PREFILL_S_PER_TOKEN = 0.00015       # prompt-token prefill, ~6.7k tok/s (BENCHMARK_DERIVED; band .00007–.0004)
GPU_HOUR_USD = 2.0                  # matches unified_replay
COST_MODES = ("provisioned_capacity", "realized_serving_work", "hybrid_capacity_work")
# hybrid: provisioned baseline, but realized work below the baseline earns a bounded discount; the idle
# floor is never free (you cannot scale a warm pool to zero within a control step).
HYBRID_IDLE_FLOOR_FRAC = 0.5        # ≥ half of provisioned GPU-seconds is billed regardless (warm floor)
# decode batching factor by policy (≤1 helps per-token throughput; >1 tail penalty past saturation).
BATCH_DECODE_FACTOR = {"conservative": 1.0, "balanced": 0.92, "aggressive": 0.82}
BATCH_SATURATION_SEQS = {"conservative": 64, "balanced": 48, "aggressive": 32}


@dataclass
class PhaseResult:
    """Per-request phase service times + period aggregates from the prefill/decode model."""
    prefill_work_s: list = field(default_factory=list)
    decode_work_s: list = field(default_factory=list)
    service_s: list = field(default_factory=list)       # prefill+decode (feeds the cluster replay)
    ttft_s: list = field(default_factory=list)
    completion_s: list = field(default_factory=list)
    prefill_tokens_total: int = 0
    prefill_tokens_saved: int = 0
    prefill_tokens_remaining: int = 0
    decode_tokens_total: int = 0
    realized_gpu_seconds: float = 0.0
    prefill_gpu_seconds: float = 0.0
    decode_gpu_seconds: float = 0.0
    active_decode_sequences_mean: float = 0.0

    def _pct(self, xs, q):
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[min(len(s) - 1, int(len(s) * q))]

    def summary(self) -> dict:
        n = len(self.service_s)
        pf, dc = self.prefill_gpu_seconds, self.decode_gpu_seconds
        # PHASE-time bottleneck (which phase dominates GPU-seconds). NOT the roofline regime
        # (compute- vs memory-bandwidth-bound) — see roofline.py. PR #108 renamed these to avoid the
        # decode-phase-bound / memory-bandwidth-bound conflation.
        bottleneck = ("decode_phase_bound" if dc > 2 * pf
                      else ("prefill_phase_bound" if pf > 2 * dc else "mixed_phase_bound"))
        return {"n": n, "prefill_tokens_total": self.prefill_tokens_total,
                "prefill_tokens_saved": self.prefill_tokens_saved,
                "prefill_tokens_remaining": self.prefill_tokens_remaining,
                "decode_tokens_total": self.decode_tokens_total,
                "realized_gpu_seconds": round(self.realized_gpu_seconds, 3),
                "prefill_gpu_seconds": round(pf, 3), "decode_gpu_seconds": round(dc, 3),
                "ttft_p50": round(self._pct(self.ttft_s, 0.5), 4),
                "ttft_p95": round(self._pct(self.ttft_s, 0.95), 4),
                "ttft_p99": round(self._pct(self.ttft_s, 0.99), 4),
                "completion_p50": round(self._pct(self.completion_s, 0.5), 4),
                "completion_p95": round(self._pct(self.completion_s, 0.95), 4),
                "completion_p99": round(self._pct(self.completion_s, 0.99), 4),
                "active_decode_sequences_mean": round(self.active_decode_sequences_mean, 3),
                "phase_bottleneck": bottleneck}


def compute_phase_serving(reqs, saved_tokens, *, model_cold_s=None, batching="balanced",
                          prefill_s_per_token=PREFILL_S_PER_TOKEN, period_seconds=60.0) -> PhaseResult:
    """Per-request prefill/decode service times from ``reqs`` (arrival, out_tok, in_tok) and the PR #106
    per-request ``saved_tokens`` (matched prefix tokens). KV reuse cuts PREFILL only; decode is the
    output-token term, untouched. Returns the combined ``service_s`` (for the cluster replay) plus the
    realized GPU-seconds and TTFT/decode diagnostics. Causal: ``saved_tokens[i]`` came from residency
    state admitted by requests < i (PR #106)."""
    res = PhaseResult()
    bf = BATCH_DECODE_FACTOR.get(batching, 1.0)
    sat = BATCH_SATURATION_SEQS.get(batching, 48)
    model_cold_s = model_cold_s or [0.0] * len(reqs)
    decode_work_sum = 0.0
    for i, r in enumerate(reqs):
        out = int(r[1])
        prompt = int(r[2]) if len(r) > 2 else out
        saved = min(int(saved_tokens[i]) if i < len(saved_tokens) else 0, prompt)
        remaining = max(prompt - saved, 0)
        prefill = TTFT_BASE_S + remaining * prefill_s_per_token + (model_cold_s[i] if i < len(model_cold_s) else 0.0)
        decode = out * TPOT_S * bf
        res.prefill_work_s.append(round(prefill, 6))
        res.decode_work_s.append(round(decode, 6))
        res.service_s.append(round(prefill + decode, 6))
        res.ttft_s.append(round(prefill, 6))                 # service-only TTFT (cluster queue added by replay)
        res.completion_s.append(round(prefill + decode, 6))
        res.prefill_tokens_total += prompt
        res.prefill_tokens_saved += saved
        res.prefill_tokens_remaining += remaining
        res.decode_tokens_total += out
        res.prefill_gpu_seconds += prefill
        res.decode_gpu_seconds += decode
        decode_work_sum += decode
    res.realized_gpu_seconds = res.prefill_gpu_seconds + res.decode_gpu_seconds
    # Little's law occupancy: mean concurrent decode sequences = decode-work / period (capped → tail).
    res.active_decode_sequences_mean = decode_work_sum / max(period_seconds, 1e-9)
    # aggressive batching past saturation pays a tail penalty on decode (memory/KV pressure).
    if res.active_decode_sequences_mean > sat:
        over = res.active_decode_sequences_mean / sat
        res.realized_gpu_seconds += res.decode_gpu_seconds * 0.1 * (over - 1.0)
    return res


def effective_gpu_hours(cost_mode, *, provisioned_gpu_seconds, realized_gpu_seconds,
                        idle_floor_frac=HYBRID_IDLE_FLOOR_FRAC) -> float:
    """Map (provisioned, realized) GPU-seconds → billable GPU-hours under ``cost_mode``.

    - provisioned_capacity: bill the provisioned capacity integral (the existing behaviour — faster
      service does NOT reduce cost; reproduces PR #106).
    - realized_serving_work: bill realized serving GPU-seconds (the upper-bound counterfactual — faster
      service directly reduces cost; label any win from this mode as a counterfactual, not production).
    - hybrid_capacity_work: bill `max(realized, idle_floor·provisioned)` — a warm/idle floor is never
      free, but realized work above the floor reduces cost (the defensible default). Bounded:
      realized ≤ provisioned by construction, and the floor caps the discount."""
    prov = max(provisioned_gpu_seconds, 0.0)
    real = max(min(realized_gpu_seconds, prov) if prov > 0 else realized_gpu_seconds, 0.0)
    if cost_mode == "provisioned_capacity":
        gpu_s = prov
    elif cost_mode == "realized_serving_work":
        gpu_s = max(real, 0.05 * prov)                       # tiny floor: never literally zero cost
    elif cost_mode == "hybrid_capacity_work":
        gpu_s = max(real, idle_floor_frac * prov)
    else:
        raise ValueError(f"unknown cost_mode {cost_mode}")
    return gpu_s / 3600.0


__all__ = ["PhaseResult", "compute_phase_serving", "effective_gpu_hours", "COST_MODES",
           "TTFT_BASE_S", "TPOT_S", "PREFILL_S_PER_TOKEN", "GPU_HOUR_USD", "HYBRID_IDLE_FLOOR_FRAC"]
