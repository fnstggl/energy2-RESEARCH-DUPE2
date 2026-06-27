"""ServingPlane — Azure-native, token-level (built from first principles).

The per-second serving plane. Requests come from the **Azure LLM** spine (real
per-request arrivals + token counts); the serving simulation is the **token-level
discrete-event** loop (`unified_replay`, which passed the quality test: causal,
deployable, no oracle). **Hard rule honored: no hourly M/M/1 queue proxy** — every
serving decision is made on the real per-request queue.

Mooncake calibrates **only** the KV prefix-reuse behaviour: a fraction of requests
are cache hits (calibrated prefix-hit rate), and a hit discounts that request's
service time by the calibrated prefill-savings share. The fleet state (this hour's
capacity envelope + best-effort class mix) parameterizes the run — state variables
flow across planes, never rows.
"""

from __future__ import annotations

import bisect

from ..benchmarks.srtf_serving_backtest import _service_time_s
from ..optimizer.unified_replay import (
    CLASS_BEST_EFFORT,
    CLASS_LATENCY,
    Job,
    run_unified_replay,
)
from .schemas import ServingRequest


def _percentile(sorted_xs: list, q: float) -> float:
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def _causal_predicted_tokens(slice_sorted: list) -> list:
    """Running-median causal token prior (deployable ordering, no oracle)."""
    n = len(slice_sorted)
    if n == 0:
        return []
    gmed = sorted(t for _, t in slice_sorted)[n // 2]
    pred, seen = [0.0] * n, []
    for i, (_, tok) in enumerate(slice_sorted):
        pred[i] = float(seen[(len(seen) - 1) // 2]) if seen else float(gmed)
        bisect.insort(seen, tok)
    return pred


class KVReuseModel:
    """KV prefix-reuse model — calibrated from Mooncake (or a documented default).

    ``hit_rate`` is the fraction of requests that reuse a cached prefix;
    ``prefill_savings_frac`` is the share of service time a hit avoids (prefill).
    A deterministic, index-strided assignment marks hits (no RNG).
    """

    def __init__(self, *, hit_rate: float = 0.0, prefill_savings_frac: float = 0.35) -> None:
        self.hit_rate = max(0.0, min(1.0, hit_rate))
        self.prefill_savings_frac = max(0.0, min(1.0, prefill_savings_frac))

    def is_hit(self, idx: int) -> bool:
        if self.hit_rate <= 0.0:
            return False
        stride = max(1, round(1.0 / self.hit_rate))
        return idx % stride == 0

    def service_s(self, tokens: int, hit: bool) -> float:
        base = _service_time_s(tokens)
        return base * (1.0 - self.prefill_savings_frac) if hit else base


class ServingPlane:
    """Runs one hour's Azure requests through the token-level serving loop."""

    def build_requests(
        self, raw_slice: list, *, warp: float, best_effort_fraction: float,
        kv: KVReuseModel | None = None, kv_model=None, idx_offset: int = 0,
    ) -> list:
        """Map a ``(arrival_s, tokens)`` Azure slice → :class:`ServingRequest`s.

        Classes are assigned deterministically at ``best_effort_fraction`` (from the
        fleet's priority mix). KV hits come from the stateful :class:`KVModel`
        (``kv_model``, Mooncake-fitted) when supplied, else the legacy ``kv``
        stride model. Both are causal (a request's KV outcome depends only on its
        own serving position + the cache state earlier requests produced).
        """
        kv = kv or KVReuseModel()
        slice_sorted = sorted(raw_slice, key=lambda r: r[0])
        pred = _causal_predicted_tokens(slice_sorted)
        be_stride = max(1, round(1.0 / best_effort_fraction)) if best_effort_fraction > 0 else 0
        out = []
        for i, (arr, tok) in enumerate(slice_sorted):
            cls = CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY
            req = ServingRequest(
                idx=idx_offset + i, arrival_s=arr / warp, tokens=int(tok),
                predicted_tokens=float(pred[i]), cls=cls)
            if kv_model is not None and getattr(kv_model, "enabled", False):
                o = kv_model.outcome(idx_offset + i, int(tok))
                req.kv_prefix_id = "hit" if o.hit else ""
                req.kv_reuse_prob = 1.0 if o.hit else 0.0
                req.kv_service_factor = o.service_factor
                req.kv_tokens_saved = o.prefill_tokens_saved
            else:
                hit = kv.is_hit(i)
                req.kv_prefix_id = "hit" if hit else ""
                req.kv_reuse_prob = 1.0 if hit else 0.0
            out.append(req)
        return out

    def run_hour(
        self, requests: list, fleet, *, tick_seconds: float, sla_s: float,
        kv: KVReuseModel | None = None, kv_model=None,
        capacity: str = "backlog_aware", ordering: str = "abs_conformal",
        admission: str = "class_aware",
    ):
        """Run :class:`ServingRequest`s through the discrete-event loop under ``fleet``.

        Returns ``(kpi, action)``. The fleet's ``capacity_envelope`` caps cold-start
        capacity; a KV hit discounts that request's service time (the stateful
        ``kv_model`` per-request factor when supplied, else the legacy prefill
        discount). No M/M/1.
        """
        kv = kv or KVReuseModel()
        use_model = kv_model is not None and getattr(kv_model, "enabled", False)

        def _svc(r):
            if use_model:
                return _service_time_s(r.tokens) * r.kv_service_factor
            return kv.service_s(r.tokens, bool(r.kv_prefix_id))

        jobs = [
            Job(idx=r.idx, arrival_s=r.arrival_s, actual_tokens=r.tokens,
                predicted_tokens=r.predicted_tokens, service_s=_svc(r), cls=r.cls)
            for r in requests
        ]
        warmup_c = max(1, min(fleet.capacity_envelope, 4))
        kpi = run_unified_replay(
            jobs, tick_seconds=tick_seconds, sla_s=sla_s, capacity=capacity,
            ordering=ordering, admission=admission, warmup_c=warmup_c)
        # run_unified_replay mutates each Job's start_s → per-request queue wait.
        waits = sorted(max(0.0, j.start_s - j.arrival_s) for j in jobs if j.start_s >= 0)
        n_hits = sum(1 for r in requests if r.kv_prefix_id)
        action = {
            "capacity": capacity, "ordering": ordering, "admission": admission,
            "warmup_c": warmup_c,
            "kv_enabled": use_model,
            "kv_hit_rate": round(n_hits / len(requests), 4) if requests else 0.0,
            "n_kv_hits": n_hits,
            "kv_tokens_saved": sum(r.kv_tokens_saved for r in requests),
            "capacity_envelope": fleet.capacity_envelope,
            "queue_delay_p50": round(_percentile(waits, 0.50), 4),
            "queue_delay_p95": round(_percentile(waits, 0.95), 4),
            "queue_delay_p99": round(_percentile(waits, 0.99), 4),
        }
        return kpi, action


__all__ = ["ServingPlane", "KVReuseModel"]
