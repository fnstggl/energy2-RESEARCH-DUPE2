"""Serving-queue decision policy — extracted from the benchmark monolith [Phase 2].

The strongest validated serving-queue discipline (run 2026-06-22-x): **Decoupled
Hybrid SRPT with Absolute-Error Conformal adaptive alpha** — +313.14% (Azure LLM
2024) / +557.12% (BurstGPT HF) SLA-safe goodput/$ vs FIFO (directional simulator
only). It is extracted VERBATIM from
``aurelius/benchmarks/srtf_serving_backtest.py`` so the benchmark no longer owns
the optimizer logic; the benchmark imports these symbols back. This is a
parity-preserving extraction (identical algorithm, constants, and tie-breaks),
NOT a research change: no new priors, no benchmark-assumption changes.

Decision vs. evaluation separation
----------------------------------
This module owns the *decision* (request ordering / preemption / dispatch). It
does NOT own *evaluation* (KPI summarization). The discipline takes a
``summarize`` callback so the benchmark keeps its evaluation function
(``_summarize``) and there is no circular import (dependency is one-way:
benchmark -> policy).

No actual-output-token leakage at decision time
-----------------------------------------------
The preemption key and dispatch key use only ``service_s`` / remaining service
(derived from the *predicted* schedule by the caller) and the request index for
deterministic tie-breaks. ``actual_tokens`` is read solely inside
``calibrator.update(...)`` AFTER a request COMPLETES (legitimate online
calibration of the global aging alpha) — never to order a pending request.
"""

from __future__ import annotations

import heapq

from .base import OptimizationPolicy

# ---------------------------------------------------------------------------
# Conformal alpha constants (values identical to the benchmark originals; this
# module is the canonical owner for the abs-conformal discipline).
#   CONFORMAL_ALPHA_MAX == DECOUPLED_HYBRID_ALPHA_DEFAULT (0.001) in the benchmark.
# ---------------------------------------------------------------------------
CONFORMAL_ALPHA_MAX: float = 0.001
CONFORMAL_WARMUP: int = 100
CONFORMAL_WINDOW: int = 200
CONFORMAL_ABS_TARGET_P90_TOKENS: float = 500.0


class AbsoluteErrorConformalCalibrator:
    """Conformal α calibrator using absolute token-count error instead of relative error.

    **Motivation [run 2026-06-22-x]:**

    The ``ConformalAlphaCalibrator`` (run 2026-06-21-q) uses *relative* prediction
    error: ``rel_err = |predicted − actual| / actual``.  On BurstGPT the relative
    error formula caps the calibrator at 2×alpha_max=0.002 for the running-median
    prior, because short ChatGPT requests (actual=7 tokens, predicted=18) produce
    rel_err=1.57 >> target=0.40 — even though the *absolute* misprediction is only
    11 tokens and has negligible scheduling impact.

    This class replaces relative error with *absolute* error:
        abs_err = |predicted − actual|   (in output tokens)

    The p90 absolute error is driven by long GPT-4 / surprise-long ChatGPT requests
    (abs_err ≈ 300–600 tokens for running-median prior) rather than by small over-
    predictions of short ChatGPT requests (abs_err ≤ 18 tokens).

    Formula:
        α = alpha_max × min(2.0, p90_abs_err / target_p90_abs_tokens)

    With running-median prior on BurstGPT (p90_abs ≈ 300–600 tokens, target=500):
        ratio ≈ 0.6–1.2  →  α ≈ 0.0006–0.0012 (near or below alpha_max=0.001)

    With oracle prior (predicted == actual):
        abs_err → 0  →  α → 0  →  dispatch = pure SRPT (same as rel-error calibrator)

    The key property: small absolute misses (short-request over-predictions) no
    longer cap the calibrator above alpha_max.

    Args:
        alpha_max:              Maximum α (same as fixed best α = 0.001).
        warmup:                 Completions required before α adaptation begins.
        window:                 Sliding-window size for error estimation.
        target_p90_abs_tokens:  Absolute token error at which α = alpha_max.
                                Set to ≈ expected p90 abs error with a "neutral" prior.
    """

    def __init__(
        self,
        alpha_max: float = CONFORMAL_ALPHA_MAX,
        warmup: int = CONFORMAL_WARMUP,
        window: int = CONFORMAL_WINDOW,
        target_p90_abs_tokens: float = CONFORMAL_ABS_TARGET_P90_TOKENS,
    ) -> None:
        self.alpha_max = alpha_max
        self.warmup = warmup
        self.window = window
        self.target_p90_abs_tokens = target_p90_abs_tokens
        self._residuals: list[float] = []
        self._n_completed: int = 0
        self._alpha_sum: float = 0.0
        self._alpha_count: int = 0

    def update(self, predicted_tokens: float, actual_tokens: int) -> None:
        """Record a completed request's absolute prediction error (tokens)."""
        self._n_completed += 1
        abs_err = abs(predicted_tokens - actual_tokens)
        self._residuals.append(abs_err)
        if len(self._residuals) > self.window:
            self._residuals.pop(0)

    def current_alpha(self) -> float:
        """Return calibrated dispatch α from empirical p90 absolute prediction error."""
        if self._n_completed < self.warmup or len(self._residuals) < self.warmup // 2:
            alpha = self.alpha_max
        else:
            sorted_r = sorted(self._residuals)
            p90_idx = min(len(sorted_r) - 1, int(0.90 * len(sorted_r)))
            p90_abs_err = sorted_r[p90_idx]
            ratio = min(2.0, p90_abs_err / max(self.target_p90_abs_tokens, 1e-9))
            alpha = self.alpha_max * ratio
        self._alpha_sum += alpha
        self._alpha_count += 1
        return alpha

    def mean_alpha(self) -> float:
        """Diagnostic: mean α across all dispatch events."""
        return self._alpha_sum / max(1, self._alpha_count)

    def p90_abs_err_tokens(self) -> float:
        """Diagnostic: current p90 absolute error in the sliding window."""
        if len(self._residuals) < self.warmup // 2:
            return float("nan")
        sorted_r = sorted(self._residuals)
        p90_idx = min(len(sorted_r) - 1, int(0.90 * len(sorted_r)))
        return sorted_r[p90_idx]


def simulate_decoupled_hybrid_abs_conformal(
    requests: list,
    servers: int,
    calibrator,
    preemption_overhead_s: float = 0.0,
    *,
    summarize,
) -> tuple[dict, dict, dict]:
    """Decoupled Hybrid SRPT with Absolute-Error Conformal Adaptive α [run 2026-06-22-x].

    Identical to ``_simulate_decoupled_hybrid_conformal`` except that the dispatch
    aging parameter α is recalibrated from the empirical p90 *absolute* prediction
    error (|predicted − actual| in output tokens) rather than the p90 *relative*
    prediction error (|predicted − actual| / actual).

    **Why absolute error [run 2026-06-22-x]:**

    On BurstGPT HF with a running-median prior (~18 tokens), the relative-error
    calibrator caps at α = 2×alpha_max = 0.002 because short ChatGPT requests
    (actual=7, predicted=18) produce rel_err=1.57 — far above target=0.40 — even
    though the absolute misprediction (11 tokens ≈ 1 second of service) is negligible
    for scheduling purposes.

    The absolute-error formula correctly ignores these small absolute misses and
    instead reports uncertainty only where it matters: long requests (GPT-4, surprise-
    long ChatGPT) with abs_err ≈ 200–600 tokens.  With target=500 tokens, the live
    running-median prior yields ratio ≈ 0.6–1.0 → α ≈ 0.0006–0.001 — at or below
    the Pareto-optimal fixed α = 0.001, instead of 2× above it.

    **Preemption key (on new arrival r):**
        remaining_s  [pure SRPT — unchanged]

    **Dispatch key (when server becomes free):**
        key(entry, t) = remaining_s / (1 + α(t) × total_wait_s)
    where α(t) = abs_calibrator.current_alpha() recalibrated from p90 abs error.

    Research basis:
    - GAP_ANALYSIS.md run -w Q2/Q7 (absolute error as calibrator metric)
    - arXiv:2508.14544 (Adaptively Robust LLM Inference)
    - arXiv:1902.00732 (Scheduling with Predictions, Mitzenmacher 2019)
    """
    n = len(requests)
    by_arrival = sorted(requests, key=lambda r: (r.arrival_s, r.idx))
    _npreempt = [0]

    s_req:          list = [None] * servers
    s_start:        list = [0.0] * servers
    s_rem0:         list = [0.0] * servers
    s_ver:          list = [0]   * servers
    s_frozen_wait:  list = [0.0] * servers

    waiting: list = []
    events: list = []
    _eseq = [n + 1]

    def _en() -> int:
        _eseq[0] += 1
        return _eseq[0]

    for i, r in enumerate(by_arrival):
        heapq.heappush(events, (r.arrival_s, 0, i, -1, -1, r))

    def _remaining(sid: int, t: float) -> float:
        return max(0.0, s_rem0[sid] - (t - s_start[sid]))

    def _abs_dispatch_key(entry: tuple, t: float, alpha: float) -> tuple:
        rem_s, frozen_wait_s, wait_entered_s, req = entry
        current_wait = t - wait_entered_s
        total_wait = frozen_wait_s + current_wait
        ek = rem_s / max(1e-9, 1.0 + alpha * total_wait)
        return (ek, req.idx)

    def _start(sid: int, req, rem: float, frozen_wait: float, t: float) -> None:
        s_req[sid]   = req
        s_start[sid] = t
        s_rem0[sid]  = rem
        s_ver[sid]  += 1
        v = s_ver[sid]
        heapq.heappush(events, (t + rem, 1, _en(), sid, v, req))

    response: dict[int, float] = {}

    while events:
        ev  = heapq.heappop(events)
        t   = ev[0]
        ety = ev[1]

        if ety == 0:  # ---- ARRIVAL ----------------------------------------
            req = ev[5]
            free = next((s for s in range(servers) if s_req[s] is None), None)
            if free is not None:
                s_frozen_wait[free] = 0.0
                _start(free, req, req.service_s, 0.0, t)
            else:
                worst_sid, worst_rem = 0, -1.0
                for s in range(servers):
                    r = _remaining(s, t)
                    if r > worst_rem:
                        worst_rem, worst_sid = r, s

                if req.service_s < worst_rem:
                    preempted = s_req[worst_sid]
                    prem = _remaining(worst_sid, t)
                    pfrozen = s_frozen_wait[worst_sid]
                    s_req[worst_sid]  = None
                    s_ver[worst_sid] += 1
                    s_frozen_wait[worst_sid] = 0.0
                    _start(worst_sid, req, req.service_s, 0.0, t)
                    _npreempt[0] += 1
                    waiting.append((prem + preemption_overhead_s, pfrozen, t, preempted))
                else:
                    waiting.append((req.service_s, 0.0, t, req))

        else:  # ---- COMPLETION ---------------------------------------------
            _, _, _, sid, ver, req = ev
            if ver != s_ver[sid]:
                continue
            response[req.idx] = t - req.arrival_s

            calibrator.update(req.predicted_tokens, req.actual_tokens)

            s_req[sid]  = None
            s_ver[sid] += 1

            if waiting:
                alpha = calibrator.current_alpha()
                best_i = min(
                    range(len(waiting)),
                    key=lambda i: _abs_dispatch_key(waiting[i], t, alpha),
                )
                rem_s, frozen_wait_s, wait_entered_s, nxt = waiting.pop(best_i)
                new_frozen = frozen_wait_s + (t - wait_entered_s)
                s_frozen_wait[sid] = new_frozen
                _start(sid, nxt, rem_s, new_frozen, t)

    wait_map = {
        r.idx: max(0.0, response[r.idx] - r.service_s)
        for r in requests if r.idx in response
    }
    resp  = [response[r.idx] for r in requests if r.idx in response]
    waits = [wait_map[r.idx] for r in requests if r.idx in response]
    summary = summarize(requests, response, wait_map, resp, waits, servers)
    summary["preemption_count"] = _npreempt[0]
    return summary, response, wait_map


class ServingQueuePolicy(OptimizationPolicy):
    """Canonical serving-queue policy = Decoupled Hybrid SRPT + abs-conformal alpha.

    Thin wrapper exposing the extracted discipline through the canonical
    :class:`AureliusOptimizer` policy interface. Behavior is identical to calling
    :func:`simulate_decoupled_hybrid_abs_conformal` directly; the benchmark uses
    the function form (injecting its own ``_summarize``), while
    ``AureliusOptimizer(policy="serving_queue")`` reaches the same logic here.
    """

    name = "serving_queue"
    #: ``simulate_queue`` discipline key this policy corresponds to.
    discipline = "decoupled_hybrid_abs_conformal"

    def __init__(
        self,
        *,
        preemption_overhead_s: float = 0.0,
        calibrator_factory=AbsoluteErrorConformalCalibrator,
    ):
        self.preemption_overhead_s = preemption_overhead_s
        self._calibrator_factory = calibrator_factory

    def optimize(
        self,
        requests,
        servers,
        *,
        summarize,
        calibrator=None,
        preemption_overhead_s=None,
    ):
        """Run the serving-queue discipline.

        Args:
            requests: serving requests (objects with ``arrival_s``, ``idx``,
                ``service_s``, ``predicted_tokens``, ``actual_tokens``).
            servers: number of homogeneous replicas behind the queue.
            summarize: KPI summarization callback (evaluation layer), e.g. the
                benchmark's ``_summarize``. Kept external so the decision layer
                does not own evaluation.
            calibrator: optional pre-built calibrator; defaults to a fresh
                :class:`AbsoluteErrorConformalCalibrator`.
            preemption_overhead_s: per-preemption recompute overhead (seconds);
                defaults to the policy's configured value.

        Returns:
            ``(summary, response_map, wait_map)`` — identical to the benchmark.
        """
        cal = calibrator if calibrator is not None else self._calibrator_factory()
        overhead = (
            self.preemption_overhead_s
            if preemption_overhead_s is None
            else preemption_overhead_s
        )
        return simulate_decoupled_hybrid_abs_conformal(
            requests, servers, cal, overhead, summarize=summarize
        )


__all__ = [
    "AbsoluteErrorConformalCalibrator",
    "simulate_decoupled_hybrid_abs_conformal",
    "ServingQueuePolicy",
    "CONFORMAL_ALPHA_MAX",
    "CONFORMAL_WARMUP",
    "CONFORMAL_WINDOW",
    "CONFORMAL_ABS_TARGET_P90_TOKENS",
]
