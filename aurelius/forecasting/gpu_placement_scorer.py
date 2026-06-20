"""Heterogeneous GPU Placement Scorer — TTFT-based routing penalty for LLM inference.

Converts per-(model_size, gpu_type, prompt_token_bin) TTFT p50 priors from
``TTFTShadowPrior`` into a normalized latency-penalty score that can be folded
into the Aurelius scheduler's candidate ranking for ``latency_critical`` and
``deadline`` SLA workloads.

Research basis
--------------
- CARA dataset (asdwb/cara_latency_prediction): 76,825 rows covering 5 GPU types
  (H100, A100, A10G, T4, etc.); 9× TTFT p99 spread across instance types.
- "Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for
  SLO-Constrained Inference" (arXiv:2604.07472): near-optimal SLO-compliant
  allocation across mixed GPU generations in < 1 s.
- "KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving"
  (arXiv:2604.16682, April 2026): hardware-aware placement + heterogeneous
  scheduling with TTFT-aware SLO routing.
- "Efficient LLM Scheduling by Learning to Rank" (arXiv:2408.15792): SRTF-like
  ranking of requests by predicted service time exploits short-request priority.

Design rules (binding)
-----------------------
- Shadow-mode only: ``GpuPlacementConfig.enabled`` defaults to False. The scorer
  produces penalty scores for logging / shadow evaluation; it NEVER overrides a
  production placement decision unless the caller explicitly opts in.
- No leakage: ``actual_ttft_s`` is NEVER a feature; only the fitted prior table
  (derived from training data) is queried at score time.
- No controller imports: this module must not import from frontier, scheduler,
  optimization, or any executor module.
- Fail-open: if the prior is missing or the subgroup is too small, penalty = 0
  (neutral; the placement proceeds unpenalized).
- Not production-ready: simulator / directional use only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Shadow-mode sentinel tag.
SHADOW_TAG = "shadow_only_not_production_ready"

# SLA classes for which TTFT penalties are non-zero.
_LATENCY_SENSITIVE_SLA_CLASSES = frozenset({"latency_critical"})

# Minimum subgroup rows in the prior before we trust the estimate.
_DEFAULT_MIN_SUBGROUP_ROWS = 50


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GpuPlacementConfig:
    """Configuration for the GPU placement scorer.

    Attributes:
        enabled: When False (default), all ``score()`` calls return a neutral
            score (penalty = 0.0). Set True to enable shadow scoring.
        latency_sensitive_sla_classes: SLA classes for which TTFT penalties
            are applied. Other SLA classes receive penalty = 0.0.
        min_subgroup_rows: Prior subgroup must have at least this many rows
            before the score is trusted; below the threshold ``status`` is
            ``insufficient_sample`` and penalty = 0.0.
        penalty_floor: Minimum possible non-zero penalty (prevents very small
            advantages from dominating the ranking).
        penalty_ceil: Maximum penalty assigned to the worst GPU type in the
            candidate set (caps the influence on the scheduler objective).
    """

    enabled: bool = False
    latency_sensitive_sla_classes: frozenset = field(
        default_factory=lambda: frozenset({"latency_critical"})
    )
    min_subgroup_rows: int = _DEFAULT_MIN_SUBGROUP_ROWS
    penalty_floor: float = 0.05
    penalty_ceil: float = 0.50


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------


@dataclass
class GpuPlacementScore:
    """TTFT-based placement score for a single (gpu_type, model_size) candidate.

    Fields:
        gpu_type: GPU type evaluated (e.g. "h100", "a100").
        model_size: Model size token from instance_type (e.g. "70b", "7b").
        prompt_token_bin: Prompt-token bucket used for the prior lookup.
        ttft_p50_s: Predicted TTFT p50 in seconds for this candidate. None if
            no prior is available.
        relative_rank: Position in the sorted candidate list, normalised to
            [0, 1]. 0.0 = fastest (best), 1.0 = slowest (worst). None when
            fewer than 2 scored candidates exist.
        latency_penalty: Additive penalty in [0, 1] to fold into the
            scheduler's objective. 0.0 means no penalty (best / neutral).
        status: One of:
            - ``scored``: full prior available, penalty computed.
            - ``insufficient_sample``: prior rows < min_subgroup_rows; neutral.
            - ``no_prior``: prior lookup returned None; neutral.
            - ``sla_neutral``: SLA class is not latency-sensitive; neutral.
            - ``disabled``: scorer is disabled; neutral.
        sla_class: SLA class of the workload being scored.
    """

    gpu_type: Optional[str]
    model_size: Optional[str]
    prompt_token_bin: str
    ttft_p50_s: Optional[float]
    relative_rank: Optional[float]
    latency_penalty: float
    status: str
    sla_class: str
    subgroup_n: int = 0
    shadow_tag: str = SHADOW_TAG


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class GpuPlacementScorer:
    """Converts TTFT priors into placement-ranking penalties for the scheduler.

    Usage (shadow mode)::

        from aurelius.forecasting.ttft_shadow_prior import TTFTShadowPrior
        from aurelius.forecasting.gpu_placement_scorer import (
            GpuPlacementScorer, GpuPlacementConfig,
        )

        prior = TTFTShadowPrior().fit_from_rows(cara_rows)
        scorer = GpuPlacementScorer(prior=prior, config=GpuPlacementConfig(enabled=True))

        scores = scorer.rank_gpu_types(
            candidates=["h100", "a100", "t4"],
            model_size="70b",
            prompt_tokens=512,
            sla_class="latency_critical",
        )
        # scores is sorted fastest → slowest; each has .latency_penalty for folding.

    The scorer is thin: it delegates all TTFT lookups to ``TTFTShadowPrior``
    and only adds normalisation + penalty mapping on top. It contains NO
    training logic and NO scheduler / controller imports.
    """

    def __init__(
        self,
        prior,
        config: Optional[GpuPlacementConfig] = None,
    ):
        """
        Args:
            prior: A fitted ``TTFTShadowPrior`` instance.
            config: Scorer configuration. Defaults to disabled.
        """
        self._prior = prior
        self._config = config or GpuPlacementConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        gpu_type: Optional[str],
        model_size: Optional[str],
        prompt_tokens: Optional[float],
        sla_class: str,
        *,
        peer_ttft_p50s: Optional[dict] = None,
    ) -> GpuPlacementScore:
        """Score a single candidate placement.

        Args:
            gpu_type: GPU type string (e.g. "h100").
            model_size: Model size string (e.g. "70b").
            prompt_tokens: Prompt token count for bin lookup.
            sla_class: SLA class of the request.
            peer_ttft_p50s: Optional mapping of ``{gpu_type: ttft_p50_s}`` for
                all candidate GPU types in this scheduling decision. When
                provided, ``relative_rank`` and ``latency_penalty`` are computed
                relative to this peer set. When None, ``relative_rank`` is None
                and penalty is computed from the absolute prior value vs global
                median.

        Returns:
            GpuPlacementScore
        """
        if not self._config.enabled:
            return GpuPlacementScore(
                gpu_type=gpu_type,
                model_size=model_size,
                prompt_token_bin=_bin_label(prompt_tokens),
                ttft_p50_s=None,
                relative_rank=None,
                latency_penalty=0.0,
                status="disabled",
                sla_class=sla_class,
            )

        if sla_class not in self._config.latency_sensitive_sla_classes:
            return GpuPlacementScore(
                gpu_type=gpu_type,
                model_size=model_size,
                prompt_token_bin=_bin_label(prompt_tokens),
                ttft_p50_s=None,
                relative_rank=None,
                latency_penalty=0.0,
                status="sla_neutral",
                sla_class=sla_class,
            )

        ttft_p50 = self._prior.predict(
            model_size=model_size,
            gpu_type=gpu_type,
            prompt_tokens=prompt_tokens,
        )
        subgroup_n = self._prior.subgroup_n(
            model_size=model_size,
            gpu_type=gpu_type,
            prompt_tokens=prompt_tokens,
        )
        prompt_bin = _bin_label(prompt_tokens)

        if ttft_p50 is None or math.isnan(ttft_p50):
            return GpuPlacementScore(
                gpu_type=gpu_type,
                model_size=model_size,
                prompt_token_bin=prompt_bin,
                ttft_p50_s=None,
                relative_rank=None,
                latency_penalty=0.0,
                status="no_prior",
                sla_class=sla_class,
                subgroup_n=subgroup_n,
            )

        if subgroup_n < self._config.min_subgroup_rows:
            return GpuPlacementScore(
                gpu_type=gpu_type,
                model_size=model_size,
                prompt_token_bin=prompt_bin,
                ttft_p50_s=ttft_p50,
                relative_rank=None,
                latency_penalty=0.0,
                status="insufficient_sample",
                sla_class=sla_class,
                subgroup_n=subgroup_n,
            )

        # Compute penalty relative to peer set when available.
        if peer_ttft_p50s and len(peer_ttft_p50s) >= 2:
            relative_rank, penalty = _peer_relative_penalty(
                ttft_p50,
                peer_ttft_p50s,
                floor=self._config.penalty_floor,
                ceil=self._config.penalty_ceil,
            )
        else:
            # No peer context: neutral penalty, rank unknown.
            relative_rank = None
            penalty = 0.0

        return GpuPlacementScore(
            gpu_type=gpu_type,
            model_size=model_size,
            prompt_token_bin=prompt_bin,
            ttft_p50_s=ttft_p50,
            relative_rank=relative_rank,
            latency_penalty=penalty,
            status="scored",
            sla_class=sla_class,
            subgroup_n=subgroup_n,
        )

    def rank_gpu_types(
        self,
        candidates: list[str],
        model_size: Optional[str],
        prompt_tokens: Optional[float],
        sla_class: str,
    ) -> list[GpuPlacementScore]:
        """Rank a list of GPU type candidates from fastest to slowest.

        Returns a list of GpuPlacementScore sorted ascending by TTFT p50
        (best first). Candidates without a valid prior appear last in the
        ranking with neutral (0.0) penalty.

        Args:
            candidates: List of GPU type strings (e.g. ["h100", "a100", "t4"]).
            model_size: Model size for prior lookup.
            prompt_tokens: Prompt token count for bin selection.
            sla_class: SLA class of the workload.

        Returns:
            Sorted list of GpuPlacementScore (length == len(candidates)).
        """
        if not self._config.enabled:
            return [
                self.score(g, model_size, prompt_tokens, sla_class)
                for g in candidates
            ]

        # First pass: collect raw p50 values for all candidates.
        peer_p50s: dict = {}
        for g in candidates:
            ttft = self._prior.predict(
                model_size=model_size,
                gpu_type=g,
                prompt_tokens=prompt_tokens,
            )
            n = self._prior.subgroup_n(
                model_size=model_size,
                gpu_type=g,
                prompt_tokens=prompt_tokens,
            )
            if (
                ttft is not None
                and not math.isnan(ttft)
                and n >= self._config.min_subgroup_rows
            ):
                peer_p50s[g] = ttft

        # Second pass: score with peer context.
        scores = [
            self.score(
                g, model_size, prompt_tokens, sla_class,
                peer_ttft_p50s=peer_p50s,
            )
            for g in candidates
        ]

        # Sort: scored candidates first (ascending TTFT), then unscored.
        def _sort_key(s: GpuPlacementScore):
            if s.ttft_p50_s is None:
                return (1, float("inf"))
            return (0, s.ttft_p50_s)

        return sorted(scores, key=_sort_key)

    def summary_report(self) -> dict:
        """Return a serialisable audit summary."""
        return {
            "scorer": "GpuPlacementScorer",
            "enabled": self._config.enabled,
            "min_subgroup_rows": self._config.min_subgroup_rows,
            "latency_sensitive_sla_classes": list(
                self._config.latency_sensitive_sla_classes
            ),
            "prior_fit_row_count": getattr(self._prior, "fit_row_count", None),
            "prior_global_p50_s": getattr(self._prior, "global_p50", None),
            "prior_gpu_types_seen": list(
                getattr(self._prior, "by_gpu", {}).keys()
            ),
            "status": SHADOW_TAG,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Prompt-token bin boundaries (shared convention with cara_latency_features).
_PROMPT_BINS = [(0, 50), (50, 200), (200, 800), (800, 3200), (3200, 1_000_000)]


def _bin_label(v, bins=_PROMPT_BINS) -> str:
    if v is None:
        return "missing"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi in bins:
        if lo <= v < hi:
            return f"[{lo},{hi})"
    return f">={bins[-1][1]}"


def _peer_relative_penalty(
    ttft_p50: float,
    peer_p50s: dict,
    floor: float,
    ceil: float,
) -> tuple[float, float]:
    """Compute relative rank and penalty for a candidate vs its peers.

    The penalty is linearly interpolated between ``floor`` and ``ceil`` based
    on relative rank (position / (n-1) for n candidates).  The best GPU type
    always receives ``floor``; the worst receives ``ceil``.  GPU types in the
    middle receive an interpolated value.

    Returns:
        (relative_rank, penalty) where relative_rank ∈ [0, 1].
    """
    sorted_vals = sorted(peer_p50s.values())
    n = len(sorted_vals)
    if n < 2:
        return (0.0, 0.0)

    # Find this candidate's position in the sorted list.
    # Use bisect to handle ties gracefully.
    pos = 0
    for i, v in enumerate(sorted_vals):
        if ttft_p50 <= v:
            pos = i
            break
    else:
        pos = n - 1

    rank = pos / (n - 1)  # 0.0 = best, 1.0 = worst
    penalty = floor + rank * (ceil - floor)
    return (rank, round(penalty, 4))
