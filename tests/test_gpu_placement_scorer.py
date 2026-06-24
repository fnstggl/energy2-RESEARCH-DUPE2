"""Tests for aurelius.forecasting.gpu_placement_scorer.

Validates:
 - Shadow-mode default (disabled → neutral scores always)
 - SLA class gating (only latency_critical gets non-zero penalty)
 - Rank ordering: faster GPU type → lower penalty
 - Peer normalization: best = floor, worst = ceil
 - Insufficient sample handling: penalty = 0, status = insufficient_sample
 - Missing prior: penalty = 0, status = no_prior
 - rank_gpu_types ordering and completeness
 - Monotone penalty interpolation
 - Summary report structure
 - No controller / scheduler imports in the module
"""

from __future__ import annotations

import importlib

import pytest

from aurelius.forecasting.gpu_placement_scorer import (
    SHADOW_TAG,
    GpuPlacementConfig,
    GpuPlacementScore,
    GpuPlacementScorer,
    _bin_label,
    _peer_relative_penalty,
)
from aurelius.forecasting.ttft_shadow_prior import TTFTShadowPrior

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_prior(gpu_p50_map: dict) -> TTFTShadowPrior:
    """Build a TTFTShadowPrior from a {gpu_type: p50_s} map.

    Generates synthetic rows so the prior's subgroup counts and table are
    populated correctly via ``fit_from_rows``.
    """
    rows = []
    for gpu_type, ttft_p50 in gpu_p50_map.items():
        # Produce 100 synthetic rows per GPU type so subgroup_n ≥ 50.
        instance_type = f"qwen2-7b_{gpu_type}"
        for i in range(100):
            # Spread around the target p50 so median ≈ ttft_p50.
            jitter = (i % 5) * 0.01 - 0.02
            rows.append({
                "instance_type": instance_type,
                "num_prompt_tokens": 256,
                "actual_ttft_s": ttft_p50 + jitter,
            })
    prior = TTFTShadowPrior()
    prior.fit_from_rows(rows)
    return prior


@pytest.fixture
def three_gpu_prior():
    """Prior with h100 (fastest), a100, t4 (slowest)."""
    return _make_prior({"h100": 0.05, "a100": 0.10, "t4": 0.45})


@pytest.fixture
def single_gpu_prior():
    return _make_prior({"h100": 0.05})


@pytest.fixture
def enabled_config():
    return GpuPlacementConfig(enabled=True)


@pytest.fixture
def disabled_config():
    return GpuPlacementConfig(enabled=False)


# ---------------------------------------------------------------------------
# Basic disabled-mode tests
# ---------------------------------------------------------------------------


class TestDisabledMode:
    def test_score_disabled_returns_neutral(self, three_gpu_prior, disabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=disabled_config)
        s = scorer.score("t4", "7b", 256, "latency_critical")
        assert s.latency_penalty == 0.0
        assert s.status == "disabled"
        assert s.relative_rank is None

    def test_rank_disabled_returns_all_disabled(self, three_gpu_prior, disabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=disabled_config)
        scores = scorer.rank_gpu_types(["h100", "a100", "t4"], "7b", 256, "latency_critical")
        assert all(s.status == "disabled" for s in scores)
        assert all(s.latency_penalty == 0.0 for s in scores)

    def test_default_config_is_disabled(self, three_gpu_prior):
        scorer = GpuPlacementScorer(prior=three_gpu_prior)
        s = scorer.score("h100", "7b", 256, "latency_critical")
        assert s.status == "disabled"
        assert s.latency_penalty == 0.0


# ---------------------------------------------------------------------------
# SLA class gating
# ---------------------------------------------------------------------------


class TestSlaGating:
    def test_best_effort_sla_neutral(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        s = scorer.score("t4", "7b", 256, "best_effort")
        assert s.status == "sla_neutral"
        assert s.latency_penalty == 0.0

    def test_deadline_sla_neutral_by_default(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        s = scorer.score("t4", "7b", 256, "deadline")
        assert s.status == "sla_neutral"
        assert s.latency_penalty == 0.0

    def test_latency_critical_sla_scored(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        s = scorer.score("h100", "7b", 256, "latency_critical",
                         peer_ttft_p50s={"h100": 0.05, "a100": 0.10, "t4": 0.45})
        assert s.status == "scored"

    def test_custom_latency_sla_class(self, three_gpu_prior):
        config = GpuPlacementConfig(
            enabled=True,
            latency_sensitive_sla_classes=frozenset({"latency_critical", "deadline"}),
        )
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=config)
        s = scorer.score("h100", "7b", 256, "deadline",
                         peer_ttft_p50s={"h100": 0.05, "t4": 0.45})
        assert s.status == "scored"


# ---------------------------------------------------------------------------
# Score ordering and peer normalization
# ---------------------------------------------------------------------------


class TestPeerNormalization:
    def test_fastest_gpu_gets_floor_penalty(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        peers = {"h100": 0.05, "a100": 0.10, "t4": 0.45}
        s = scorer.score("h100", "7b", 256, "latency_critical", peer_ttft_p50s=peers)
        assert s.status == "scored"
        assert s.latency_penalty == pytest.approx(enabled_config.penalty_floor, rel=1e-3)
        assert s.relative_rank == pytest.approx(0.0, abs=0.01)

    def test_slowest_gpu_gets_ceil_penalty(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        peers = {"h100": 0.05, "a100": 0.10, "t4": 0.45}
        s = scorer.score("t4", "7b", 256, "latency_critical", peer_ttft_p50s=peers)
        assert s.status == "scored"
        assert s.latency_penalty == pytest.approx(enabled_config.penalty_ceil, rel=1e-3)
        assert s.relative_rank == pytest.approx(1.0, abs=0.01)

    def test_middle_gpu_gets_intermediate_penalty(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        peers = {"h100": 0.05, "a100": 0.10, "t4": 0.45}
        s = scorer.score("a100", "7b", 256, "latency_critical", peer_ttft_p50s=peers)
        assert s.status == "scored"
        assert enabled_config.penalty_floor < s.latency_penalty < enabled_config.penalty_ceil

    def test_penalty_monotone_with_ttft(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        peers = {"h100": 0.05, "a100": 0.10, "t4": 0.45}
        scores = {}
        for gpu in ["h100", "a100", "t4"]:
            s = scorer.score(gpu, "7b", 256, "latency_critical", peer_ttft_p50s=peers)
            scores[gpu] = s.latency_penalty
        assert scores["h100"] < scores["a100"] < scores["t4"]


# ---------------------------------------------------------------------------
# rank_gpu_types ordering
# ---------------------------------------------------------------------------


class TestRankGpuTypes:
    def test_rank_order_fastest_first(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        ranked = scorer.rank_gpu_types(["t4", "a100", "h100"], "7b", 256, "latency_critical")
        gpu_order = [s.gpu_type for s in ranked]
        assert gpu_order[0] == "h100"
        assert gpu_order[-1] == "t4"

    def test_rank_returns_all_candidates(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        ranked = scorer.rank_gpu_types(["t4", "a100", "h100"], "7b", 256, "latency_critical")
        assert len(ranked) == 3
        assert {s.gpu_type for s in ranked} == {"t4", "a100", "h100"}

    def test_rank_sla_neutral_all_zero_penalty(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        ranked = scorer.rank_gpu_types(["t4", "a100", "h100"], "7b", 256, "best_effort")
        assert all(s.latency_penalty == 0.0 for s in ranked)

    def test_rank_unknown_gpu_last(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        ranked = scorer.rank_gpu_types(
            ["h100", "unknown_gpu"], "7b", 256, "latency_critical"
        )
        assert ranked[0].gpu_type == "h100"
        assert ranked[-1].gpu_type == "unknown_gpu"

    def test_rank_single_candidate_no_peer_penalty(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        ranked = scorer.rank_gpu_types(["h100"], "7b", 256, "latency_critical")
        assert len(ranked) == 1
        # Single candidate: no peer context → penalty = 0.0
        assert ranked[0].latency_penalty == 0.0


# ---------------------------------------------------------------------------
# Insufficient sample / no prior
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_insufficient_sample_neutral_penalty(self, enabled_config):
        """Subgroup rows < min_subgroup_rows → penalty = 0, status = insufficient_sample."""
        rows = [
            {"instance_type": "qwen2-7b_h100", "num_prompt_tokens": 256, "actual_ttft_s": 0.05}
            for _ in range(10)  # Only 10 rows, below default min of 50
        ]
        prior = TTFTShadowPrior().fit_from_rows(rows)
        scorer = GpuPlacementScorer(prior=prior, config=enabled_config)
        s = scorer.score("h100", "7b", 256, "latency_critical",
                         peer_ttft_p50s={"h100": 0.05, "a100": 0.10})
        assert s.status == "insufficient_sample"
        assert s.latency_penalty == 0.0
        assert s.subgroup_n < enabled_config.min_subgroup_rows

    def test_no_prior_neutral_penalty(self, enabled_config):
        """Empty prior (no rows) → predict() returns None → status = no_prior."""
        prior = TTFTShadowPrior()
        prior.fit_from_rows([])  # empty → global_p50 = NaN → predict() returns None
        scorer = GpuPlacementScorer(prior=prior, config=enabled_config)
        s = scorer.score("h100", "7b", 256, "latency_critical",
                         peer_ttft_p50s={"h100": 0.05})
        assert s.status == "no_prior"
        assert s.latency_penalty == 0.0

    def test_missing_prompt_tokens_handled(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        s = scorer.score("h100", "7b", None, "latency_critical",
                         peer_ttft_p50s={"h100": 0.05})
        # Should not raise; penalty may be 0 due to missing bin
        assert isinstance(s, GpuPlacementScore)
        assert s.latency_penalty >= 0.0

    def test_none_gpu_type_handled(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        s = scorer.score(None, "7b", 256, "latency_critical")
        assert s.latency_penalty == 0.0


# ---------------------------------------------------------------------------
# Penalty helper unit tests
# ---------------------------------------------------------------------------


class TestPeerRelativePenalty:
    def test_two_peers_best_gets_floor(self):
        peers = {"h100": 0.05, "t4": 0.45}
        rank, penalty = _peer_relative_penalty(0.05, peers, floor=0.05, ceil=0.50)
        assert rank == pytest.approx(0.0, abs=0.01)
        assert penalty == pytest.approx(0.05, abs=0.001)

    def test_two_peers_worst_gets_ceil(self):
        peers = {"h100": 0.05, "t4": 0.45}
        rank, penalty = _peer_relative_penalty(0.45, peers, floor=0.05, ceil=0.50)
        assert rank == pytest.approx(1.0, abs=0.01)
        assert penalty == pytest.approx(0.50, abs=0.001)

    def test_single_peer_no_penalty(self):
        peers = {"h100": 0.05}
        rank, penalty = _peer_relative_penalty(0.05, peers, floor=0.05, ceil=0.50)
        assert rank == 0.0
        assert penalty == 0.0

    def test_penalty_is_monotone_ascending(self):
        peers = {"h100": 0.05, "a100": 0.10, "v100": 0.25, "t4": 0.45}
        penalties = []
        for v in [0.05, 0.10, 0.25, 0.45]:
            _, p = _peer_relative_penalty(v, peers, floor=0.05, ceil=0.50)
            penalties.append(p)
        for i in range(len(penalties) - 1):
            assert penalties[i] <= penalties[i + 1]


# ---------------------------------------------------------------------------
# Bin label helper
# ---------------------------------------------------------------------------


class TestBinLabel:
    def test_small_prompt(self):
        assert _bin_label(10) == "[0,50)"

    def test_medium_prompt(self):
        assert _bin_label(512) == "[200,800)"

    def test_large_prompt(self):
        # 5000 falls within the last bin [3200, 1000000)
        assert _bin_label(5000) == "[3200,1000000)"

    def test_very_large_prompt_beyond_bins(self):
        # Values >= 1_000_000 fall through all bins → ">=" sentinel
        assert _bin_label(1_000_001) == ">=1000000"

    def test_none_returns_missing(self):
        assert _bin_label(None) == "missing"

    def test_zero_boundary(self):
        assert _bin_label(0) == "[0,50)"

    def test_exact_boundary(self):
        assert _bin_label(50) == "[50,200)"


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


class TestSummaryReport:
    def test_report_structure(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        report = scorer.summary_report()
        assert report["scorer"] == "GpuPlacementScorer"
        assert report["enabled"] is True
        assert report["status"] == SHADOW_TAG
        assert isinstance(report["prior_gpu_types_seen"], list)
        assert len(report["prior_gpu_types_seen"]) > 0

    def test_disabled_report_structure(self, three_gpu_prior, disabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=disabled_config)
        report = scorer.summary_report()
        assert report["enabled"] is False


# ---------------------------------------------------------------------------
# Shadow tag propagation
# ---------------------------------------------------------------------------


class TestShadowTagPropagation:
    def test_score_has_shadow_tag(self, three_gpu_prior, enabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=enabled_config)
        s = scorer.score("h100", "7b", 256, "latency_critical")
        assert s.shadow_tag == SHADOW_TAG

    def test_disabled_score_has_shadow_tag(self, three_gpu_prior, disabled_config):
        scorer = GpuPlacementScorer(prior=three_gpu_prior, config=disabled_config)
        s = scorer.score("h100", "7b", 256, "latency_critical")
        assert s.shadow_tag == SHADOW_TAG


# ---------------------------------------------------------------------------
# Module isolation — no controller / scheduler imports
# ---------------------------------------------------------------------------


class TestModuleIsolation:
    FORBIDDEN_MODULES = {
        "aurelius.frontier",
        "aurelius.optimization",
        "aurelius.execution",
        "aurelius.api",
        "aurelius.sla",
    }

    def test_no_forbidden_imports(self):
        """gpu_placement_scorer must not import controller/scheduler modules."""
        import aurelius.forecasting.gpu_placement_scorer as mod_under_test

        source = importlib.util.find_spec(
            "aurelius.forecasting.gpu_placement_scorer"
        )
        assert source is not None, "module not found"

        # Check that none of the forbidden modules are in sys.modules as a
        # side-effect of importing gpu_placement_scorer.
        import sys

        for mod_name in list(sys.modules):
            if any(mod_name.startswith(f) for f in self.FORBIDDEN_MODULES):
                # These may have been imported by other tests; check that
                # gpu_placement_scorer doesn't re-import them internally.
                pass  # We verify via source inspection below.

        # Verify by reading module __file__ content (a static check).
        src_file = mod_under_test.__file__
        assert src_file is not None
        src_text = open(src_file).read()
        for forbidden in self.FORBIDDEN_MODULES:
            forbidden.replace(".", r"\.")
            # Simple substring check (not regex)
            assert f"from {forbidden}" not in src_text, (
                f"gpu_placement_scorer must not import from {forbidden}"
            )
            assert f"import {forbidden}" not in src_text, (
                f"gpu_placement_scorer must not import {forbidden}"
            )


# ---------------------------------------------------------------------------
# Integration: GpuPlacementScorer + real TTFTShadowPrior round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_round_trip_with_prior(self):
        """Synthetic end-to-end: prior fit → score → rank → penalties monotone."""
        prior = _make_prior({
            "h100": 0.03,
            "a100": 0.08,
            "v100": 0.15,
            "t4": 0.50,
        })
        config = GpuPlacementConfig(enabled=True, penalty_floor=0.05, penalty_ceil=0.50)
        scorer = GpuPlacementScorer(prior=prior, config=config)

        ranked = scorer.rank_gpu_types(
            ["t4", "v100", "a100", "h100"],
            model_size="7b",
            prompt_tokens=256,
            sla_class="latency_critical",
        )
        assert len(ranked) == 4
        # Fastest (h100) must be first.
        assert ranked[0].gpu_type == "h100"
        # Slowest (t4) must be last.
        assert ranked[-1].gpu_type == "t4"
        # Penalties must be non-decreasing.
        penalties = [s.latency_penalty for s in ranked]
        for i in range(len(penalties) - 1):
            assert penalties[i] <= penalties[i + 1]
        # Best must have floor penalty.
        assert ranked[0].latency_penalty == pytest.approx(config.penalty_floor, rel=1e-2)
        # Worst must have ceil penalty.
        assert ranked[-1].latency_penalty == pytest.approx(config.penalty_ceil, rel=1e-2)
