"""Phase 1b-A parity tests: ReplayHarness unified entry point.

Verifies that calling ``ReplayHarness.run()`` for each supported benchmark_id
produces the same ``kpi_sla_safe_goodput_per_dollar`` as calling the underlying
backend directly.  0% KPI drift is guaranteed by construction (the harness is a
pure routing facade), and these tests provide the regression guard.

Coverage:
  - Config validation: bad benchmark_id, empty policies, non-positive tick_seconds
  - replica_scaling: KPI identity with run_backtest() on BurstGPT fixture (51 req)
  - genai_serving:   KPI identity with genai run_backtest() on Alibaba fixture
  - serving_queue:   pass-through adapter round-trips a sim_dict correctly
  - Import: ReplayHarness + ReplayHarnessConfig exported from aurelius.optimizer
  - to_dict: all required fields present on returned ReplayEvaluationResult
  - ordering: result list respects config.policies ordering
  - missing policies: policies absent from backend silently skipped
"""
from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.optimizer import (
    ReplayHarness,
    ReplayHarnessConfig,
    ReplayHarnessError,
)

BURSTGPT_FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv")
AZURE_FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv")
GENAI_FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_genai_sample")


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestReplayHarnessConfigValidation:
    def test_valid_config(self):
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=["constraint_aware"],
        )
        assert cfg.benchmark_id == "replica_scaling"
        assert cfg.tick_seconds == 60.0  # default

    def test_invalid_benchmark_id(self):
        with pytest.raises(ReplayHarnessError, match="Unknown benchmark_id"):
            ReplayHarnessConfig(
                benchmark_id="nonexistent",
                trace_id="x",
                policies=["constraint_aware"],
            )

    def test_empty_policies(self):
        with pytest.raises(ReplayHarnessError, match="non-empty"):
            ReplayHarnessConfig(
                benchmark_id="replica_scaling",
                trace_id="x",
                policies=[],
            )

    def test_nonpositive_tick_seconds(self):
        with pytest.raises(ReplayHarnessError, match="positive"):
            ReplayHarnessConfig(
                benchmark_id="replica_scaling",
                trace_id="x",
                policies=["constraint_aware"],
                tick_seconds=0.0,
            )

    def test_all_benchmark_ids_accepted(self):
        from aurelius.optimizer.replay_result import BENCHMARK_IDS
        for bid in BENCHMARK_IDS:
            cfg = ReplayHarnessConfig(
                benchmark_id=bid,
                trace_id="test",
                policies=["constraint_aware"],
            )
            assert cfg.benchmark_id == bid


# ---------------------------------------------------------------------------
# replica_scaling backend — KPI identity
# ---------------------------------------------------------------------------

class TestReplicaScalingParity:
    @pytest.fixture(scope="class")
    def burstgpt_requests(self):
        from aurelius.traces.burstgpt import load_csv
        return load_csv(BURSTGPT_FIXTURE)

    def test_harness_imports_cleanly(self):
        harness = ReplayHarness()
        assert harness is not None

    def test_kpi_identity_constraint_aware(self, burstgpt_requests):
        """ReplayHarness KPI == run_backtest() KPI for constraint_aware."""
        from aurelius.traces.backtest import run_backtest

        policy = "constraint_aware"
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=[policy],
        )
        harness = ReplayHarness()
        results = harness.run(cfg, burstgpt_requests)

        # Direct call for reference
        direct = run_backtest(burstgpt_requests, policies=[policy])
        direct_kpi = direct.policy_results[policy].kpi.sla_safe_goodput_per_infra_dollar

        assert len(results) == 1
        assert results[0].policy == policy
        assert results[0].benchmark_id == "replica_scaling"
        assert results[0].kpi_sla_safe_goodput_per_dollar == pytest.approx(
            direct_kpi, rel=1e-9
        )

    def test_kpi_identity_multiple_policies(self, burstgpt_requests):
        """All requested policies match direct run_backtest() KPIs."""
        from aurelius.traces.backtest import run_backtest

        policies = ["constraint_aware", "safe_high_utilization", "min_cost_safe"]
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=policies,
        )
        harness = ReplayHarness()
        results = harness.run(cfg, burstgpt_requests)

        direct = run_backtest(burstgpt_requests, policies=policies)

        assert len(results) == len(policies)
        for r in results:
            direct_kpi = direct.policy_results[r.policy].kpi.sla_safe_goodput_per_infra_dollar
            assert r.kpi_sla_safe_goodput_per_dollar == pytest.approx(
                direct_kpi, rel=1e-9
            ), f"KPI mismatch for policy {r.policy}"

    def test_result_fields_populated(self, burstgpt_requests):
        """All ReplayEvaluationResult fields are non-trivially populated."""
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=["constraint_aware"],
        )
        result = ReplayHarness().run(cfg, burstgpt_requests)[0]

        assert result.trace_id == "burstgpt_v1"
        assert result.n_requests > 0
        assert result.n_ticks > 0
        assert result.tick_seconds == 60.0
        assert result.kpi_sla_safe_goodput_per_dollar > 0.0

    def test_to_dict_keys(self, burstgpt_requests):
        """to_dict() emits all required schema keys."""
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=["constraint_aware"],
        )
        result = ReplayHarness().run(cfg, burstgpt_requests)[0]
        d = result.to_dict()

        required = {
            "benchmark_id", "trace_id", "policy",
            "kpi_sla_safe_goodput_per_dollar",
            "kpi_sla_compliant_goodput",
            "kpi_gpu_hours", "kpi_total_cost",
            "n_requests", "n_ticks", "tick_seconds",
            "metadata",
        }
        assert required <= set(d.keys())

    def test_policy_ordering_preserved(self, burstgpt_requests):
        """Result list follows config.policies ordering."""
        policies = ["min_cost_safe", "constraint_aware", "safe_high_utilization"]
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=policies,
        )
        results = ReplayHarness().run(cfg, burstgpt_requests)
        assert [r.policy for r in results] == policies

    def test_subset_of_all_policies_returned(self, burstgpt_requests):
        """Requesting a subset of valid policies returns only those policies."""
        from aurelius.traces.backtest import ALL_POLICIES

        subset = list(ALL_POLICIES)[:2]
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=subset,
        )
        results = ReplayHarness().run(cfg, burstgpt_requests)
        returned_policies = [r.policy for r in results]
        assert returned_policies == subset

    def test_trace_id_propagated(self, burstgpt_requests):
        """config.trace_id is reflected in every ReplayEvaluationResult."""
        custom_id = "my_trace_v42"
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id=custom_id,
            policies=["constraint_aware"],
        )
        results = ReplayHarness().run(cfg, burstgpt_requests)
        assert all(r.trace_id == custom_id for r in results)

    def test_azure_fixture_kpi_identity(self):
        """KPI identity on Azure LLM 2024 sample fixture."""
        from aurelius.traces import azure_llm
        from aurelius.traces.backtest import run_backtest

        requests = azure_llm.load_csv(AZURE_FIXTURE)
        policy = "constraint_aware"
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="azure_llm_2024",
            policies=[policy],
        )
        results = ReplayHarness().run(cfg, requests)
        direct = run_backtest(requests, policies=[policy])
        direct_kpi = direct.policy_results[policy].kpi.sla_safe_goodput_per_infra_dollar

        assert len(results) == 1
        assert results[0].kpi_sla_safe_goodput_per_dollar == pytest.approx(
            direct_kpi, rel=1e-9
        )


# ---------------------------------------------------------------------------
# genai_serving backend — KPI identity
# ---------------------------------------------------------------------------

class TestGenAIServingParity:
    @pytest.fixture(scope="class")
    def genai_data(self):
        from aurelius.traces import alibaba_genai as ag
        layers = ag.load_all_layers(GENAI_FIXTURE, request_kwargs=dict(include_failures=False))
        return layers["requests"]

    def test_kpi_identity_constraint_aware(self, genai_data):
        """ReplayHarness KPI == genai run_backtest() KPI for constraint_aware."""
        from aurelius.traces.genai_backtest import run_backtest as run_genai

        policy = "constraint_aware"
        cfg = ReplayHarnessConfig(
            benchmark_id="genai_serving",
            trace_id="alibaba_genai",
            policies=[policy],
            tick_seconds=3600.0,
        )
        results = ReplayHarness().run(cfg, genai_data)
        direct = run_genai(genai_data, tick_seconds=3600.0, policies=[policy])
        direct_kpi = direct.policy_results[policy].kpi.sla_safe_goodput_per_infra_dollar

        assert len(results) == 1
        assert results[0].policy == policy
        assert results[0].benchmark_id == "genai_serving"
        assert results[0].kpi_sla_safe_goodput_per_dollar == pytest.approx(
            direct_kpi, rel=1e-9
        )

    def test_genai_n_requests_populated(self, genai_data):
        cfg = ReplayHarnessConfig(
            benchmark_id="genai_serving",
            trace_id="alibaba_genai",
            policies=["constraint_aware"],
            tick_seconds=3600.0,
        )
        results = ReplayHarness().run(cfg, genai_data)
        assert results[0].n_requests > 0
        assert results[0].n_requests == len(genai_data)


# ---------------------------------------------------------------------------
# serving_queue backend — pass-through adapter
# ---------------------------------------------------------------------------

class TestServingQueuePassThrough:
    def _make_sim_dict(self, goodput: float) -> dict:
        return {
            "sla_safe_goodput_per_dollar": goodput,
            "mean_response_s": 1.5,
            "short_p90_response_s": 2.0,
            "long_p99_response_s": 5.0,
        }

    def test_single_policy_round_trip(self):
        """serving_queue pass-through emits exact goodput from sim_dict."""
        sim_dicts = {"fifo": self._make_sim_dict(12345.0)}
        cfg = ReplayHarnessConfig(
            benchmark_id="serving_queue",
            trace_id="burstgpt_v1",
            policies=["fifo"],
        )
        data = {
            "sim_dicts": sim_dicts,
            "n_requests": 100,
            "n_ticks": 50,
            "servers": 4,
        }
        results = ReplayHarness().run(cfg, data)
        assert len(results) == 1
        assert results[0].kpi_sla_safe_goodput_per_dollar == pytest.approx(12345.0)
        assert results[0].benchmark_id == "serving_queue"
        assert results[0].metadata["servers"] == 4

    def test_multiple_policies_pass_through(self):
        sim_dicts = {
            "fifo": self._make_sim_dict(10000.0),
            "abs_conformal_srpt": self._make_sim_dict(15000.0),
        }
        cfg = ReplayHarnessConfig(
            benchmark_id="serving_queue",
            trace_id="burstgpt_v1",
            policies=["fifo", "abs_conformal_srpt"],
        )
        data = {"sim_dicts": sim_dicts, "n_requests": 200, "n_ticks": 100, "servers": 4}
        results = ReplayHarness().run(cfg, data)

        assert len(results) == 2
        assert results[0].kpi_sla_safe_goodput_per_dollar == pytest.approx(10000.0)
        assert results[1].kpi_sla_safe_goodput_per_dollar == pytest.approx(15000.0)

    def test_missing_policy_skipped(self):
        sim_dicts = {"fifo": self._make_sim_dict(10000.0)}
        cfg = ReplayHarnessConfig(
            benchmark_id="serving_queue",
            trace_id="burstgpt_v1",
            policies=["fifo", "absent_policy"],
        )
        data = {"sim_dicts": sim_dicts, "n_requests": 100, "n_ticks": 50, "servers": 4}
        results = ReplayHarness().run(cfg, data)
        assert len(results) == 1
        assert results[0].policy == "fifo"

    def test_metadata_cost_basis(self):
        sim_dicts = {"fifo": self._make_sim_dict(8000.0)}
        cfg = ReplayHarnessConfig(
            benchmark_id="serving_queue", trace_id="x", policies=["fifo"]
        )
        data = {"sim_dicts": sim_dicts, "n_requests": 50, "n_ticks": 25, "servers": 2}
        results = ReplayHarness().run(cfg, data)
        assert results[0].metadata["cost_basis"] == "busy_gpu_hours"


# ---------------------------------------------------------------------------
# Cross-loop schema consistency
# ---------------------------------------------------------------------------

class TestCrossLoopSchemaConsistency:
    @pytest.fixture(scope="class")
    def burstgpt_requests(self):
        from aurelius.traces.burstgpt import load_csv
        return load_csv(BURSTGPT_FIXTURE)

    def test_to_dict_schema_identical_across_loops(self, burstgpt_requests):
        """to_dict() emits the same keys regardless of benchmark_id."""
        # replica_scaling result
        cfg_rs = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=["constraint_aware"],
        )
        rs_dict = ReplayHarness().run(cfg_rs, burstgpt_requests)[0].to_dict()

        # serving_queue result
        sim_dicts = {"fifo": {"sla_safe_goodput_per_dollar": 5000.0}}
        cfg_sq = ReplayHarnessConfig(
            benchmark_id="serving_queue",
            trace_id="burstgpt_v1",
            policies=["fifo"],
        )
        data = {"sim_dicts": sim_dicts, "n_requests": 51, "n_ticks": 10, "servers": 4}
        sq_dict = ReplayHarness().run(cfg_sq, data)[0].to_dict()

        assert set(rs_dict.keys()) == set(sq_dict.keys()), (
            "to_dict() key mismatch across benchmark_id types: "
            f"replica_scaling has {set(rs_dict.keys())}, "
            f"serving_queue has {set(sq_dict.keys())}"
        )

    def test_benchmark_ids_match_registry(self, burstgpt_requests):
        """benchmark_id in result matches BENCHMARK_IDS registry."""
        from aurelius.optimizer.replay_result import BENCHMARK_IDS

        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="x",
            policies=["constraint_aware"],
        )
        results = ReplayHarness().run(cfg, burstgpt_requests)
        for r in results:
            assert r.benchmark_id in BENCHMARK_IDS
