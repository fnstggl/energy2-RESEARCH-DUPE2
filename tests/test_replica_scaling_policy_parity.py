"""Parity tests for ReplicaScalingPolicy — Phase 2/3 extraction.

Verifies that every function extracted from the benchmark monolith into
``aurelius.optimizer.policies.replica_scaling`` produces bit-identical results
to the original benchmark implementations it replaced.

Test classes:
  1. TestErlangCParity          — _replica_erlang_c_sla_timeout_pct vs _erlang_c_sla_timeout_pct
  2. TestMCSScheduleParity      — compute_mcs_c_schedule vs _joint_mcs_c_schedule
  3. TestOracleFIFOParity       — _oracle_fifo_response_times vs _simulate_fifo_variable_c
  4. TestSOTSSScheduleParity    — compute_sotss_min_schedule vs _sotss_min_cost_schedule
  5. TestReplicaScalingPolicy   — ReplicaScalingPolicy contract and IMPLEMENTED_POLICIES
  6. TestEdgeCases              — empty input, single tick
"""
import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    _erlang_c_sla_timeout_pct,
    _joint_mcs_c_schedule,
    _Request,
    _service_time_s,
    _simulate_fifo_variable_c,
    _sotss_min_cost_schedule,
    calibrate_time_warp,
    load_serving_requests,
)
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.policies import IMPLEMENTED_POLICIES, ReplicaScalingPolicy
from aurelius.optimizer.policies.replica_scaling import (
    REPLICA_AGGRESSIVE_GATE,
    REPLICA_MAX_ORACLE_ITERS,
    REPLICA_SAFE_GATE,
    REPLICA_TPOT_S,
    REPLICA_TTFT_BASE_S,
    ReplicaScalingConfig,
    ReplicaScalingResult,
    _oracle_fifo_response_times,
    _replica_calibrate_warp,
    _replica_erlang_c_sla_timeout_pct,
    _replica_service_time_s,
    compute_mcs_c_schedule,
    compute_sotss_min_schedule,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_azure_small():
    """200-request slice of Azure LLM 2024 fixture — fast but non-trivial."""
    return load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=200)


@pytest.fixture(scope="module")
def raw_azure_medium():
    """500-request slice — exposes more oracle iterations."""
    return load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=500)


@pytest.fixture(scope="module")
def raw_burstgpt():
    """BurstGPT fixture (54 requests)."""
    from aurelius.benchmarks.srtf_serving_backtest import load_burstgpt_serving_requests
    return load_burstgpt_serving_requests(DEFAULT_BURSTGPT_FIXTURE)


@pytest.fixture(scope="module")
def warp_azure_small(raw_azure_small):
    return calibrate_time_warp(raw_azure_small, servers=5, target_rho=0.85)


@pytest.fixture(scope="module")
def warp_azure_medium(raw_azure_medium):
    return calibrate_time_warp(raw_azure_medium, servers=5, target_rho=0.85)


# ---------------------------------------------------------------------------
# 1. Erlang-C parity
# ---------------------------------------------------------------------------

class TestErlangCParity:
    """_replica_erlang_c_sla_timeout_pct matches _erlang_c_sla_timeout_pct exactly."""

    def _cmp(self, lam, mean_s, c, thresh):
        bench = _erlang_c_sla_timeout_pct(lam, mean_s, c, thresh)
        policy = _replica_erlang_c_sla_timeout_pct(lam, mean_s, c, thresh)
        assert policy == pytest.approx(bench, rel=1e-10), (
            f"Mismatch at lam={lam}, mean_s={mean_s}, c={c}, thresh={thresh}: "
            f"bench={bench}, policy={policy}"
        )

    def test_overloaded_returns_100(self):
        result = _replica_erlang_c_sla_timeout_pct(10.0, 1.0, 1, 5.0)
        assert result == pytest.approx(100.0)
        assert _erlang_c_sla_timeout_pct(10.0, 1.0, 1, 5.0) == pytest.approx(100.0)

    def test_lightly_loaded(self):
        self._cmp(lam=0.1, mean_s=2.0, c=2, thresh=5.0)

    def test_moderately_loaded(self):
        self._cmp(lam=0.8, mean_s=5.0, c=6, thresh=4.5)

    def test_heavily_loaded_subunit(self):
        # rho = (1.0/2.0)/c = 0.5/c; c=2 → rho=0.25
        self._cmp(lam=1.0, mean_s=2.0, c=4, thresh=3.0)

    def test_zero_threshold_gives_erlang_c_prob(self):
        self._cmp(lam=0.5, mean_s=4.0, c=3, thresh=0.0)

    def test_large_c_gives_zero(self):
        result = _replica_erlang_c_sla_timeout_pct(0.5, 2.0, 50, 1.0)
        bench = _erlang_c_sla_timeout_pct(0.5, 2.0, 50, 1.0)
        assert result == pytest.approx(bench, rel=1e-6)
        assert result < 1.0

    def test_constants_match_benchmark_defaults(self):
        assert REPLICA_TTFT_BASE_S == pytest.approx(0.150)
        assert REPLICA_TPOT_S == pytest.approx(0.020)

    def test_service_time_matches(self):
        for tok in [7, 50, 200, 500, 1000]:
            assert _replica_service_time_s(tok) == pytest.approx(_service_time_s(tok))


# ---------------------------------------------------------------------------
# 2. MCS schedule parity
# ---------------------------------------------------------------------------

class TestMCSScheduleParity:
    """compute_mcs_c_schedule produces identical schedule to _joint_mcs_c_schedule."""

    def test_amcsg_gate_azure_small(self, raw_azure_small, warp_azure_small):
        bench = _joint_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=12.5, sla_s=10.0)
        policy = compute_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=12.5, sla_s=10.0)
        assert policy == bench

    def test_safe_gate_default_azure_small(self, raw_azure_small, warp_azure_small):
        bench = _joint_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=9.5, sla_s=10.0)
        policy = compute_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=9.5, sla_s=10.0)
        assert policy == bench

    def test_aggressive_gate_100_azure_small(self, raw_azure_small, warp_azure_small):
        bench = _joint_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=100.0, sla_s=10.0)
        policy = compute_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=100.0, sla_s=10.0)
        assert policy == bench

    def test_burstgpt_30s_sla(self, raw_burstgpt, tmp_path):
        raw = raw_burstgpt
        if not raw:
            pytest.skip("BurstGPT fixture unavailable")
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        bench = _joint_mcs_c_schedule(raw, 60.0, warp, mcs_gate=12.5, sla_s=30.0)
        policy = compute_mcs_c_schedule(raw, 60.0, warp, mcs_gate=12.5, sla_s=30.0)
        assert policy == bench

    def test_schedule_length_matches(self, raw_azure_small, warp_azure_small):
        bench = _joint_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=12.5, sla_s=10.0)
        policy = compute_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=12.5, sla_s=10.0)
        assert len(policy) == len(bench)

    def test_warp_parity(self, raw_azure_small):
        bench_warp = calibrate_time_warp(raw_azure_small, servers=5, target_rho=0.85)
        policy_warp = _replica_calibrate_warp(raw_azure_small, servers=5, target_rho=0.85)
        assert policy_warp == pytest.approx(bench_warp, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. Oracle FIFO parity
# ---------------------------------------------------------------------------

class TestOracleFIFOParity:
    """_oracle_fifo_response_times matches _simulate_fifo_variable_c response map."""

    @pytest.fixture(scope="class")
    def setup_azure(self, raw_azure_small, warp_azure_small):
        raw = raw_azure_small
        warp = warp_azure_small
        c_sched = _joint_mcs_c_schedule(raw, 60.0, warp, mcs_gate=12.5, sla_s=10.0)
        reqs = [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]
        pairs = [(arr / warp, _service_time_s(tok)) for arr, tok in raw]
        _, bench_resp, _ = _simulate_fifo_variable_c(reqs, c_sched, 60.0)
        policy_resp = _oracle_fifo_response_times(pairs, c_sched, 60.0)
        return bench_resp, policy_resp, reqs

    def test_same_keys(self, setup_azure):
        bench_resp, policy_resp, _ = setup_azure
        assert set(policy_resp.keys()) == set(bench_resp.keys())

    def test_same_response_times(self, setup_azure):
        bench_resp, policy_resp, _ = setup_azure
        for idx in bench_resp:
            assert policy_resp[idx] == pytest.approx(bench_resp[idx], rel=1e-9), (
                f"idx={idx}: bench={bench_resp[idx]:.6f}, policy={policy_resp[idx]:.6f}"
            )

    def test_same_sla_safe_count(self, setup_azure):
        bench_resp, policy_resp, reqs = setup_azure
        sla_s = 10.0
        bench_safe = sum(1 for r in reqs if r.idx in bench_resp and bench_resp[r.idx] <= sla_s)
        policy_safe = sum(1 for i in range(len(reqs)) if i in policy_resp and policy_resp[i] <= sla_s)
        assert policy_safe == bench_safe

    def test_variable_c_amcsg_gate_20(self, raw_azure_small, warp_azure_small):
        raw = raw_azure_small
        warp = warp_azure_small
        c_sched = _joint_mcs_c_schedule(raw, 60.0, warp, mcs_gate=20.0, sla_s=10.0)
        reqs = [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]
        pairs = [(arr / warp, _service_time_s(tok)) for arr, tok in raw]
        _, bench_resp, _ = _simulate_fifo_variable_c(reqs, c_sched, 60.0)
        policy_resp = _oracle_fifo_response_times(pairs, c_sched, 60.0)
        assert set(policy_resp.keys()) == set(bench_resp.keys())
        for idx in bench_resp:
            assert policy_resp[idx] == pytest.approx(bench_resp[idx], rel=1e-9)

    def test_constant_c_schedule(self, raw_azure_small, warp_azure_small):
        raw = raw_azure_small
        warp = warp_azure_small
        c_sched = [5] * 20  # fixed c=5 for all ticks
        reqs = [
            _Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=float(tok),
                service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(raw)
        ]
        pairs = [(arr / warp, _service_time_s(tok)) for arr, tok in raw]
        _, bench_resp, _ = _simulate_fifo_variable_c(reqs, c_sched, 60.0)
        policy_resp = _oracle_fifo_response_times(pairs, c_sched, 60.0)
        assert set(policy_resp.keys()) == set(bench_resp.keys())
        for idx in bench_resp:
            assert policy_resp[idx] == pytest.approx(bench_resp[idx], rel=1e-9)


# ---------------------------------------------------------------------------
# 4. SOTSS schedule parity
# ---------------------------------------------------------------------------

class TestSOTSSScheduleParity:
    """compute_sotss_min_schedule matches _sotss_min_cost_schedule output exactly."""

    @pytest.fixture(scope="class")
    def azure_parity(self, raw_azure_small, warp_azure_small):
        raw = raw_azure_small
        warp = warp_azure_small
        bench = _sotss_min_cost_schedule(
            raw, 60.0, warp, 10.0,
            safe_gate=12.5, aggressive_gate=100.0, max_iters=500,
        )
        policy = compute_sotss_min_schedule(
            raw, 60.0, warp, 10.0,
            safe_gate=12.5, aggressive_gate=100.0, max_iters=500,
        )
        return bench, policy

    def test_c_schedule_identical(self, azure_parity):
        bench, policy = azure_parity
        assert policy[0] == bench[0], (
            f"c_schedule mismatch: first diff at {next(i for i,(a,b) in enumerate(zip(policy[0],bench[0])) if a!=b)}"
        )

    def test_n_iters_identical(self, azure_parity):
        bench, policy = azure_parity
        assert policy[1] == bench[1], f"n_iters: bench={bench[1]}, policy={policy[1]}"

    def test_initial_violations_identical(self, azure_parity):
        bench, policy = azure_parity
        assert policy[2] == bench[2]

    def test_n_ticks_cheaper_identical(self, azure_parity):
        bench, policy = azure_parity
        assert policy[3] == bench[3]

    def test_baseline_n_sla_safe_identical(self, azure_parity):
        bench, policy = azure_parity
        assert policy[4] == bench[4]

    def test_explicit_baseline_override(self, raw_azure_small, warp_azure_small):
        raw = raw_azure_small
        warp = warp_azure_small
        baseline = 195  # arbitrary safe floor
        bench = _sotss_min_cost_schedule(
            raw, 60.0, warp, 10.0,
            safe_gate=12.5, aggressive_gate=20.0, max_iters=100,
            baseline_n_sla_safe=baseline,
        )
        policy = compute_sotss_min_schedule(
            raw, 60.0, warp, 10.0,
            safe_gate=12.5, aggressive_gate=20.0, max_iters=100,
            baseline_n_sla_safe=baseline,
        )
        assert policy[0] == bench[0]
        assert policy[4] == bench[4] == baseline

    def test_aggressive_gate_20_parity(self, raw_azure_small, warp_azure_small):
        raw = raw_azure_small
        warp = warp_azure_small
        bench = _sotss_min_cost_schedule(
            raw, 60.0, warp, 10.0,
            safe_gate=12.5, aggressive_gate=20.0, max_iters=200,
        )
        policy = compute_sotss_min_schedule(
            raw, 60.0, warp, 10.0,
            safe_gate=12.5, aggressive_gate=20.0, max_iters=200,
        )
        assert policy[0] == bench[0]
        assert policy[1] == bench[1]


# ---------------------------------------------------------------------------
# 5. ReplicaScalingPolicy contract
# ---------------------------------------------------------------------------

class TestReplicaScalingPolicy:
    """ReplicaScalingPolicy contract and integration with AureliusOptimizer."""

    def test_replica_scaling_in_implemented_policies(self):
        assert "replica_scaling" in IMPLEMENTED_POLICIES

    def test_policy_name(self):
        assert ReplicaScalingPolicy.name == "replica_scaling"

    def test_amcsg_mode_returns_result(self, raw_azure_small, warp_azure_small):
        policy = ReplicaScalingPolicy()
        cfg = ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0)
        result = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg)
        assert isinstance(result, ReplicaScalingResult)
        assert result.mode == "amcsg"
        assert result.oracle_iters == 0
        assert result.n_ticks_cheaper == 0

    def test_sotss_min_mode_returns_result(self, raw_azure_small, warp_azure_small):
        policy = ReplicaScalingPolicy()
        cfg = ReplicaScalingConfig(mode="sotss_min", tick_seconds=60.0, sla_s=10.0)
        result = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg)
        assert isinstance(result, ReplicaScalingResult)
        assert result.mode == "sotss_min"
        assert result.oracle_iters >= 1

    def test_amcsg_c_schedule_matches_benchmark(self, raw_azure_small, warp_azure_small):
        policy = ReplicaScalingPolicy()
        cfg = ReplicaScalingConfig(
            mode="amcsg", tick_seconds=60.0, sla_s=10.0,
            safe_gate_pct=12.5,
        )
        result = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg)
        bench = _joint_mcs_c_schedule(raw_azure_small, 60.0, warp_azure_small, mcs_gate=12.5, sla_s=10.0)
        assert result.c_schedule == bench

    def test_sotss_min_c_mean_le_amcsg(self, raw_azure_small, warp_azure_small):
        policy = ReplicaScalingPolicy()
        cfg_amcsg = ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0)
        cfg_sotss = ReplicaScalingConfig(mode="sotss_min", tick_seconds=60.0, sla_s=10.0)
        amcsg = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg_amcsg)
        sotss = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg_sotss)
        assert sotss.c_mean <= amcsg.c_mean + 1e-9

    def test_aurelius_optimizer_replica_scaling(self, raw_azure_small, warp_azure_small):
        ao = AureliusOptimizer(policy="replica_scaling")
        cfg = ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0)
        result = ao.optimize(raw_azure_small, warp=warp_azure_small, config=cfg)
        assert isinstance(result, ReplicaScalingResult)

    def test_unknown_mode_raises(self, raw_azure_small):
        policy = ReplicaScalingPolicy()
        cfg = ReplicaScalingConfig(mode="invalid_xyz")
        with pytest.raises(ValueError, match="unknown mode"):
            policy.optimize(raw_azure_small, config=cfg)

    def test_result_n_ticks_matches_schedule_length(self, raw_azure_small, warp_azure_small):
        policy = ReplicaScalingPolicy()
        cfg = ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0)
        result = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg)
        assert result.n_ticks == len(result.c_schedule)

    def test_result_c_mean_consistent(self, raw_azure_small, warp_azure_small):
        import statistics
        policy = ReplicaScalingPolicy()
        cfg = ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0)
        result = policy.optimize(raw_azure_small, warp=warp_azure_small, config=cfg)
        assert result.c_mean == pytest.approx(statistics.mean(result.c_schedule), rel=1e-9)

    def test_constants_exported(self):
        assert REPLICA_SAFE_GATE == pytest.approx(12.5)
        assert REPLICA_AGGRESSIVE_GATE == pytest.approx(100.0)
        assert REPLICA_MAX_ORACLE_ITERS == 500


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty input, single-request, single-tick."""

    def test_empty_raw_mcs_returns_empty(self):
        result = compute_mcs_c_schedule([], 60.0, 1.0, mcs_gate=12.5, sla_s=10.0)
        assert result == []

    def test_empty_raw_oracle_returns_empty_dict(self):
        result = _oracle_fifo_response_times([], [1], tick_seconds=60.0)
        assert result == {}

    def test_single_request_oracle(self):
        pairs = [(0.0, 2.0)]  # arrival=0s, service=2s
        result = _oracle_fifo_response_times(pairs, [1], tick_seconds=60.0)
        assert 0 in result
        assert result[0] == pytest.approx(2.0)

    def test_single_request_mcs(self):
        raw = [(0.0, 100)]  # 1 request, 100 output tokens
        c_sched = compute_mcs_c_schedule(raw, 60.0, 1.0, mcs_gate=12.5, sla_s=10.0)
        assert len(c_sched) == 1
        assert c_sched[0] >= 1

    def test_single_request_oracle_matches_benchmark(self):
        warp = 1.0
        c_sched = [1]
        reqs = [
            _Request(
                idx=0,
                arrival_s=0.0,
                actual_tokens=200,
                predicted_tokens=200.0,
                service_s=_service_time_s(200),
            )
        ]
        pairs = [(0.0 / warp, _service_time_s(200))]
        _, bench_resp, _ = _simulate_fifo_variable_c(reqs, c_sched, 60.0)
        policy_resp = _oracle_fifo_response_times(pairs, c_sched, 60.0)
        assert policy_resp.get(0) == pytest.approx(bench_resp.get(0), rel=1e-9)
