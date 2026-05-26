"""Validation tests for the KV-cache / prefix-affinity / memory-pressure layer.

Covers the audited realism gaps the KV upgrade targets:
- KV memory scaling law (incl. GQA/MQA via kv_heads, not hidden_size);
- KV pressure regions (LOW → ELEVATED → THROTTLING → PREEMPTION);
- PagedAttention internal block slack (NOT heap fragmentation);
- prefix-cache hit-rate sigmoid gated by routing locality;
- cold-reroute penalties + reuse-driven locality warmup/decay;
- preemption / recompute under KV exhaustion;
- cache-aware batching, telemetry-confidence tiers;
- the headline product properties: cold reroutes destroy TTFT, KV exhaustion
  drives preemption storms, and cache-aware affinity beats naive arbitrage;
- calibration metadata has no hidden constants.

All pure functions are deterministic; integration scenarios run under a fixed
seed so results are reproducible.
"""

from __future__ import annotations

from aurelius.benchmarks.constraint_runner import ConstraintBenchmarkRunner
from aurelius.simulation.cluster import kv_cache as kvc
from aurelius.simulation.cluster.calibration import (
    KV_CACHE_PARAMS,
    MODEL_KV_PROFILES,
    calibration_table,
    kv_value,
    model_profile_table,
    resolve_kv_profile,
)
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.scenarios import load_scenario

# ---------------------------------------------------------------------------
# KV memory scaling law
# ---------------------------------------------------------------------------

class TestKVScalingLaw:
    def test_kv_bytes_per_token_matches_formula(self):
        p = MODEL_KV_PROFILES["llama2-7b"]  # 32 layers, 32 kv heads, 128 head_dim, fp16
        expected = 32 * 32 * 128 * 2 * 2.0
        assert kvc.kv_bytes_per_token(p) == expected

    def test_gqa_uses_fewer_kv_heads_than_mha(self):
        gqa = kvc.kv_bytes_per_token(MODEL_KV_PROFILES["llama3-8b"])   # 8 kv heads
        mha = kvc.kv_bytes_per_token(MODEL_KV_PROFILES["llama2-7b"])   # 32 kv heads
        assert gqa < mha
        # GQA with 8 vs 32 KV heads → 4x smaller KV at equal layers/head_dim.
        assert abs(mha / gqa - 4.0) < 1e-6

    def test_mqa_smallest(self):
        mqa = kvc.kv_bytes_per_token(MODEL_KV_PROFILES["mqa-7b"])      # 1 kv head
        gqa = kvc.kv_bytes_per_token(MODEL_KV_PROFILES["llama3-8b"])
        assert mqa < gqa

    def test_kv_bytes_scales_with_batch_and_seq(self):
        p = MODEL_KV_PROFILES["llama3-8b"]
        base = kvc.kv_bytes(p, 4, 1024)
        assert kvc.kv_bytes(p, 8, 1024) == 2 * base       # batch
        assert kvc.kv_bytes(p, 4, 2048) == 2 * base       # seq_len

    def test_quantization_shrinks_footprint(self):
        fp16 = resolve_kv_profile("llama3-8b")
        fp8 = resolve_kv_profile("llama3-8b", {"kv_bytes_per_elem": 1.0})
        assert kvc.kv_bytes_per_token(fp8) == kvc.kv_bytes_per_token(fp16) / 2.0


# ---------------------------------------------------------------------------
# Pressure + regions
# ---------------------------------------------------------------------------

class TestPressureRegions:
    def test_pressure_ratio(self):
        assert kvc.kv_pressure(50, 100) == 0.5
        assert kvc.kv_pressure(0, 100) == 0.0
        assert kvc.kv_pressure(100, 0) == 1.5   # no budget → maxed

    def test_region_ordering(self):
        assert kvc.pressure_region(0.3) == kvc.PressureRegion.LOW
        assert kvc.pressure_region(0.75) == kvc.PressureRegion.ELEVATED
        assert kvc.pressure_region(0.92) == kvc.PressureRegion.THROTTLING
        assert kvc.pressure_region(0.99) == kvc.PressureRegion.PREEMPTION

    def test_thresholds_configurable(self):
        cfg = {"kv_pressure_elevated": 0.5}
        assert kvc.pressure_region(0.55, cfg) == kvc.PressureRegion.ELEVATED

    def test_ttft_multiplier_grows_convexly(self):
        # Equally-spaced points from the elevated threshold (0.7) to 1.0.
        m = [kvc.pressure_ttft_multiplier(p) for p in (0.7, 0.775, 0.85, 0.925, 1.0)]
        assert m[0] == 1.0                          # at/below elevated
        assert m[1] < m[2] < m[3] < m[4]            # rises into exhaustion
        diffs = [m[i + 1] - m[i] for i in range(len(m) - 1)]
        assert all(diffs[i] < diffs[i + 1] for i in range(len(diffs) - 1))  # convex

    def test_batch_efficiency_degrades_under_pressure(self):
        assert kvc.pressure_batch_efficiency(0.5) == 1.0
        assert kvc.pressure_batch_efficiency(1.0) < kvc.pressure_batch_efficiency(0.8) < 1.0


# ---------------------------------------------------------------------------
# PagedAttention fragmentation (internal slack only)
# ---------------------------------------------------------------------------

class TestFragmentation:
    def test_block_slack_nonnegative_and_scales(self):
        bpt = kvc.kv_bytes_per_token(MODEL_KV_PROFILES["llama3-8b"])
        s1 = kvc.block_slack_bytes(4, 7.5, bpt)
        s2 = kvc.block_slack_bytes(8, 7.5, bpt)
        assert 0 < s1 < s2

    def test_fragmentation_frac_bounded(self):
        p = MODEL_KV_PROFILES["llama3-8b"]
        frac = kvc.fragmentation_frac(8, 1024, p)
        assert 0.0 <= frac < 1.0
        # Longer sequences amortize the fixed tail-block slack → less fragmentation.
        assert kvc.fragmentation_frac(8, 4096, p) < kvc.fragmentation_frac(8, 256, p)


# ---------------------------------------------------------------------------
# Prefix cache reuse
# ---------------------------------------------------------------------------

class TestPrefixReuse:
    def test_hit_rate_rises_with_overlap(self):
        lo = kvc.prefix_hit_rate(0.1, 1.0)
        hi = kvc.prefix_hit_rate(0.9, 1.0)
        assert hi > lo

    def test_locality_gates_hit_rate(self):
        # Same overlap, cold route (low locality) → much lower hit rate.
        warm = kvc.prefix_hit_rate(0.9, 1.0)
        cold = kvc.prefix_hit_rate(0.9, 0.05)
        assert cold < warm * 0.2

    def test_prefill_savings_capped(self):
        cap = kv_value("prefix_max_prefill_savings_frac")
        assert kvc.prefill_savings_frac(1.0) == cap
        assert kvc.prefill_savings_frac(0.0) == 0.0

    def test_lost_tokens_and_cold_penalty(self):
        lost = kvc.lost_prefill_tokens(4096, 0.8)
        assert lost == 4096 * 0.8
        pen = kvc.cold_route_penalty_ms(lost)
        assert pen == lost * kv_value("prefill_cost_per_token_ms")
        assert pen > 0


# ---------------------------------------------------------------------------
# Locality confidence dynamics (reuse-driven)
# ---------------------------------------------------------------------------

class TestLocalityConfidence:
    def test_warmup_grows_toward_one(self):
        c = 0.05
        for _ in range(50):
            c = kvc.locality_confidence_step(c, reused=True)
        assert c > 0.9

    def test_decay_when_not_reused(self):
        c = 0.8
        c2 = kvc.locality_confidence_step(c, reused=False)
        assert c2 < c

    def test_cold_route_can_rewarm(self):
        # A near-zero (cold) route still begins to warm once reused.
        c = kv_value("cold_route_confidence")
        assert kvc.locality_confidence_step(c, reused=True) > c


# ---------------------------------------------------------------------------
# Preemption / recompute
# ---------------------------------------------------------------------------

class TestPreemption:
    def test_no_preemption_below_throttling(self):
        assert kvc.preemption_probability(0.5) == 0.0
        assert kvc.preemption_probability(0.85) == 0.0

    def test_preemption_prob_rises(self):
        assert kvc.preemption_probability(0.95) > 0.0
        assert kvc.preemption_probability(1.0) > kvc.preemption_probability(0.95)

    def test_recompute_not_free(self):
        assert kvc.recompute_penalty_ms(3, 2048) > 0
        assert kvc.recompute_penalty_ms(0, 2048) == 0


# ---------------------------------------------------------------------------
# Telemetry confidence → routing aggressiveness
# ---------------------------------------------------------------------------

class TestTelemetryTiers:
    def test_tiers(self):
        assert kvc.telemetry_confidence_tier(True, True, True, True) == "high"
        assert kvc.telemetry_confidence_tier(False, False, False, True) == "medium"
        assert kvc.telemetry_confidence_tier(False, False, False, False) == "low"

    def test_aggressiveness_decreases_with_confidence(self):
        hi = kvc.routing_aggressiveness("high")
        med = kvc.routing_aggressiveness("medium")
        lo = kvc.routing_aggressiveness("low")
        assert hi >= med >= lo
        assert hi == 1.0


# ---------------------------------------------------------------------------
# Cache-aware migration policy
# ---------------------------------------------------------------------------

class TestAffinityPolicy:
    def test_preserve_when_cache_loss_exceeds_queue_gain(self):
        # Long warm prefix, modest queue gain → preserve affinity (don't reroute).
        assert kvc.should_preserve_affinity(
            overlap=0.9, shared_prefix_tokens=4096,
            expected_queue_gain_ms=50.0, locality_confidence=1.0,
        ) is True

    def test_break_when_queue_gain_dominates(self):
        assert kvc.should_preserve_affinity(
            overlap=0.9, shared_prefix_tokens=4096,
            expected_queue_gain_ms=1e9, locality_confidence=1.0,
        ) is False

    def test_severe_imbalance_overrides(self):
        assert kvc.should_preserve_affinity(
            overlap=0.9, shared_prefix_tokens=4096,
            expected_queue_gain_ms=1.0, locality_confidence=1.0,
            severe_imbalance=True,
        ) is False

    def test_low_confidence_little_to_lose(self):
        # Cold route → little reuse to lose → don't bother preserving.
        assert kvc.should_preserve_affinity(
            overlap=0.9, shared_prefix_tokens=4096,
            expected_queue_gain_ms=10.0, locality_confidence=0.02,
        ) is False


# ---------------------------------------------------------------------------
# Integration: simulator-level emergent behaviour
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_kv_pressure_telemetry_emitted(self):
        sc = load_scenario("kv_exhaustion_preemption_storm")
        sim = ClusterSimulator(sc.config, seed=42)
        ticks = sim.run(20)
        svc_states = [
            t.cluster_state.regions["us-east"].services.get("longctx-inference")
            for t in ticks
        ]
        pressures = [s.kv_cache_usage for s in svc_states if s and s.kv_cache_usage]
        assert pressures and max(pressures) > 0.9   # enters throttling/preemption

    def test_preemption_storm_under_exhaustion(self):
        sc = load_scenario("kv_exhaustion_preemption_storm")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(24)
        wl = next(iter(sim._cluster.workloads.values()))
        assert wl.cache.preemption.cumulative_count > 0   # preemption storm
        assert wl.cache.eviction.cumulative_evictions > 0

    def test_cold_reroute_penalty_spikes_ttft(self):
        # A migration on a high-overlap warm workload injects a large cold-route
        # TTFT penalty (lost reusable prefill) and resets locality confidence.
        sc = load_scenario("prefix_affinity_energy_arbitrage")
        sim = ClusterSimulator(sc.config, seed=42)
        sim.run(8)   # warm the prefix cache on the home route
        wl = sim._cluster.workloads["chat-affinity-wl"]
        hit_before = wl.cache.prefix.hit_rate
        assert hit_before > 0.3   # cache was warm
        assert sim.migrate_workload("chat-affinity-wl", "us-west") is True
        # Cold route carries a substantial pending penalty (lost prefill) and a
        # near-zero locality confidence — TTFT will pay for the lost reuse.
        assert wl.cache.affinity.cold_reroute_count == 1
        assert wl.cache.affinity.cold_route_penalty_ms > 100.0
        assert wl.cache.locality.confidence <= kv_value("cold_route_confidence") + 1e-9

    def test_high_overlap_beats_low_overlap_ttft(self):
        # Same scenario, only prefix overlap differs → higher overlap warms a
        # better prefix cache → lower steady-state TTFT.
        def steady_ttft(overlap: float) -> float:
            sc = load_scenario("queue_surge_latency_sensitive")
            cfg = sc.config
            for w in cfg.workloads:
                w["prefix_overlap"] = overlap
                w["avg_seq_len_tokens"] = 4096
            sim = ClusterSimulator(cfg, seed=42)
            sim.run(8)   # warm before the surge
            return sim._cluster.regions["us-east"].queues[0].ttft_p50_ms

        assert steady_ttft(0.9) < steady_ttft(0.1)

    def test_locality_confidence_warms_over_time(self):
        sc = load_scenario("prefix_affinity_energy_arbitrage")
        sim = ClusterSimulator(sc.config, seed=42)
        wl = sim._cluster.workloads["chat-affinity-wl"]
        c0 = wl.cache.locality.confidence
        sim.run(8)
        assert wl.cache.locality.confidence > c0   # reuse-driven warmup

    def test_low_telemetry_hides_kv_internals(self):
        sc = load_scenario("kv_exhaustion_preemption_storm")
        cfg = sc.config
        cfg.serving_config = dict(cfg.serving_config or {})
        cfg.serving_config["kv_telemetry_tier"] = "low"
        sim = ClusterSimulator(cfg, seed=42)
        ticks = sim.run(8)
        svc = ticks[-1].cluster_state.regions["us-east"].services.get("longctx-inference")
        # Missing telemetry → KV usage not visible (must NOT imply 'no pressure').
        assert svc.kv_cache_usage is None
        # ...but the underlying pressure state still exists internally.
        wl = sim._cluster.workloads["longctx-wl"]
        assert wl.cache.pressure.pressure > 0.0

    def test_cache_aware_routing_beats_naive_arbitrage(self):
        # In a migratable high-overlap arbitrage scenario, the naive energy-greedy
        # policy reroutes for cheap power, loses the warm prefix cache on each hop,
        # and pays it back in TTFT — losing to the affinity-preserving policy.
        result = ConstraintBenchmarkRunner().run_scenario(
            "prefix_affinity_energy_arbitrage", steps=24, seed=42
        )
        agg = result.report.aggregated
        greedy = agg["greedy_energy"]
        ca = agg["constraint_aware"]
        assert greedy.total_migrations > ca.total_migrations
        assert greedy.p99_latency_ms > ca.p99_latency_ms


# ---------------------------------------------------------------------------
# Calibration metadata: no hidden constants
# ---------------------------------------------------------------------------

class TestCalibrationMetadata:
    def test_kv_params_have_full_provenance(self):
        assert KV_CACHE_PARAMS
        for name, p in KV_CACHE_PARAMS.items():
            assert p.source and p.source_type and p.calibration_notes, name
            assert p.confidence in ("high", "medium", "low"), name

    def test_combined_table_includes_kv_group(self):
        table = calibration_table()
        groups = {row.get("group") for row in table}
        assert "kv_cache" in groups and "serving" in groups

    def test_model_profile_table_has_provenance(self):
        rows = model_profile_table()
        assert rows
        for r in rows:
            assert r["kv_bytes_per_token"] > 0
            assert r["source"] and r["source_type"]

    def test_kv_params_overridable(self):
        assert kv_value("kv_pressure_throttling", {"kv_pressure_throttling": 0.8}) == 0.8
