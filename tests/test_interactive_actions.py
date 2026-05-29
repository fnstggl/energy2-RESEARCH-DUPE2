"""Part E — interactive queue / proxy / prefix candidate actions (deepened).

Covers the first-class interactive-relief actions and their gates:
  Queue:  SCALE_REPLICAS, PREWARM_REPLICA, RESERVE_CAPACITY_FOR_SLA, REROUTE,
          and "mild pressure must not scale".
  Proxy:  proxy-bottleneck detection, useless-scale suppression
          (blocked_useless_scale_proxy_bottleneck), reroute when a safe peer
          exists, KEEP when none, and the suppression reason in the report.
  Prefix: high cache hit-rate blocks the cache-destroying energy move
          (PRESERVE_AFFINITY), low hit-rate allows it.

Generator-level assertions exercise the candidate set; engine-level assertions
exercise the full gate pipeline (no pinned benchmark numbers).
"""

from __future__ import annotations

from datetime import datetime, timezone

from aurelius.constraints.classifier import ConstraintConfig
from aurelius.constraints.engine import (
    ConstraintAwareEngine,
    _gen_energy,
    _gen_queue,
    _proxy_bottleneck,
    _region_has_idle_capacity,
)
from aurelius.sla.actions import ActionType
from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    EnergyState,
    GPUState,
    InferenceServiceState,
    NodeState,
    Provenance,
    RegionState,
)

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _p():
    return Provenance(source="test", fetched_at=NOW, confidence="high")


def _gpu(uuid, node, region, util=None):
    return GPUState(gpu_uuid=uuid, node_id=node, region=region, timestamp=NOW,
                    provenance=_p(), gpu_index=0, util_pct=util)


def _svc(sid, region, **kw):
    return InferenceServiceState(service_id=sid, engine="vllm", timestamp=NOW,
                                 provenance=_p(), region=region, **kw)


def _region(rid, *, price=None, pct=None, spare=None, services=None, gpus=None,
            allocatable=4, allocated=2):
    energy = (EnergyState(region=rid, timestamp=NOW, provenance=_p(),
                          price_per_mwh=price, price_percentile=pct)
              if price is not None else None)
    node_gpus = {g.gpu_uuid: g for g in (gpus or [])}
    n = NodeState(node_id=f"{rid}-n0", region=rid, timestamp=NOW, provenance=_p(),
                  gpu_capacity=allocatable, gpu_allocatable=allocatable,
                  gpu_allocated=allocated, gpus=node_gpus)
    return RegionState(region=rid, timestamp=NOW, provenance=_p(), nodes={n.node_id: n},
                       services=services or {}, energy=energy, spare_capacity_pct=spare)


def _cluster(regions):
    return ClusterState(timestamp=NOW, provenance=_p(), regions=regions)


def _assessment(binding=ConstraintType.QUEUE):
    return ConstraintAssessment(timestamp=NOW, provenance=_p(), region=None,
                                scores={}, binding_constraint=binding,
                                confidence=0.9, missing_signals=[], rationale="")


def _engine():
    return ConstraintAwareEngine(classifier_config=ConstraintConfig(hysteresis_count=1))


# ===========================================================================
# Part A — queue surge candidates
# ===========================================================================

def test_1_queue_surge_emits_scale_when_sla_risk_real():
    svc = _svc("s", "r", priority_tier="standard", requests_waiting=80,
               queue_time_p95_ms=1500, p99_latency_ms=1800)
    state = _cluster({"r": _region("r", spare=30, services={"s": svc})})
    types = {c.action_type for c in _gen_queue(svc, state, _assessment())}
    assert ActionType.SCALE_REPLICAS in types


def test_2_critical_queue_surge_emits_prewarm():
    svc = _svc("s", "r", priority_tier="critical", requests_waiting=90,
               queue_time_p95_ms=2000, p99_latency_ms=2500)
    state = _cluster({"r": _region("r", spare=30, services={"s": svc})})
    types = {c.action_type for c in _gen_queue(svc, state, _assessment())}
    assert ActionType.PREWARM_REPLICA in types, "critical interactive should pre-warm"
    assert ActionType.SCALE_REPLICAS in types


def test_3_queue_surge_emits_reserve_when_batch_crowds():
    crit = _svc("crit", "r", priority_tier="critical", requests_waiting=80,
                queue_time_p95_ms=1500)
    batch = _svc("batch", "r", priority_tier="batch")
    state = _cluster({"r": _region("r", spare=20, services={"crit": crit, "batch": batch})})
    types = {c.action_type for c in _gen_queue(crit, state, _assessment())}
    assert ActionType.RESERVE_CAPACITY in types, "batch co-tenant should trigger reserve"


def test_4_mild_queue_pressure_does_not_scale_batch():
    # Batch workload, only mild queue pressure -> economic gate must block the
    # expensive scale-up (no SLA-safe goodput/$ gain).
    svc = _svc("b", "r", priority_tier="batch", workload_type="batch_inference",
               requests_waiting=2, queue_time_p95_ms=60, p99_latency_ms=120)
    state = _cluster({"r": _region("r", spare=60, services={"b": svc},
                                   gpus=[_gpu("a", "r-n0", "r", util=30)])})
    recs = [r for r in _engine().run(state).recommendations if r.workload_id == "b"]
    assert recs and recs[0].action_type != ActionType.SCALE_REPLICAS.value


# ===========================================================================
# Part B — proxy bottleneck
# ===========================================================================

def test_5_proxy_saturation_high_low_gpu_detects_proxy():
    svc = _svc("s", "r", proxy_saturation=2.5)
    # Low GPU utilization -> proxy is the binding bottleneck.
    low = _cluster({"r": _region("r", spare=30, services={"s": svc},
                                 gpus=[_gpu("a", "r-n0", "r", util=20)])})
    bound, sat, util = _proxy_bottleneck(svc, low)
    assert bound and sat == 2.5 and util == 20.0
    # High GPU utilization -> replicas ALSO bind -> not proxy-suppressed.
    svc2 = _svc("s", "r", proxy_saturation=2.5)
    high = _cluster({"r": _region("r", spare=30, services={"s": svc2},
                                  gpus=[_gpu("a", "r-n0", "r", util=96)])})
    bound2, _, util2 = _proxy_bottleneck(svc2, high)
    assert not bound2 and util2 == 96.0


def test_6_proxy_bottleneck_suppresses_scale():
    svc = _svc("s", "hot", priority_tier="critical", requests_waiting=120,
               queue_time_p95_ms=3000, p99_latency_ms=4000, proxy_saturation=3.0,
               latency_sensitive=True)
    state = _cluster({
        "hot": _region("hot", spare=40, services={"s": svc},
                       gpus=[_gpu("a", "hot-n0", "hot", util=25)]),
    })
    result = _engine().run(state)
    reasons = [r["reject_reason"] for r in result.rejected if r["service_id"] == "s"]
    assert any("blocked_useless_scale_proxy_bottleneck" in r for r in reasons)
    rec = [r for r in result.recommendations if r.workload_id == "s"][0]
    assert rec.action_type != ActionType.SCALE_REPLICAS.value


def test_7_proxy_bottleneck_emits_reroute_when_safe_peer():
    svc = _svc("s", "hot", priority_tier="standard", requests_waiting=90,
               queue_time_p95_ms=2000, proxy_saturation=2.4)
    state = _cluster({
        "hot": _region("hot", spare=8, services={"s": svc},
                       gpus=[_gpu("a", "hot-n0", "hot", util=20)]),
        "cool": _region("cool", spare=70, gpus=[_gpu("b", "cool-n0", "cool", util=15)]),
    })
    cands = _gen_queue(svc, state, _assessment())
    reroute = [c for c in cands if c.action_type == ActionType.REROUTE]
    assert reroute and reroute[0].target_region == "cool"
    assert reroute[0].metadata.get("flag") == "proxy_bottleneck"


def test_8_proxy_bottleneck_keeps_when_no_safe_target():
    # Proxy-bound, single region (no safe peer) -> scale suppressed, no reroute,
    # spread suppressed (useless vs proxy) -> KEEP.
    svc = _svc("s", "hot", priority_tier="critical", requests_waiting=120,
               queue_time_p95_ms=3000, p99_latency_ms=4000, proxy_saturation=3.0,
               latency_sensitive=True)
    state = _cluster({
        "hot": _region("hot", spare=8, services={"s": svc},
                       gpus=[_gpu("a", "hot-n0", "hot", util=20)]),
    })
    rec = [r for r in _engine().run(state).recommendations if r.workload_id == "s"][0]
    assert rec.is_noop and rec.action_type == ActionType.KEEP.value


def test_9_report_includes_proxy_bottleneck_reason():
    svc = _svc("s", "hot", priority_tier="critical", requests_waiting=120,
               queue_time_p95_ms=3000, p99_latency_ms=4000, proxy_saturation=3.0,
               latency_sensitive=True)
    state = _cluster({"hot": _region("hot", spare=40, services={"s": svc},
                                     gpus=[_gpu("a", "hot-n0", "hot", util=25)])})
    d = _engine().run(state).to_dict()
    blob = str(d["rejected"])
    assert "blocked_useless_scale_proxy_bottleneck" in blob


def test_proxy_branch_skips_spread_but_replica_branch_keeps_it():
    # Proxy-bound branch must not offer SPREAD (useless vs proxy)...
    svc = _svc("s", "hot", proxy_saturation=2.4, requests_waiting=90,
               queue_time_p95_ms=2000)
    state = _cluster({"hot": _region("hot", spare=8, services={"s": svc},
                                     gpus=[_gpu("a", "hot-n0", "hot", util=20)]),
                      "cool": _region("cool", spare=70)})
    assert ActionType.SPREAD not in {c.action_type for c in _gen_queue(svc, state, _assessment())}
    # ...but a non-proxy queue surge keeps SPREAD as an in-region option.
    svc2 = _svc("s2", "r", requests_waiting=80, queue_time_p95_ms=1500)
    st2 = _cluster({"r": _region("r", spare=40, services={"s2": svc2})})
    assert ActionType.SPREAD in {c.action_type for c in _gen_queue(svc2, st2, _assessment())}


def test_region_idle_capacity_helper():
    full = _cluster({"r": _region("r", allocatable=4, allocated=4)})
    assert not _region_has_idle_capacity(full, "r")
    spare = _cluster({"r": _region("r", allocatable=4, allocated=2)})
    assert _region_has_idle_capacity(spare, "r")


# ===========================================================================
# Part C — prefix affinity (engine-side energy candidates)
# ===========================================================================

def _energy_state(hit_rate):
    svc = _svc("chat", "expensive", prefix_cache_hit_rate=hit_rate,
               p99_latency_ms=300, priority_tier="standard")
    return _cluster({
        "expensive": _region("expensive", price=200, pct=98, spare=60,
                             services={"chat": svc},
                             gpus=[_gpu("a", "expensive-n0", "expensive", util=60)]),
        "cheap": _region("cheap", price=40, pct=8, spare=70,
                        gpus=[_gpu("b", "cheap-n0", "cheap", util=30)]),
    })


def test_10_high_prefix_hit_rate_blocks_energy_move():
    state = _energy_state(0.85)
    svc = state.regions["expensive"].services["chat"]
    cands = _gen_energy(svc, state, _assessment(ConstraintType.ENERGY))
    types = {c.action_type for c in cands}
    assert ActionType.CHOOSE_CHEAPER_REGION not in types
    assert ActionType.PRESERVE_AFFINITY in types
    assert cands[0].metadata.get("reason") == "preserve_affinity_high_cache_hit_rate"


def test_11_low_prefix_hit_rate_allows_energy_move():
    state = _energy_state(0.15)
    svc = state.regions["expensive"].services["chat"]
    types = {c.action_type for c in _gen_energy(svc, state, _assessment(ConstraintType.ENERGY))}
    assert ActionType.CHOOSE_CHEAPER_REGION in types


def test_engine_blocks_cache_destroying_energy_move():
    state = _energy_state(0.9)
    rec = [r for r in _engine().run(state).recommendations if r.workload_id == "chat"][0]
    assert rec.target_region in (None, "expensive")
    assert rec.action_type != ActionType.CHOOSE_CHEAPER_REGION.value
