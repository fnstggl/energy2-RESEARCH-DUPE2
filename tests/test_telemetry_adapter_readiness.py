"""Telemetry adapter readiness audit — tests.

Asserts:

 1.  The `telemetry_tick_from_inference_service_state` bridge exists and
     produces a valid :class:`ServingTelemetryTick`.
 2.  Missing connector fields stay ``None`` in the resulting tick (no
     silent zero-fill; `docs/PILOT_TELEMETRY_CONTRACT.md` §1).
 3.  A vLLM `InferenceServiceState` (the canonical real-telemetry leaf
     produced by `aurelius/connectors/vllm.py`) bridges to a tick that
     :func:`validate_dynamic_window` accepts as the basis for a window.
 4.  A Ray Serve `InferenceServiceState` (which carries native replica
     count + percentile histograms) bridges with `active_replicas` set.
 5.  A Triton `InferenceServiceState` (averages, no histograms by
     default) bridges with the documented partial coverage — p95/p99
     latency stay ``None``.
 6.  The bridge does NOT invent `mean_utilization` — it stays None
     unless the caller passes it explicitly (DCGM is the real source).
 7.  The bridge does NOT invent `scale_events_delta` / `churn_delta` —
     they stay None unless passed in.
 8.  `gpu_hours_delta` is derived only when both `replicas` and
     `tick_duration_s` are provided.
 9.  `telemetry_confidence` is inherited from the connector
     `Provenance.confidence` when present.
 10. The audit doc / JSON exist and the doc contains no production-
     savings claims.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from aurelius.frontier import (
    ServingTelemetryTick,
    build_serving_telemetry_window,
    telemetry_tick_from_inference_service_state,
    validate_dynamic_window,
)
from aurelius.state.models import InferenceServiceState, Provenance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DOC = os.path.join(REPO_ROOT, "docs",
                         "TELEMETRY_ADAPTER_READINESS_AUDIT.md")
AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "telemetry_adapter_readiness_summary.json")


def _provenance(confidence="medium", source="unit"):
    return Provenance(source=source, fetched_at=datetime.now(timezone.utc),
                       confidence=confidence, is_sandbox=False)


def _vllm_state(*, requests_running=10.0, requests_waiting=2.0,
                p99_latency_ms=900.0, p95_latency_ms=700.0,
                p50_latency_ms=300.0,
                ttft_p99_ms=200.0, kv_cache_usage=0.55,
                tokens_per_s=120.0,
                error_rate_pct=1.2) -> InferenceServiceState:
    return InferenceServiceState(
        service_id="vllm/llama3-70b", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(),
        requests_running=requests_running,
        requests_waiting=requests_waiting,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        p99_latency_ms=p99_latency_ms,
        ttft_p99_ms=ttft_p99_ms,
        kv_cache_usage=kv_cache_usage,
        tokens_per_s=tokens_per_s,
        error_rate_pct=error_rate_pct)


def _ray_state(*, replicas=4) -> InferenceServiceState:
    return InferenceServiceState(
        service_id="ray/serve/llama", engine="ray_serve",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="high"),
        requests_waiting=3.0,
        p50_latency_ms=80.0, p95_latency_ms=150.0, p99_latency_ms=300.0,
        error_rate_pct=0.5, replicas=replicas)


def _triton_state() -> InferenceServiceState:
    # Triton default: averages only, no p95/p99.
    return InferenceServiceState(
        service_id="triton/resnet", engine="triton",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="low"),
        requests_waiting=1.0,
        p50_latency_ms=45.0, p95_latency_ms=None, p99_latency_ms=None,
        queue_time_p50_ms=5.0, error_rate_pct=0.1)


# ---------------------------------------------------------------------------
# 1 — bridge exists and produces a valid tick
# ---------------------------------------------------------------------------

def test_bridge_exists_and_returns_serving_telemetry_tick():
    state = _vllm_state()
    tick = telemetry_tick_from_inference_service_state(
        state, tick_duration_s=60.0)
    assert isinstance(tick, ServingTelemetryTick)
    assert tick.source == "inference_service_state"


# ---------------------------------------------------------------------------
# 2 — missing fields stay None
# ---------------------------------------------------------------------------

def test_bridge_preserves_none_for_missing_fields():
    # Minimal state: only the engine + timestamp + provenance are set.
    state = InferenceServiceState(
        service_id="vllm/minimal", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance())
    tick = telemetry_tick_from_inference_service_state(state)
    for f in ("observed_rps", "queue_p99_ms", "latency_p99_ms",
              "timeout_pct", "mean_utilization", "active_replicas",
              "gpu_hours_delta", "scale_events_delta", "churn_delta"):
        assert getattr(tick, f) is None, f"{f} should be None"


# ---------------------------------------------------------------------------
# 3 — vLLM bridge feeds validate_dynamic_window
# ---------------------------------------------------------------------------

def test_vllm_bridge_can_validate_as_dynamic_window():
    # Build 12 ticks from a steady-state vLLM observation; validate_dynamic_window
    # checks: observed_rps, queue_p99_ms, active_replicas.
    state = _vllm_state()
    ticks = [
        telemetry_tick_from_inference_service_state(
            state, timestamp_s=i * 60.0, tick_duration_s=60.0,
            # Caller passes mean_utilization from DCGM + replicas from K8s.
            mean_utilization=0.65,
            scale_events_delta=0, churn_delta=0.0,
        )
        for i in range(12)
    ]
    # Replicas is not exposed by vLLM — the caller must supply it via the
    # K8s connector. Override on each tick using the dict-builder path.
    enriched_dicts = [
        {**t.to_dict(), "active_replicas": 4} for t in ticks
    ]
    window = build_serving_telemetry_window(enriched_dicts)
    out = validate_dynamic_window(window, min_ticks=8)
    assert out.ok, f"window should validate: {out.reason}"


# ---------------------------------------------------------------------------
# 4 — Ray Serve carries native replicas
# ---------------------------------------------------------------------------

def test_ray_serve_bridge_carries_active_replicas():
    state = _ray_state(replicas=6)
    tick = telemetry_tick_from_inference_service_state(
        state, tick_duration_s=60.0, mean_utilization=0.70)
    assert tick.active_replicas == 6
    # latency percentiles round-trip
    assert tick.latency_p99_ms == 300.0
    assert tick.latency_p95_ms == 150.0


# ---------------------------------------------------------------------------
# 5 — Triton has documented partial coverage (no p95/p99)
# ---------------------------------------------------------------------------

def test_triton_bridge_partial_p95_p99_latency_stays_none():
    state = _triton_state()
    tick = telemetry_tick_from_inference_service_state(
        state, tick_duration_s=60.0, mean_utilization=0.55)
    assert tick.latency_p95_ms is None
    assert tick.latency_p99_ms is None
    # queue_p50_ms (Triton average) propagates
    assert tick.queue_p50_ms == 5.0


# ---------------------------------------------------------------------------
# 6 — bridge does NOT invent mean_utilization
# ---------------------------------------------------------------------------

def test_bridge_does_not_invent_mean_utilization():
    state = _vllm_state()
    tick = telemetry_tick_from_inference_service_state(
        state, tick_duration_s=60.0)
    assert tick.mean_utilization is None, (
        "mean_utilization must come from DCGM, not be invented")


# ---------------------------------------------------------------------------
# 7 — bridge does NOT invent scale_events_delta / churn_delta
# ---------------------------------------------------------------------------

def test_bridge_does_not_invent_scale_events_or_churn():
    state = _vllm_state()
    tick = telemetry_tick_from_inference_service_state(
        state, tick_duration_s=60.0)
    assert tick.scale_events_delta is None
    assert tick.churn_delta is None


# ---------------------------------------------------------------------------
# 8 — gpu_hours_delta derived only when replicas + duration present
# ---------------------------------------------------------------------------

def test_gpu_hours_delta_derived_when_replicas_and_duration_present():
    state = _ray_state(replicas=4)
    tick = telemetry_tick_from_inference_service_state(
        state, tick_duration_s=60.0)
    # 4 replicas * 60s = 240 GPU-s = 240/3600 = 0.0667 GPU-h
    assert tick.gpu_hours_delta is not None
    assert abs(tick.gpu_hours_delta - 4.0 * 60.0 / 3600.0) < 1e-9


def test_gpu_hours_delta_stays_none_when_duration_missing():
    state = _ray_state(replicas=4)
    tick = telemetry_tick_from_inference_service_state(state)
    # No tick_duration_s → no derivation.
    assert tick.gpu_hours_delta is None


# ---------------------------------------------------------------------------
# 9 — telemetry_confidence inherited from Provenance
# ---------------------------------------------------------------------------

def test_telemetry_confidence_inherited_from_provenance():
    state = InferenceServiceState(
        service_id="vllm/x", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="high"))
    tick = telemetry_tick_from_inference_service_state(state)
    assert tick.telemetry_confidence == "high"

    state_low = InferenceServiceState(
        service_id="vllm/y", engine="vllm",
        timestamp=datetime.now(timezone.utc),
        provenance=_provenance(confidence="low"))
    tick_low = telemetry_tick_from_inference_service_state(state_low)
    assert tick_low.telemetry_confidence == "low"


# ---------------------------------------------------------------------------
# 10 — audit doc + JSON exist and contain no production-savings claim
# ---------------------------------------------------------------------------

def test_audit_doc_exists_and_has_no_production_savings_claim():
    assert os.path.exists(AUDIT_DOC), f"missing {AUDIT_DOC}"
    with open(AUDIT_DOC, encoding="utf-8") as fh:
        text = fh.read().lower()
    forbidden = (
        "production savings",
        "guaranteed savings",
        "we save customers",
        "saved customers",
    )
    for phrase in forbidden:
        assert phrase not in text, (
            f"audit doc must not contain '{phrase}'")
    # Must contain at least one of the standard disclaimers.
    assert ("shadow" in text or "simulator" in text
            or "production-savings claim" in text)


def test_audit_json_well_formed():
    assert os.path.exists(AUDIT_JSON), f"missing {AUDIT_JSON}"
    with open(AUDIT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("scope", "adapters_inventoried", "coverage_matrix",
              "production_real_vs_simulated_only",
              "can_calibration_run_on_real_telemetry_today",
              "remaining_gaps_before_pilot_production_claim",
              "honesty_contract"):
        assert k in d, f"audit JSON missing key {k!r}"
    # honesty_contract must hold the non-negotiable claims.
    h = d["honesty_contract"]
    assert h["no_new_dataset"] is True
    assert h["no_engine_change"] is True
    assert h["no_real_execution_enabled"] is True
    assert h["no_production_savings_claim"] is True
    assert h["missing_fields_stay_none"] is True
