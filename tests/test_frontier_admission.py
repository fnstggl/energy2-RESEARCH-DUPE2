"""Tests for the WorkloadAdmissionGate (aurelius/frontier/admission.py).

Research basis: arXiv:2604.11001 "Flow-Controlled Scheduling for LLM
Inference with Provable Stability Guarantees."

These tests verify:
- Gate-disabled default: always ADMIT regardless of pressure.
- Latency-critical SLA classes: always ADMIT even when enabled.
- Fail-open on missing telemetry: ADMIT with 'none' confidence.
- ADMIT under low pressure.
- DEFER under KV soft-ceiling pressure.
- DEFER under queue tail pressure.
- DEFER under elevated timeout rate (conservative mode).
- REJECT only for best-effort workloads at KV hard ceiling saturation.
- Batch evaluation produces per-workload decisions.
- AdmissionDecision serialization via to_dict().
- AdmissionGateConfig validation.
- SLA-class exemption coverage.
"""

from __future__ import annotations

import pytest

from aurelius.frontier.admission import (
    ADMISSION_ADMIT,
    ADMISSION_DEFER,
    ADMISSION_REJECT,
    AdmissionDecision,
    AdmissionGateConfig,
    evaluate_admission,
    evaluate_admission_batch,
)
from aurelius.frontier.dynamic_models import ServingTelemetryTick

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tick(
    *,
    kv: float = 0.5,
    q99: float = 500.0,
    timeout_pct: float = 1.0,
    confidence: str = "medium",
    ts: float = 0.0,
) -> ServingTelemetryTick:
    """Build a ServingTelemetryTick with controllable KV / queue signals.

    ``mean_utilization`` is used as the KV proxy (matches risk.py convention).
    """
    return ServingTelemetryTick(
        timestamp_s=ts,
        mean_utilization=kv,
        queue_p99_ms=q99,
        timeout_pct=timeout_pct,
        telemetry_confidence=confidence,
    )


def _window(n: int = 6, *, kv: float = 0.5, q99: float = 500.0,
             timeout_pct: float = 1.0, confidence: str = "medium"):
    return [_tick(kv=kv, q99=q99, timeout_pct=timeout_pct,
                  confidence=confidence, ts=float(i))
            for i in range(n)]


def _enabled_cfg(**kwargs) -> AdmissionGateConfig:
    return AdmissionGateConfig(enabled=True, **kwargs)


# ---------------------------------------------------------------------------
# Gate-disabled behaviour
# ---------------------------------------------------------------------------

class TestGateDisabled:
    def test_disabled_admits_regardless_of_kv_pressure(self):
        cfg = AdmissionGateConfig(enabled=False)
        window = _window(kv=0.99)  # extreme KV pressure
        d = evaluate_admission(sla_class="best_effort", window=window, config=cfg)
        assert d.action == ADMISSION_ADMIT
        assert not d.gate_enabled
        assert "gate_disabled" in d.reason_codes

    def test_disabled_admits_for_all_sla_classes(self):
        cfg = AdmissionGateConfig(enabled=False)
        window = _window(kv=0.95, q99=1800.0)
        for sla in ["realtime_inference", "llm_batch_inference",
                    "training", "best_effort"]:
            d = evaluate_admission(sla_class=sla, window=window, config=cfg)
            assert d.action == ADMISSION_ADMIT, sla

    def test_default_config_is_disabled(self):
        """Default config must have enabled=False (shadow-mode-only)."""
        cfg = AdmissionGateConfig()
        assert not cfg.enabled


# ---------------------------------------------------------------------------
# Latency-critical SLA class exemption
# ---------------------------------------------------------------------------

class TestLatencyCriticalExemption:
    CRITICAL_CLASSES = [
        "latency_critical",
        "realtime",
        "realtime_inference",
        "interactive",
    ]

    def test_realtime_always_admits_under_load(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.5, kv_hard_ceiling=0.7)
        window = _window(kv=0.99, q99=1900.0, timeout_pct=8.0)
        for sla in self.CRITICAL_CLASSES:
            d = evaluate_admission(sla_class=sla, window=window, config=cfg)
            assert d.action == ADMISSION_ADMIT, f"{sla} must always admit"
            assert "latency_critical_sla_exempt" in d.reason_codes

    def test_realtime_gate_enabled_true(self):
        cfg = _enabled_cfg()
        window = _window(kv=0.99)
        d = evaluate_admission(sla_class="realtime_inference", window=window, config=cfg)
        assert d.gate_enabled is True
        assert d.action == ADMISSION_ADMIT


# ---------------------------------------------------------------------------
# Fail-open on missing telemetry
# ---------------------------------------------------------------------------

class TestFailOpenOnMissingTelemetry:
    def test_empty_window_admits(self):
        cfg = _enabled_cfg()
        # Empty window — telemetry confidence check fails → fail-open.
        d = evaluate_admission(sla_class="llm_batch_inference", window=[], config=cfg)
        assert d.action == ADMISSION_ADMIT
        assert "insufficient_telemetry_fail_open" in d.reason_codes
        assert d.confidence == "none"

    def test_unknown_confidence_window_admits(self):
        cfg = _enabled_cfg(min_telemetry_confidence="low")
        window = [_tick(confidence="unknown") for _ in range(4)]
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        # "unknown" doesn't meet "low" confidence threshold → fail-open
        assert d.action == ADMISSION_ADMIT
        assert "insufficient_telemetry_fail_open" in d.reason_codes


# ---------------------------------------------------------------------------
# ADMIT under low pressure
# ---------------------------------------------------------------------------

class TestAdmitLowPressure:
    def test_low_kv_low_queue_admits(self):
        cfg = _enabled_cfg()
        window = _window(kv=0.3, q99=200.0)
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        assert d.action == ADMISSION_ADMIT
        assert d.gate_enabled is True

    def test_admit_carries_pressure_scores(self):
        cfg = _enabled_cfg()
        window = _window(kv=0.3, q99=200.0)
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        assert d.kv_pressure_score is not None
        assert 0.0 <= d.kv_pressure_score <= 1.0
        assert d.queue_pressure_score is not None
        assert 0.0 <= d.queue_pressure_score <= 1.0

    def test_batch_admits_at_moderate_utilization(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80)
        window = _window(kv=0.70, q99=400.0)
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        assert d.action == ADMISSION_ADMIT


# ---------------------------------------------------------------------------
# DEFER under KV soft-ceiling pressure
# ---------------------------------------------------------------------------

class TestDeferKVPressure:
    def test_kv_above_soft_ceiling_defers_batch(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        window = _window(kv=0.85, q99=300.0)
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        assert d.action == ADMISSION_DEFER
        assert d.defer_until_ms > 0.0

    def test_defer_reason_codes_include_kv(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        window = _window(kv=0.87)
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        assert d.action == ADMISSION_DEFER
        assert any("kv" in r for r in d.reason_codes)

    def test_defer_ms_positive_minimum(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95,
                            max_defer_ms=1000.0)
        window = _window(kv=0.85)
        d = evaluate_admission(sla_class="llm_batch_inference", window=window, config=cfg)
        assert d.action == ADMISSION_DEFER
        assert d.defer_until_ms >= 100.0  # minimum 100 ms enforced

    def test_high_kv_longer_defer_than_low_kv(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        low_window = _window(kv=0.82)
        high_window = _window(kv=0.93)
        d_low = evaluate_admission(sla_class="llm_batch_inference",
                                   window=low_window, config=cfg)
        d_high = evaluate_admission(sla_class="llm_batch_inference",
                                    window=high_window, config=cfg)
        assert d_low.action == ADMISSION_DEFER
        assert d_high.action == ADMISSION_DEFER
        assert d_high.defer_until_ms >= d_low.defer_until_ms


# ---------------------------------------------------------------------------
# DEFER under queue tail pressure
# ---------------------------------------------------------------------------

class TestDeferQueuePressure:
    def test_high_queue_p99_defers_batch(self):
        cfg = _enabled_cfg(  # noqa: F841
            kv_soft_ceiling=0.80,  # KV pressure absent
            risk_config=type(
                "R", (), {"max_queue_p99_ms": 2000.0,
                          "max_timeout_pct": 10.0,
                          "max_latency_p99_ms": None,
                          "min_telemetry_confidence": "low"}
            )(),  # noqa: E501 — quick anonymous struct
        )
        # Build proper config without anonymous struct:
        from aurelius.frontier.risk import RiskConfig
        cfg2 = _enabled_cfg(kv_soft_ceiling=0.80)
        cfg2 = AdmissionGateConfig(
            enabled=True,
            kv_soft_ceiling=0.80,
            kv_hard_ceiling=0.95,
            queue_soft_fraction=0.65,
            risk_config=RiskConfig(max_queue_p99_ms=2000.0),
        )
        # q99 = 1400 ms → 1400/2000 = 0.70 > 0.65 queue_soft_fraction → DEFER
        window = _window(kv=0.40, q99=1400.0)
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg2)
        assert d.action == ADMISSION_DEFER

    def test_low_queue_p99_does_not_defer(self):
        from aurelius.frontier.risk import RiskConfig
        cfg = AdmissionGateConfig(
            enabled=True,
            kv_soft_ceiling=0.80,
            queue_soft_fraction=0.65,
            risk_config=RiskConfig(max_queue_p99_ms=2000.0),
        )
        window = _window(kv=0.40, q99=500.0)  # 500/2000 = 0.25 < 0.65
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        assert d.action == ADMISSION_ADMIT


# ---------------------------------------------------------------------------
# Conservative mode from elevated timeout
# ---------------------------------------------------------------------------

class TestConservativeModeTimeout:
    def test_high_timeout_triggers_conservative_defer(self):
        cfg = _enabled_cfg(
            kv_soft_ceiling=0.80,      # KV pressure absent (kv=0.60)
            timeout_conservative_threshold_pct=5.0,
        )
        window = _window(kv=0.60, q99=300.0, timeout_pct=7.0)
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        assert d.action == ADMISSION_DEFER
        assert "timeout_pct_elevated" in d.reason_codes

    def test_low_timeout_does_not_trigger_conservative(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80,
                            timeout_conservative_threshold_pct=5.0)
        window = _window(kv=0.60, q99=300.0, timeout_pct=2.0)
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        assert d.action == ADMISSION_ADMIT


# ---------------------------------------------------------------------------
# REJECT for best-effort at KV saturation
# ---------------------------------------------------------------------------

class TestRejectBestEffort:
    def test_reject_best_effort_at_kv_saturation(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        # kv=0.99 / hard_ceiling=0.95 → kv_score ≈ 1.0 → REJECT for best_effort
        window = _window(kv=0.99, q99=300.0)
        d = evaluate_admission(sla_class="best_effort", window=window, config=cfg)
        assert d.action == ADMISSION_REJECT
        assert "reject_best_effort_kv_saturated" in d.reason_codes

    def test_batch_not_rejected_at_high_kv(self):
        """Non-best-effort workloads are DEFER, not REJECT, even at high KV."""
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        window = _window(kv=0.99, q99=300.0)
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        # High KV → DEFER (not REJECT) for non-best-effort
        assert d.action in (ADMISSION_DEFER, ADMISSION_ADMIT)
        assert d.action != ADMISSION_REJECT

    def test_background_maintenance_rejected_at_kv_saturation(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        window = _window(kv=0.99)
        d = evaluate_admission(sla_class="background_maintenance",
                               window=window, config=cfg)
        assert d.action == ADMISSION_REJECT


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

class TestBatchEvaluation:
    def test_batch_returns_per_workload_decisions(self):
        cfg = _enabled_cfg()
        window = _window(kv=0.50)
        workloads = [
            ("job-1", "llm_batch_inference"),
            ("job-2", "realtime_inference"),
            ("job-3", "best_effort"),
        ]
        results = evaluate_admission_batch(workloads=workloads, window=window, config=cfg)
        assert set(results.keys()) == {"job-1", "job-2", "job-3"}
        for wid, d in results.items():
            assert isinstance(d, AdmissionDecision)

    def test_batch_latency_critical_always_admit(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        window = _window(kv=0.99)
        workloads = [
            ("rt-1", "realtime_inference"),
            ("rt-2", "interactive"),
        ]
        results = evaluate_admission_batch(workloads=workloads,
                                           window=window, config=cfg)
        for wid, d in results.items():
            assert d.action == ADMISSION_ADMIT, wid

    def test_batch_mixed_sla_class_decisions(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, kv_hard_ceiling=0.95)
        window = _window(kv=0.88)  # above soft ceiling → DEFER for non-realtime
        workloads = [
            ("rt", "realtime_inference"),    # exempt
            ("batch", "llm_batch_inference"),  # eligible for DEFER
        ]
        results = evaluate_admission_batch(workloads=workloads,
                                           window=window, config=cfg)
        assert results["rt"].action == ADMISSION_ADMIT
        assert results["batch"].action == ADMISSION_DEFER


# ---------------------------------------------------------------------------
# AdmissionDecision serialization
# ---------------------------------------------------------------------------

class TestAdmissionDecisionSerialization:
    def test_to_dict_has_required_keys(self):
        d = AdmissionDecision(
            action=ADMISSION_ADMIT,
            sla_class="llm_batch_inference",
            defer_until_ms=0.0,
            kv_pressure_score=0.3,
            queue_pressure_score=0.2,
            reason_codes=("no_pressure",),
            confidence="medium",
            gate_enabled=True,
        )
        as_dict = d.to_dict()
        required = {
            "action", "sla_class", "defer_until_ms",
            "kv_pressure_score", "queue_pressure_score",
            "reason_codes", "confidence", "gate_enabled",
        }
        assert required <= set(as_dict.keys())

    def test_to_dict_reason_codes_is_list(self):
        d = AdmissionDecision(
            action=ADMISSION_DEFER,
            sla_class="training",
            defer_until_ms=500.0,
            kv_pressure_score=0.85,
            queue_pressure_score=0.4,
            reason_codes=("kv_above_soft_ceiling", "defer_due_to_load"),
            confidence="medium",
            gate_enabled=True,
        )
        assert isinstance(d.to_dict()["reason_codes"], list)

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            AdmissionDecision(
                action="UNKNOWN_ACTION",
                sla_class="batch",
                defer_until_ms=0.0,
                kv_pressure_score=None,
                queue_pressure_score=None,
                reason_codes=(),
                confidence="none",
                gate_enabled=True,
            )


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

class TestAdmissionGateConfig:
    def test_default_disabled(self):
        cfg = AdmissionGateConfig()
        assert cfg.enabled is False

    def test_kv_soft_below_hard(self):
        cfg = AdmissionGateConfig()
        assert cfg.kv_soft_ceiling < cfg.kv_hard_ceiling

    def test_max_defer_ms_positive(self):
        cfg = AdmissionGateConfig()
        assert cfg.max_defer_ms > 0.0

    def test_custom_config_persists(self):
        cfg = AdmissionGateConfig(
            enabled=True,
            kv_soft_ceiling=0.75,
            kv_hard_ceiling=0.90,
            max_defer_ms=3000.0,
        )
        assert cfg.enabled
        assert cfg.kv_soft_ceiling == 0.75
        assert cfg.kv_hard_ceiling == 0.90
        assert cfg.max_defer_ms == 3000.0


# ---------------------------------------------------------------------------
# Single-tick window (edge case)
# ---------------------------------------------------------------------------

class TestSingleTickWindow:
    def test_single_tick_low_pressure_admits(self):
        cfg = _enabled_cfg(min_window_for_trends=1)
        window = [_tick(kv=0.3, q99=200.0, confidence="medium")]
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        assert d.action == ADMISSION_ADMIT

    def test_single_tick_high_kv_defers(self):
        cfg = _enabled_cfg(kv_soft_ceiling=0.80, min_window_for_trends=1)
        window = [_tick(kv=0.90, q99=200.0, confidence="medium")]
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        assert d.action == ADMISSION_DEFER


# ---------------------------------------------------------------------------
# Pressure scores are in [0, 1]
# ---------------------------------------------------------------------------

class TestPressureScoreBounds:
    @pytest.mark.parametrize("kv,q99", [
        (0.1, 100.0),
        (0.5, 500.0),
        (0.8, 1000.0),
        (0.95, 1900.0),
        (0.99, 2000.0),
    ])
    def test_pressure_scores_bounded(self, kv, q99):
        cfg = _enabled_cfg()
        window = _window(kv=kv, q99=q99)
        d = evaluate_admission(sla_class="llm_batch_inference",
                               window=window, config=cfg)
        if d.kv_pressure_score is not None:
            assert 0.0 <= d.kv_pressure_score <= 1.0, f"kv={kv}"
        if d.queue_pressure_score is not None:
            assert 0.0 <= d.queue_pressure_score <= 1.0, f"q99={q99}"
