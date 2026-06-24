"""Tests for compound economic × queue scheduling backtest [run 2026-06-22-z].

Validates the compound system:
  - Queue layer: abs-conformal SRPT (run 2026-06-22-x/y)
  - Provisioning layer: economic scheduling (+25.75% vs SLA-aware, BENCHMARK_REGISTRY §1.1)

The compound is multiplicative because the two layers are orthogonal:
  compound_goodput/$ = abs_conformal_goodput/$ × economic_cost_factor

Key invariants tested:
  1. Compound > queue-only (economic factor > 1.0 always increases goodput/$)
  2. Compound > SLA-aware by > queue-only margin (economic adds to queue gain)
  3. North-star (+300% vs SLA-aware) NOT achieved by compound (honest finding)
  4. Economic factor needed > current factor (more savings required for north-star)
  5. run-t over-estimate is captured and corrected

Research basis:
  - BENCHMARK_REGISTRY §1.1: +25.75% vs sla_aware, -21.2% GPU-hours (Azure LLM 2024)
  - run 2026-06-22-y: abs-conformal +83.27% vs oracle SLA-aware (Azure)
  - run 2026-06-22-y: abs-conformal +111.55% vs oracle SLA-aware (BurstGPT)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY,
    NORTH_STAR_MULTIPLIER,
    CompoundEconomicQueueReport,
    _compute_compound_economic_queue,
    run_compound_economic_queue_azure_backtest,
    run_compound_economic_queue_burstgpt_backtest,
    run_sla_aware_abs_conformal_azure_backtest,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    f.close()
    return f.name


def _make_bimodal_records(
    n: int = 120,
    gap_s: float = 0.5,
    short_tokens: int = 50,
    long_tokens: int = 500,
) -> list[dict]:
    """Alternating short/long requests — strong ordering signal."""
    records = []
    for i in range(n):
        out_tok = short_tokens if i % 2 == 0 else long_tokens
        records.append({
            "request_arrival_ts_s": float(i) * gap_s,
            "output_tokens": out_tok,
            "input_tokens": 200,
            "model_id": "test-model",
        })
    return records


# ---------------------------------------------------------------------------
# Class 1: Report structure
# ---------------------------------------------------------------------------

class TestCompoundEconomicQueueReportStructure:
    """Validate CompoundEconomicQueueReport fields and serialization."""

    def _make_report(self) -> CompoundEconomicQueueReport:
        path = _write_jsonl(_make_bimodal_records(120))
        try:
            return run_compound_economic_queue_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_returns_correct_type(self):
        rpt = self._make_report()
        assert isinstance(rpt, CompoundEconomicQueueReport)

    def test_required_scalar_fields_present(self):
        rpt = self._make_report()
        for field in (
            "fifo_goodput_per_dollar",
            "sla_aware_oracle_goodput_per_dollar",
            "abs_conformal_goodput_per_dollar",
            "compound_goodput_per_dollar",
            "queue_vs_sla_aware_oracle_delta_pct",
            "compound_vs_sla_aware_oracle_delta_pct",
            "compound_vs_fifo_delta_pct",
            "economic_cost_factor",
            "north_star_target_pct",
            "economic_factor_needed_for_north_star",
            "run_t_compound_estimate_vs_fifo_pct",
            "corrected_compound_vs_fifo_pct",
            "over_estimate_factor",
        ):
            val = getattr(rpt, field)
            assert isinstance(val, (int, float)), f"{field} should be numeric, got {type(val)}"

    def test_north_star_achieved_is_bool(self):
        rpt = self._make_report()
        assert isinstance(rpt.north_star_achieved, bool)

    def test_north_star_target_is_300(self):
        rpt = self._make_report()
        assert rpt.north_star_target_pct == pytest.approx(300.0)

    def test_economic_cost_factor_matches_constant(self):
        rpt = self._make_report()
        assert rpt.economic_cost_factor == pytest.approx(ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY)

    def test_shadow_tag_present(self):
        rpt = self._make_report()
        assert "shadow" in rpt.shadow_tag

    def test_to_dict_serializable(self):
        rpt = self._make_report()
        d = rpt.to_dict()
        json_str = json.dumps(d)
        assert json_str
        restored = json.loads(json_str)
        assert "compound_vs_sla_aware_oracle_delta_pct" in restored
        assert "north_star_achieved" in restored

    def test_to_dict_all_float_fields_finite(self):
        rpt = self._make_report()
        d = rpt.to_dict()
        for k, v in d.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"non-finite in to_dict(): {k}={v}"

    def test_trace_name_compound_azure(self):
        rpt = run_compound_economic_queue_azure_backtest()
        assert "compound" in rpt.trace
        assert "azure" in rpt.trace

    def test_trace_name_compound_burstgpt(self):
        path = _write_jsonl(_make_bimodal_records(120))
        try:
            rpt = run_compound_economic_queue_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert "compound" in rpt.trace
        assert "burstgpt" in rpt.trace

    def test_economic_factor_source_documented(self):
        rpt = self._make_report()
        assert "BENCHMARK_REGISTRY" in rpt.economic_cost_factor_source
        assert len(rpt.economic_cost_factor_source) > 20


# ---------------------------------------------------------------------------
# Class 2: Compound invariants
# ---------------------------------------------------------------------------

class TestCompoundInvariants:
    """Verify structural invariants of the compound calculation."""

    def _azure_report(self) -> CompoundEconomicQueueReport:
        return run_compound_economic_queue_azure_backtest()

    def test_compound_exceeds_queue_only(self):
        """Economic factor > 1.0 means compound always beats queue-alone."""
        rpt = self._azure_report()
        assert rpt.compound_goodput_per_dollar > rpt.abs_conformal_goodput_per_dollar

    def test_compound_exceeds_sla_aware_oracle(self):
        """Compound must be > oracle SLA-aware (queue alone already beats it)."""
        rpt = self._azure_report()
        assert rpt.compound_goodput_per_dollar > rpt.sla_aware_oracle_goodput_per_dollar

    def test_compound_exceeds_fifo(self):
        rpt = self._azure_report()
        assert rpt.compound_goodput_per_dollar > rpt.fifo_goodput_per_dollar

    def test_compound_delta_exceeds_queue_delta(self):
        """Compound vs SLA-aware must exceed queue-only vs SLA-aware."""
        rpt = self._azure_report()
        assert rpt.compound_vs_sla_aware_oracle_delta_pct > rpt.queue_vs_sla_aware_oracle_delta_pct

    def test_compound_vs_fifo_exceeds_abs_vs_fifo(self):
        rpt = self._azure_report()
        assert rpt.compound_vs_fifo_delta_pct > rpt.abs_vs_fifo_delta_pct

    def test_compound_goodput_equals_abs_times_factor(self):
        """Direct arithmetic check: compound = abs_conformal × economic_factor."""
        rpt = self._azure_report()
        expected = rpt.abs_conformal_goodput_per_dollar * rpt.economic_cost_factor
        assert rpt.compound_goodput_per_dollar == pytest.approx(expected, rel=1e-6)

    def test_economic_factor_needed_exceeds_current(self):
        """More economic gain is needed for north-star than currently exists."""
        rpt = self._azure_report()
        assert rpt.economic_factor_needed_for_north_star > rpt.economic_cost_factor

    def test_economic_factor_needed_is_positive(self):
        rpt = self._azure_report()
        assert rpt.economic_factor_needed_for_north_star > 1.0

    def test_run_t_estimate_exceeds_corrected(self):
        """run-t over-estimate must be larger than the corrected compound vs FIFO."""
        rpt = self._azure_report()
        assert rpt.run_t_compound_estimate_vs_fifo_pct > rpt.corrected_compound_vs_fifo_pct

    def test_over_estimate_factor_exceeds_one(self):
        rpt = self._azure_report()
        assert rpt.over_estimate_factor > 1.0

    def test_north_star_not_achieved_on_azure(self):
        """Primary finding: compound (+130%) does not meet north-star (+300%)."""
        rpt = self._azure_report()
        assert not rpt.north_star_achieved

    def test_compound_delta_positive_on_azure(self):
        rpt = self._azure_report()
        assert rpt.compound_vs_sla_aware_oracle_delta_pct > 0.0


# ---------------------------------------------------------------------------
# Class 3: North-star analysis correctness
# ---------------------------------------------------------------------------

class TestNorthStarAnalysis:
    """Verify the north-star analysis fields are computed correctly."""

    def _azure_report(self) -> CompoundEconomicQueueReport:
        return run_compound_economic_queue_azure_backtest()

    def test_factor_needed_implies_north_star_when_applied(self):
        """Applying factor_needed to abs_conformal should reach exactly north-star."""
        rpt = self._azure_report()
        hypothetical_compound = (
            rpt.abs_conformal_goodput_per_dollar * rpt.economic_factor_needed_for_north_star
        )
        hypothetical_delta = (
            (hypothetical_compound - rpt.sla_aware_oracle_goodput_per_dollar)
            / rpt.sla_aware_oracle_goodput_per_dollar * 100.0
        )
        assert hypothetical_delta == pytest.approx(300.0, rel=1e-4)

    def test_factor_delta_vs_current_is_positive(self):
        """Additional economic gain needed must be positive."""
        rpt = self._azure_report()
        assert rpt.economic_factor_needed_delta_vs_current > 0.0

    def test_factor_delta_vs_current_matches_needed_minus_current(self):
        rpt = self._azure_report()
        expected = rpt.economic_factor_needed_for_north_star - rpt.economic_cost_factor
        assert rpt.economic_factor_needed_delta_vs_current == pytest.approx(expected, rel=1e-6)

    def test_north_star_achieved_false_when_compound_below_300(self):
        rpt = self._azure_report()
        assert rpt.compound_vs_sla_aware_oracle_delta_pct < 300.0
        assert not rpt.north_star_achieved

    def test_north_star_achieved_true_when_factor_sufficient(self):
        """With a 2.18× economic factor, north-star should be achieved."""
        queue_rpt = run_sla_aware_abs_conformal_azure_backtest()
        sufficient_factor = queue_rpt.sla_aware_oracle_goodput_per_dollar * NORTH_STAR_MULTIPLIER / queue_rpt.abs_conformal_goodput_per_dollar + 0.01
        compound_rpt = _compute_compound_economic_queue(queue_rpt, economic_cost_factor=sufficient_factor)
        assert compound_rpt.north_star_achieved

    def test_queue_vs_sla_aware_matches_run_y(self):
        """Queue-only result should match the run-y abs-conformal finding (+83%)."""
        rpt = self._azure_report()
        # run-y Azure result: +83.27% vs oracle SLA-aware
        # Allow 1% tolerance for floating-point consistency
        assert rpt.queue_vs_sla_aware_oracle_delta_pct == pytest.approx(83.27, abs=1.0)

    def test_corrected_compound_equals_compound_vs_fifo(self):
        """corrected_compound_vs_fifo_pct must equal compound_vs_fifo_delta_pct."""
        rpt = self._azure_report()
        assert rpt.corrected_compound_vs_fifo_pct == pytest.approx(
            rpt.compound_vs_fifo_delta_pct, rel=1e-6
        )


# ---------------------------------------------------------------------------
# Class 4: BurstGPT cross-validation
# ---------------------------------------------------------------------------

class TestBurstGPTCrossValidation:
    """Cross-validate compound invariants on BurstGPT fixture."""

    def _burstgpt_report(self) -> CompoundEconomicQueueReport:
        path = _write_jsonl(_make_bimodal_records(n=120, short_tokens=30, long_tokens=800))
        try:
            return run_compound_economic_queue_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_compound_exceeds_queue_on_burstgpt(self):
        rpt = self._burstgpt_report()
        assert rpt.compound_goodput_per_dollar >= rpt.abs_conformal_goodput_per_dollar

    def test_compound_exceeds_fifo_on_burstgpt(self):
        rpt = self._burstgpt_report()
        assert rpt.compound_goodput_per_dollar > rpt.fifo_goodput_per_dollar

    def test_north_star_not_achieved_on_burstgpt(self):
        """With current economic factor (+25.75%), north-star is not achieved."""
        rpt = self._burstgpt_report()
        assert not rpt.north_star_achieved

    def test_sla_s_is_30s_on_burstgpt(self):
        rpt = self._burstgpt_report()
        assert rpt.sla_s == pytest.approx(30.0)

    def test_economic_factor_needed_positive_on_burstgpt(self):
        rpt = self._burstgpt_report()
        assert rpt.economic_factor_needed_for_north_star > 1.0


# ---------------------------------------------------------------------------
# Class 5: Customization and edge cases
# ---------------------------------------------------------------------------

class TestCustomizationAndEdgeCases:
    """Validate parameter propagation and edge cases."""

    def test_custom_economic_factor_above_one_increases_compound(self):
        """Higher economic factor → higher compound goodput/$."""
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt_low = run_compound_economic_queue_burstgpt_backtest(
                jsonl_path=path, economic_cost_factor=1.10
            )
            rpt_high = run_compound_economic_queue_burstgpt_backtest(
                jsonl_path=path, economic_cost_factor=2.00
            )
        finally:
            os.unlink(path)
        assert rpt_high.compound_goodput_per_dollar > rpt_low.compound_goodput_per_dollar

    def test_economic_factor_of_one_compound_equals_queue(self):
        """Economic factor = 1.0 → compound exactly equals queue abs-conformal."""
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt = run_compound_economic_queue_burstgpt_backtest(
                jsonl_path=path, economic_cost_factor=1.0
            )
        finally:
            os.unlink(path)
        assert rpt.compound_goodput_per_dollar == pytest.approx(
            rpt.abs_conformal_goodput_per_dollar, rel=1e-9
        )

    def test_north_star_achieved_with_large_economic_factor(self):
        """With 5× economic factor, north-star is achieved."""
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt = run_compound_economic_queue_burstgpt_backtest(
                jsonl_path=path, economic_cost_factor=5.0
            )
        finally:
            os.unlink(path)
        assert rpt.north_star_achieved

    def test_total_requests_propagated(self):
        n = 60
        path = _write_jsonl(_make_bimodal_records(n))
        try:
            rpt = run_compound_economic_queue_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.total_requests == n

    def test_compute_compound_from_run_y_report(self):
        """_compute_compound_economic_queue can accept an existing run-y report."""
        queue_rpt = run_sla_aware_abs_conformal_azure_backtest()
        compound_rpt = _compute_compound_economic_queue(queue_rpt)
        assert isinstance(compound_rpt, CompoundEconomicQueueReport)
        assert compound_rpt.economic_cost_factor == pytest.approx(
            ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY
        )
