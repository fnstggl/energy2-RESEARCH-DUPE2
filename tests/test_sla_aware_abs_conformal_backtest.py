"""Tests for SLA-aware vs abs-conformal head-to-head [run 2026-06-22-y].

Validates the six-discipline comparison that directly answers the north-star
question: does abs-conformal (live prior) outperform SLA-aware scheduling?

Disciplines tested:
  1. fifo               — FIFO baseline
  2. sla_aware_oracle   — binary short/long using actual token counts
  3. sla_aware_live     — binary short/long using running-median prior
  4. rel_conformal_live — decoupled hybrid + rel-error conformal, live prior
  5. abs_conformal_live — decoupled hybrid + abs-error conformal, live prior
  6. conformal_oracle   — decoupled hybrid + conformal, oracle prior

Research basis:
  - arXiv:2506.14851 (Probabilistic Demand Modeling, Jun 2026)
  - arXiv:2605.16867 (GoodServe, May 2026)
  - arXiv:2604.11001 (Flow-Controlled Scheduling, Apr 2026)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_ABS_TARGET_P90_TOKENS,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    SLAAwareAbsConformalReport,
    run_sla_aware_abs_conformal_azure_backtest,
    run_sla_aware_abs_conformal_burstgpt_backtest,
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


def _make_uniform_records(n: int = 60, tokens: int = 100, gap_s: float = 1.0) -> list[dict]:
    """Uniform-length requests — ordering has no benefit."""
    return [
        {
            "request_arrival_ts_s": float(i) * gap_s,
            "output_tokens": tokens,
            "input_tokens": 100,
            "model_id": "uniform-model",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Class 1: Report structure
# ---------------------------------------------------------------------------

class TestSLAAwareAbsConformalReportStructure:
    """Validate SLAAwareAbsConformalReport fields and serialization."""

    def _make_report(self) -> SLAAwareAbsConformalReport:
        path = _write_jsonl(_make_bimodal_records(120))
        try:
            return run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_returns_correct_type(self):
        rpt = self._make_report()
        assert isinstance(rpt, SLAAwareAbsConformalReport)

    def test_six_discipline_dicts_present(self):
        rpt = self._make_report()
        for attr in ("fifo", "sla_aware_oracle", "sla_aware_live",
                     "rel_conformal_live", "abs_conformal_live", "conformal_oracle"):
            assert isinstance(getattr(rpt, attr), dict), f"missing dict: {attr}"

    def test_six_goodput_fields_present(self):
        rpt = self._make_report()
        for field in (
            "fifo_goodput_per_dollar",
            "sla_aware_oracle_goodput_per_dollar",
            "sla_aware_live_goodput_per_dollar",
            "rel_conformal_goodput_per_dollar",
            "abs_conformal_goodput_per_dollar",
            "oracle_goodput_per_dollar",
        ):
            val = getattr(rpt, field)
            assert isinstance(val, float) and val >= 0, f"invalid {field}: {val}"

    def test_delta_fields_present(self):
        rpt = self._make_report()
        for field in (
            "sla_aware_oracle_delta_pct",
            "sla_aware_live_delta_pct",
            "rel_conformal_delta_pct",
            "abs_conformal_delta_pct",
            "oracle_delta_pct",
            "abs_vs_sla_aware_oracle_delta_pct",
            "abs_vs_sla_aware_live_delta_pct",
            "abs_vs_rel_delta_pct",
        ):
            val = getattr(rpt, field)
            assert isinstance(val, float), f"not a float: {field}"

    def test_retention_fields_between_0_and_100(self):
        rpt = self._make_report()
        for field in (
            "abs_vs_oracle_retention_pct",
            "rel_vs_oracle_retention_pct",
            "sla_aware_oracle_retention_pct",
            "sla_aware_live_retention_pct",
        ):
            val = getattr(rpt, field)
            assert 0.0 <= val <= 100.0, f"{field}={val} out of [0,100]"

    def test_shadow_tag_present(self):
        rpt = self._make_report()
        assert "shadow" in rpt.shadow_tag

    def test_to_dict_serializable(self):
        rpt = self._make_report()
        d = rpt.to_dict()
        json_str = json.dumps(d)
        assert json_str
        restored = json.loads(json_str)
        assert restored["trace"] == rpt.trace
        assert "abs_vs_sla_aware_oracle_delta_pct" in restored

    def test_trace_name_azure(self):
        rpt = run_sla_aware_abs_conformal_azure_backtest()
        assert "azure" in rpt.trace

    def test_trace_name_burstgpt(self):
        path = _write_jsonl(_make_bimodal_records(120))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert "burstgpt" in rpt.trace

    def test_total_requests_matches_input(self):
        n = 80
        path = _write_jsonl(_make_bimodal_records(n))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.total_requests == n

    def test_sla_s_matches_default(self):
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(DEFAULT_BURSTGPT_SLA_S)

    def test_calibrator_diagnostics_non_negative(self):
        rpt = self._make_report()
        assert rpt.abs_mean_alpha >= 0.0
        assert rpt.rel_mean_alpha >= 0.0
        assert rpt.abs_p90_abs_err_tokens >= 0.0


# ---------------------------------------------------------------------------
# Class 2: Ordering invariants
# ---------------------------------------------------------------------------

class TestOrderingInvariants:
    """Verify discipline ordering relationships on strong-signal bimodal fixture."""

    def _bimodal_azure_report(self) -> SLAAwareAbsConformalReport:
        return run_sla_aware_abs_conformal_azure_backtest()

    def test_oracle_beats_fifo_on_azure(self):
        rpt = self._bimodal_azure_report()
        assert rpt.oracle_goodput_per_dollar > rpt.fifo_goodput_per_dollar

    def test_abs_conformal_beats_fifo_on_azure(self):
        rpt = self._bimodal_azure_report()
        assert rpt.abs_conformal_goodput_per_dollar > rpt.fifo_goodput_per_dollar

    def test_sla_aware_oracle_beats_fifo_on_azure(self):
        rpt = self._bimodal_azure_report()
        assert rpt.sla_aware_oracle_goodput_per_dollar > rpt.fifo_goodput_per_dollar

    def test_abs_conformal_beats_sla_aware_oracle_on_azure(self):
        """Primary finding: abs-conformal live > oracle SLA-aware on Azure."""
        rpt = self._bimodal_azure_report()
        assert rpt.abs_conformal_goodput_per_dollar > rpt.sla_aware_oracle_goodput_per_dollar, (
            f"abs_conformal={rpt.abs_conformal_goodput_per_dollar:.2f} should beat "
            f"sla_aware_oracle={rpt.sla_aware_oracle_goodput_per_dollar:.2f}"
        )

    def test_abs_conformal_beats_sla_aware_live_on_azure(self):
        """Abs-conformal beats live-prior SLA-aware on Azure."""
        rpt = self._bimodal_azure_report()
        assert rpt.abs_conformal_goodput_per_dollar > rpt.sla_aware_live_goodput_per_dollar

    def test_abs_conformal_beats_rel_conformal_on_azure(self):
        """Abs-conformal outperforms rel-conformal (replicates run-x finding)."""
        rpt = self._bimodal_azure_report()
        assert rpt.abs_conformal_goodput_per_dollar >= rpt.rel_conformal_goodput_per_dollar

    def test_abs_vs_sla_aware_oracle_delta_positive_on_azure(self):
        rpt = self._bimodal_azure_report()
        assert rpt.abs_vs_sla_aware_oracle_delta_pct > 0.0

    def test_abs_vs_sla_aware_live_delta_positive_on_azure(self):
        rpt = self._bimodal_azure_report()
        assert rpt.abs_vs_sla_aware_live_delta_pct > 0.0

    def test_oracle_retention_above_90pct_on_azure(self):
        """Abs-conformal retains ≥90% of oracle goodput on Azure (replicates run-x ~97.8%)."""
        rpt = self._bimodal_azure_report()
        assert rpt.abs_vs_oracle_retention_pct >= 90.0, (
            f"Oracle retention {rpt.abs_vs_oracle_retention_pct:.1f}% < 90%"
        )


# ---------------------------------------------------------------------------
# Class 3: BurstGPT cross-validation
# ---------------------------------------------------------------------------

class TestBurstGPTCrossValidation:
    """Cross-validate ordering on BurstGPT HF bimodal fixture."""

    def _burstgpt_report(self) -> SLAAwareAbsConformalReport:
        path = _write_jsonl(_make_bimodal_records(n=120, short_tokens=30, long_tokens=800))
        try:
            return run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_abs_conformal_at_least_fifo_on_burstgpt(self):
        """Abs-conformal is never worse than FIFO on BurstGPT fixture."""
        rpt = self._burstgpt_report()
        # At low utilization, all disciplines converge; at high utilization abs_conformal wins.
        assert rpt.abs_conformal_goodput_per_dollar >= rpt.fifo_goodput_per_dollar

    def test_abs_conformal_at_least_sla_aware_oracle_on_burstgpt(self):
        rpt = self._burstgpt_report()
        assert rpt.abs_conformal_goodput_per_dollar >= rpt.sla_aware_oracle_goodput_per_dollar

    def test_abs_vs_sla_aware_live_non_negative_on_burstgpt(self):
        rpt = self._burstgpt_report()
        assert rpt.abs_vs_sla_aware_live_delta_pct >= 0.0

    def test_sla_s_is_30s(self):
        rpt = self._burstgpt_report()
        assert rpt.sla_s == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Class 4: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case safety: uniform workloads, minimum size, custom parameters."""

    def test_minimum_two_requests(self):
        path = _write_jsonl(_make_bimodal_records(n=2))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.total_requests == 2

    def test_fewer_than_two_raises(self):
        path = _write_jsonl(_make_bimodal_records(n=1))
        try:
            with pytest.raises((ValueError, Exception)):
                run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_custom_target_p90_propagated(self):
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(
                jsonl_path=path, target_p90_abs_tokens=300.0
            )
        finally:
            os.unlink(path)
        assert rpt.target_p90_abs_tokens == pytest.approx(300.0)

    def test_custom_sla_propagated(self):
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(
                jsonl_path=path, sla_s=60.0
            )
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(60.0)

    def test_goodput_fields_finite(self):
        path = _write_jsonl(_make_bimodal_records(80))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        for field in (
            "fifo_goodput_per_dollar",
            "sla_aware_oracle_goodput_per_dollar",
            "sla_aware_live_goodput_per_dollar",
            "rel_conformal_goodput_per_dollar",
            "abs_conformal_goodput_per_dollar",
            "oracle_goodput_per_dollar",
        ):
            val = getattr(rpt, field)
            assert math.isfinite(val), f"{field}={val} not finite"

    def test_to_dict_all_numeric_fields_serializable(self):
        path = _write_jsonl(_make_bimodal_records(60))
        try:
            rpt = run_sla_aware_abs_conformal_burstgpt_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        d = rpt.to_dict()
        # All float/int values must be JSON-serializable (no NaN, no Inf)
        for k, v in d.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"non-finite in to_dict(): {k}={v}"
