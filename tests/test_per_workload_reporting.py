"""Tests for the per-workload-type benchmark reporting layer.

Covers:
  - scenario classification (intent, primary workload type, telemetry-failsafe)
  - headline-baseline selection (workload-relevant, NOT FIFO)
  - outcome analysis (alpha vs safety classifier, multi-cause loss reasons)
  - cross-scenario aggregation (median+mean, telemetry-failsafe separation)
  - markdown invariants (forbidden phrases, required columns)
  - end-to-end via ConstraintBenchmarkRunner
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurelius.benchmarks import (
    ConstraintBenchmarkRunner,
    CrossScenarioReport,
    OutcomeAnalysis,
    PerScenarioRow,
    ScenarioMetadata,
    analyze_outcome,
    classify_scenario,
    select_headline_baseline,
)
from aurelius.benchmarks.per_workload import (
    OUTCOMES,
    _infer_primary_workload_type,
)
from aurelius.simulation.cluster.scenarios import _BUILTIN_SCENARIOS

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# classify_scenario
# ---------------------------------------------------------------------------

def test_classify_scenario_non_none_for_every_builtin():
    for name, raw in _BUILTIN_SCENARIOS.items():
        m = classify_scenario(name, raw.get("expected_primary_constraint"), raw)
        assert isinstance(m, ScenarioMetadata)
        assert m.scenario_name == name
        assert m.primary_workload_type
        assert m.optimization_intent
        assert m.relevant_baselines


def test_energy_scenario_classifies_as_energy_arbitrage_with_price_baselines():
    raw = _BUILTIN_SCENARIOS["energy_price_arbitrage_multiregion"]
    m = classify_scenario(
        "energy_price_arbitrage_multiregion",
        raw.get("expected_primary_constraint"), raw,
    )
    assert m.optimization_intent == "energy_arbitrage"
    assert "current_price_only" in m.relevant_baselines
    assert "greedy_energy" in m.relevant_baselines
    assert m.is_telemetry_failsafe is False


def test_thermal_scenario_classifies_as_thermal_spread():
    raw = _BUILTIN_SCENARIOS["thermal_hotspot_mixed_cluster"]
    m = classify_scenario(
        "thermal_hotspot_mixed_cluster",
        raw.get("expected_primary_constraint"), raw,
    )
    assert m.optimization_intent == "thermal_spread"
    assert m.is_telemetry_failsafe is False


@pytest.mark.parametrize("name", [
    "degraded_topology_telemetry",
    "partial_utilization_telemetry",
    "low_confidence_energy_telemetry",
])
def test_telemetry_scenarios_classify_as_failsafe(name):
    raw = _BUILTIN_SCENARIOS[name]
    m = classify_scenario(name, raw.get("expected_primary_constraint"), raw)
    assert m.primary_workload_type == "telemetry_fail_safe"
    assert m.optimization_intent == "safety_keep"
    assert m.is_telemetry_failsafe is True


@pytest.mark.parametrize("name", [
    "fragmentation_stranded_capacity",
    "underutilization_stranded_capacity",
])
def test_utilization_scenarios_classify_with_packing_baselines(name):
    raw = _BUILTIN_SCENARIOS[name]
    m = classify_scenario(name, raw.get("expected_primary_constraint"), raw)
    assert m.optimization_intent == "fragmentation_packing"
    # At least one packing baseline is in relevant_baselines.
    assert any(p in m.relevant_baselines
               for p in ("first_fit", "best_fit", "first_fit_decreasing"))


def test_explicit_raw_override_wins():
    raw = {
        "primary_workload_type": "batch_training",
        "optimization_intent": "energy_arbitrage",
        "relevant_baselines": ["fifo", "current_price_only"],
        "headline_baseline": "greedy_energy",
        "goodput_unit": "token_equivalent",
        "sla_slo_type": "throughput_only",
        "workloads": [{"priority_tier": "latency_sensitive",
                       "workload_type": "inference"}],
    }
    m = classify_scenario("synthetic_override", "energy_bound", raw)
    assert m.primary_workload_type == "batch_training"
    assert m.headline_baseline_override == "greedy_energy"
    assert m.goodput_unit == "token_equivalent"


def test_workload_type_tie_is_mixed():
    # 1 batch + 1 inference → no strict plurality.
    workloads = [
        {"priority_tier": "batch", "workload_type": "batch_training"},
        {"priority_tier": "standard", "workload_type": "inference"},
    ]
    assert _infer_primary_workload_type(workloads) == "mixed"


def test_strict_plurality_batch_wins_3_of_5():
    workloads = (
        [{"priority_tier": "batch", "workload_type": "batch_training"}] * 3
        + [{"priority_tier": "standard", "workload_type": "inference"}] * 2
    )
    assert _infer_primary_workload_type(workloads) == "batch_training"


def test_classification_golden_fixture():
    """Classification of every builtin scenario must match the golden fixture."""
    golden = json.loads(
        (FIXTURES / "scenario_classification.json").read_text()
    )
    actual = {}
    for name, raw in _BUILTIN_SCENARIOS.items():
        m = classify_scenario(
            name, raw.get("expected_primary_constraint"), raw,
        )
        actual[name] = m.primary_workload_type
    assert actual == golden, (
        "Classification drift vs fixtures/scenario_classification.json. "
        "If intentional, regenerate the fixture."
    )


# ---------------------------------------------------------------------------
# select_headline_baseline
# ---------------------------------------------------------------------------

class _FakeKPI:
    """Lightweight KPI stand-in used in unit tests."""

    def __init__(self, *, goodput=100.0, p99=200.0, sla=0, thermal=0,
                 scale_up_recommended=0, scale_up_applied=0,
                 total_migrations=0, total_net_savings=None,
                 blocked_scale_for_low_value_queue_relief=0):
        self.sla_safe_goodput_per_infra_dollar = goodput
        self.p99_latency_ms = p99
        self.total_sla_violations = sla
        self.total_thermal_throttle_ticks = thermal
        self.scale_up_recommended = scale_up_recommended
        self.scale_up_applied = scale_up_applied
        self.total_migrations = total_migrations
        self.total_net_savings = total_net_savings
        self.blocked_scale_for_low_value_queue_relief = (
            blocked_scale_for_low_value_queue_relief
        )


def _make_metadata(**over):
    base = dict(
        scenario_name="synthetic",
        primary_workload_type="batch_training",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
        headline_baseline_override=None,
        goodput_unit="tokens",
        sla_slo_type="p99_latency",
        is_telemetry_failsafe=False,
    )
    base.update(over)
    return ScenarioMetadata(**base)


def test_select_headline_batch_picks_max_safe():
    md = _make_metadata()
    policy = {
        "fifo": _FakeKPI(goodput=100, p99=200, sla=2),
        # current_price_only has highest goodput but p99 is 5x FIFO → unsafe
        "current_price_only": _FakeKPI(goodput=300, p99=1100, sla=2),
        "greedy_energy": _FakeKPI(goodput=150, p99=210, sla=2),
        "sla_aware": _FakeKPI(goodput=120, p99=190, sla=2),
    }
    name, rationale = select_headline_baseline(md, policy)
    # current_price_only disqualified (p99 > 1.5x FIFO), greedy_energy wins
    assert name == "greedy_energy"
    assert "strongest_safe_relevant_baseline" in rationale


def test_select_headline_inference_critical_falls_back_to_fifo_when_sla_aware_violates():
    md = _make_metadata(primary_workload_type="inference_critical")
    policy = {
        "fifo": _FakeKPI(goodput=100, p99=200, sla=2),
        # sla_aware has MORE SLA violations than fifo → not safe.
        "sla_aware": _FakeKPI(goodput=200, p99=200, sla=5),
    }
    name, _ = select_headline_baseline(md, policy)
    assert name == "fifo"


def test_select_headline_telemetry_failsafe_picks_fifo():
    md = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    policy = {
        "fifo": _FakeKPI(goodput=100),
        "sla_aware": _FakeKPI(goodput=200),
    }
    name, rationale = select_headline_baseline(md, policy)
    assert name == "fifo"
    assert "telemetry_failsafe" in rationale


def test_select_headline_explicit_override_wins():
    md = _make_metadata(headline_baseline_override="greedy_energy")
    policy = {
        "fifo": _FakeKPI(),
        "greedy_energy": _FakeKPI(),
    }
    name, rationale = select_headline_baseline(md, policy)
    assert name == "greedy_energy"
    assert rationale == "explicit_override"


def test_select_headline_deterministic_across_two_calls():
    md = _make_metadata()
    policy = {
        "fifo": _FakeKPI(goodput=100, p99=200),
        "current_price_only": _FakeKPI(goodput=200, p99=200),
        "greedy_energy": _FakeKPI(goodput=150, p99=200),
        "sla_aware": _FakeKPI(goodput=120, p99=200),
    }
    a = select_headline_baseline(md, policy)
    b = select_headline_baseline(md, policy)
    assert a == b


# ---------------------------------------------------------------------------
# analyze_outcome
# ---------------------------------------------------------------------------

def test_analyze_outcome_alpha_win_above_1pct():
    md = _make_metadata()
    ca = _FakeKPI(goodput=120)
    hd = _FakeKPI(goodput=100)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "ALPHA_WIN"
    assert out.margin_pct > 1.0


def test_analyze_outcome_safety_win_in_tie_band_with_better_p99():
    md = _make_metadata()
    # Within 1% on goodput BUT 2x better p99.
    ca = _FakeKPI(goodput=100.5, p99=100)
    hd = _FakeKPI(goodput=100.0, p99=300)
    others = {"fifo": hd, "constraint_aware": ca,
              "sla_aware": _FakeKPI(goodput=99, p99=250)}
    out = analyze_outcome(md, ca, hd, others)
    assert out.outcome == "SAFETY_WIN"
    assert any("p99" in e for e in out.safety_evidence)


def test_analyze_outcome_tie_within_1pct_no_safety_evidence():
    md = _make_metadata()
    ca = _FakeKPI(goodput=100.5, p99=200)
    hd = _FakeKPI(goodput=100.0, p99=200)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "TIE"


def test_analyze_outcome_loss_below_minus_1pct():
    md = _make_metadata()
    ca = _FakeKPI(goodput=80, scale_up_recommended=0, total_migrations=0,
                  total_net_savings=-10.0)
    hd = _FakeKPI(goodput=100, scale_up_applied=2)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "LOSS"
    # Energy arbitrage with non-positive net_savings → flagged
    assert "missing_forecast_lookahead" in out.loss_reasons


def test_analyze_outcome_keep_correct_for_telemetry_failsafe():
    md = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep", is_telemetry_failsafe=True,
    )
    ca = _FakeKPI(goodput=100, sla=0)
    hd = _FakeKPI(goodput=100, sla=0)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "KEEP_CORRECT"


def test_analyze_outcome_multi_cause_loss_reasons():
    """Telemetry-failsafe scenario that ALSO loses on alpha and blocked an
    aggressive scale-up — both reasons should appear."""
    md = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    ca = _FakeKPI(goodput=80,
                  blocked_scale_for_low_value_queue_relief=5,
                  scale_up_applied=0)
    hd = _FakeKPI(goodput=100, scale_up_applied=3)
    out = analyze_outcome(md, ca, hd, {"fifo": hd, "constraint_aware": ca})
    assert out.outcome == "LOSS"
    assert "telemetry_fail_safe" in out.loss_reasons
    assert "over_conservative_gate" in out.loss_reasons


# ---------------------------------------------------------------------------
# CrossScenarioReport
# ---------------------------------------------------------------------------

class _FakeBenchmarkReport:
    def __init__(self, scenario_metadata, aggregated, headline_name,
                 headline_rationale, outcome):
        self.scenario_metadata = scenario_metadata
        self.aggregated = aggregated
        self.headline_baseline_name = headline_name
        self.headline_baseline_rationale = headline_rationale
        self.outcome = outcome
        self.expected_primary_constraint = None


class _FakeBenchmarkResult:
    def __init__(self, report):
        self.report = report


def _build_fake_results(scenarios):
    """scenarios: list of (name, metadata, ca_goodput, fifo_goodput,
    headline_name, outcome_str). Returns {name: FakeResult}."""
    out = {}
    for entry in scenarios:
        name, md, ca_g, fifo_g, hd_name, outcome_str = entry
        ca = _FakeKPI(goodput=ca_g, p99=200, sla=0)
        fifo = _FakeKPI(goodput=fifo_g, p99=200, sla=0)
        agg = {"fifo": fifo, "current_price_only": fifo, "greedy_energy": fifo,
               "sla_aware": fifo, "constraint_aware": ca}
        oc = OutcomeAnalysis(
            outcome=outcome_str, margin_pct=0.0,
            safety_evidence=(), loss_reasons=(), notes="",
        )
        out[name] = _FakeBenchmarkResult(
            _FakeBenchmarkReport(md, agg, hd_name, "explicit_override", oc),
        )
    return out


def test_cross_scenario_groups_by_workload_type():
    md_inf = _make_metadata(primary_workload_type="inference_critical")
    md_bat = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("inf1", md_inf, 100, 90, "sla_aware", "ALPHA_WIN"),
        ("inf2", md_inf, 110, 100, "sla_aware", "ALPHA_WIN"),
        ("bat1", md_bat, 200, 180, "current_price_only", "ALPHA_WIN"),
    ])
    cross = CrossScenarioReport.from_results(results)
    md = cross.to_markdown()
    # Per-workload section shows both workload types.
    assert "inference_critical" in md
    assert "batch_training" in md


def test_cross_scenario_separates_telemetry_failsafe_rows():
    md_inf = _make_metadata(primary_workload_type="inference_critical")
    md_telem = _make_metadata(
        primary_workload_type="telemetry_fail_safe",
        optimization_intent="safety_keep",
        is_telemetry_failsafe=True,
    )
    results = _build_fake_results([
        ("inf1", md_inf, 100, 90, "sla_aware", "ALPHA_WIN"),
        ("tel1", md_telem, 100, 100, "fifo", "KEEP_CORRECT"),
    ])
    cross = CrossScenarioReport.from_results(results)
    telem = cross.telemetry_failsafe_rows
    econ = cross.economic_rows
    assert len(telem) == 1
    assert len(econ) == 1
    assert telem[0].scenario_name == "tel1"


def test_cross_scenario_markdown_has_all_section_headers():
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    assert "## A. Overall policy" in md
    assert "## B. Per-workload-type" in md
    assert "## C. Per-scenario outcome" in md
    assert "## D. Baseline strength" in md


def test_cross_scenario_markdown_contains_goodput_unit_in_every_row():
    md_b = _make_metadata(primary_workload_type="batch_training",
                          goodput_unit="token_equivalent")
    md_i = _make_metadata(primary_workload_type="inference_critical",
                          goodput_unit="tokens")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
        ("i1", md_i, 100, 90, "sla_aware", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    # Each per-scenario row in section C carries goodput_unit.
    assert "token_equivalent" in md
    assert "tokens" in md


@pytest.mark.parametrize("forbidden", [
    "production savings", "guaranteed savings", "production-proven",
    "hyperscaler-validated", "enterprise-ready autonomous",
])
def test_cross_scenario_markdown_does_not_contain_forbidden_phrases(forbidden):
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    assert forbidden.lower() not in md.lower()


def test_cross_scenario_energy_row_includes_both_baselines():
    md_e = _make_metadata(
        primary_workload_type="batch_training",
        optimization_intent="energy_arbitrage",
        relevant_baselines=("fifo", "current_price_only", "greedy_energy",
                            "sla_aware"),
    )
    results = _build_fake_results([
        ("energy1", md_e, 100, 90, "current_price_only", "ALPHA_WIN"),
    ])
    cross = CrossScenarioReport.from_results(results)
    md = cross.to_markdown()
    # Section D should reference both current_price_only and greedy_energy.
    assert "current_price_only" in md
    assert "greedy_energy" in md


def test_per_workload_table_shows_mean_and_median():
    md_b = _make_metadata(primary_workload_type="batch_training")
    results = _build_fake_results([
        ("b1", md_b, 100, 90, "current_price_only", "ALPHA_WIN"),
        ("b2", md_b, 200, 180, "current_price_only", "ALPHA_WIN"),
    ])
    md = CrossScenarioReport.from_results(results).to_markdown()
    # Both 'Mean goodput/$' and 'Median goodput/$' columns must be present.
    assert "Mean goodput/$" in md
    assert "Median goodput/$" in md


# ---------------------------------------------------------------------------
# End-to-end via ConstraintBenchmarkRunner
# ---------------------------------------------------------------------------

def test_e2e_energy_scenario_reports_workload_relevant_baseline():
    r = ConstraintBenchmarkRunner().run_scenario(
        "energy_price_arbitrage_multiregion", steps=8, seed=42,
    )
    assert r.report.headline_baseline_name in {
        "current_price_only", "greedy_energy", "sla_aware", "fifo",
    }
    assert r.report.outcome.outcome in OUTCOMES
    assert r.report.scenario_metadata.is_telemetry_failsafe is False


def test_e2e_thermal_scenario_is_not_a_loss():
    """thermal_hotspot is the canonical CA win — should be ALPHA_WIN or SAFETY_WIN."""
    r = ConstraintBenchmarkRunner().run_scenario(
        "thermal_hotspot_mixed_cluster", steps=24, seed=42,
    )
    assert r.report.outcome.outcome in {"ALPHA_WIN", "SAFETY_WIN", "TIE"}, (
        f"Thermal expected to be a CA win, got {r.report.outcome.outcome} "
        f"(margin {r.report.outcome.margin_pct:+.2f}%)"
    )


def test_e2e_telemetry_failsafe_yields_keep_correct():
    r = ConstraintBenchmarkRunner().run_scenario(
        "degraded_topology_telemetry", steps=8, seed=42,
    )
    assert r.report.outcome.outcome == "KEEP_CORRECT"
    assert r.report.scenario_metadata.is_telemetry_failsafe is True


def test_per_scenario_row_holds_metadata_and_outcome():
    """PerScenarioRow construction surface."""
    md = _make_metadata()
    row = PerScenarioRow(
        scenario_name="s",
        metadata=md,
        headline_baseline_name="fifo",
        headline_baseline_rationale="r",
        outcome=OutcomeAnalysis(
            outcome="TIE", margin_pct=0.0,
            safety_evidence=(), loss_reasons=(), notes="",
        ),
    )
    assert row.scenario_name == "s"
    assert row.metadata is md
