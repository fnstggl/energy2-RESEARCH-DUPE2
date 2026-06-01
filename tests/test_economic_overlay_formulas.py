"""Phase-3 / Phase-4 tests: formula correctness + no-invented-constants.

Asserts each derived economic term:
  - has the expected algebraic form (within float tolerance),
  - is missing whenever its required inputs are missing,
  - records the formula string in `formula_by_field`,
  - records `value_quality` in the documented vocabulary,
  - carbon_cost requires operator-supplied carbon price and is never
    computed from public data alone.

Also covers the SLA-safe goodput/$ aggregator: carbon cost is NOT in the
cost denominator under the default empty OperatorPricingPolicy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting.economic_overlay import (  # noqa: E402
    VALUE_QUALITY_LABELS,
    OperatorPricingPolicy,
    OverlayBuilder,
    OverlayBuilderConfig,
)


@pytest.fixture
def builder() -> OverlayBuilder:
    return OverlayBuilder(OverlayBuilderConfig(
        gpu_price_path=REPO_ROOT / "data" / "external" / "economic_overlay"
                       / "economic_overlay_samples"
                       / "gpu_price_overlay_2026-06-01.jsonl",
        pjm_path=REPO_ROOT / "data" / "external" / "economic_overlay"
                 / "economic_overlay_samples"
                 / "pjm_da_energy_price_7day.jsonl",
    ))


def _op(**kw):
    base = {"source_trace_id": "t", "source_dataset_id": "ds",
            "gpu_type": "A100", "gpu_count": 1, "model_id": "m"}
    base.update(kw)
    return base


# ────────────────────── 1. GPU cost formula ──────────────────────


def test_gpu_cost_formula(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    output_tokens=100))
    p = rec.gpu_price_usd_per_hour
    assert p is not None
    expected = p * 1 * 10.0 / 3600.0
    assert abs(rec.estimated_gpu_cost_usd - expected) < 1e-9
    assert "gpu_price_usd_per_hour * gpu_count" in \
        rec.formula_by_field["estimated_gpu_cost_usd"]


def test_gpu_cost_missing_when_e2e_missing(builder):
    rec = builder.build_record(_op(sla_s=20.0))
    assert rec.estimated_gpu_cost_usd is None
    assert rec.value_quality_by_field.get(
        "estimated_gpu_cost_usd", "missing") == "missing"


def test_gpu_cost_missing_when_gpu_type_unknown(builder):
    """Unknown GPU type → no public listing → no fuzzy match either: the
    overlay should fall back to nearest family or report missing. Either
    outcome is fine, but a fuzzy match must be labelled as such."""
    rec = builder.build_record(_op(gpu_type="EXOTIC-XYZ",
                                    e2e_latency_s=5.0, sla_s=10.0))
    vq = rec.value_quality_by_field.get("gpu_price_usd_per_hour")
    assert vq in {"missing", "prior_fuzzy_match"}, vq
    if vq == "prior_fuzzy_match":
        assert "nearest" in rec.formula_by_field["gpu_price_usd_per_hour"]
    else:
        assert rec.estimated_gpu_cost_usd is None


# ────────────────────── 2. Energy cost formula ──────────────────────


def test_energy_cost_formula(builder):
    rec = builder.build_record(_op(e2e_latency_s=20.0, sla_s=60.0,
                                    energy_kwh=0.005))
    if rec.electricity_price_usd_per_kwh is None:
        pytest.skip("PJM overlay not loaded")
    expected = 0.005 * rec.electricity_price_usd_per_kwh
    assert abs(rec.estimated_energy_cost_usd - expected) < 1e-12


def test_energy_kwh_derived_from_power_and_duration(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    gpu_power_w=300.0))
    # 300 W * 10 s / 3,600,000 = 0.0008333... kWh
    assert rec.energy_kwh is not None
    assert abs(rec.energy_kwh - 300.0 * 10.0 / 3_600_000.0) < 1e-12
    assert rec.value_quality_by_field["energy_kwh"] == "derived_from_power_prior"


def test_energy_kwh_missing_when_no_power_no_kwh(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0))
    assert rec.energy_kwh is None


# ────────────────────── 3. Carbon kg + carbon cost ──────────────────────


def test_carbon_kg_derived_with_scenario_intensity(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    energy_kwh=0.001))
    # default carbon overlay = watttime scenario midpoint (410 g/kWh)
    expected_kg = 0.001 * 410.0 / 1000.0
    assert abs(rec.estimated_carbon_kg - expected_kg) < 1e-12
    assert rec.value_quality_by_field[
        "estimated_carbon_kg"] == "scenario_prior"


def test_carbon_cost_missing_without_operator_carbon_price(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    energy_kwh=0.001))
    assert rec.estimated_carbon_cost_usd is None
    assert rec.value_quality_by_field[
        "estimated_carbon_cost_usd"] == "missing"


def test_carbon_cost_computed_with_operator_carbon_price():
    cfg = OverlayBuilderConfig(
        gpu_price_path=REPO_ROOT / "data" / "external" / "economic_overlay"
                       / "economic_overlay_samples"
                       / "gpu_price_overlay_2026-06-01.jsonl",
        pjm_path=REPO_ROOT / "data" / "external" / "economic_overlay"
                 / "economic_overlay_samples"
                 / "pjm_da_energy_price_7day.jsonl",
        operator_policy=OperatorPricingPolicy(
            carbon_price_per_kg_usd=0.04,
        ),
    )
    builder = OverlayBuilder(cfg)
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                   energy_kwh=0.001))
    assert rec.estimated_carbon_cost_usd is not None
    expected = rec.estimated_carbon_kg * 0.04
    assert abs(rec.estimated_carbon_cost_usd - expected) < 1e-12
    assert rec.value_quality_by_field["estimated_carbon_cost_usd"] == "derived"


# ────────────────────── 4. Cache value formula ──────────────────────


def test_cache_value_formula(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    ttft_s=0.40, tpot_s=0.020,
                                    output_tokens=100,
                                    cache_reuse_pct=0.30))
    p = rec.gpu_price_usd_per_hour
    assert p is not None
    expected = 0.30 * 0.40 * (p / 3600.0) * 1
    assert abs(rec.estimated_cache_value_usd - expected) < 1e-12


def test_cache_value_missing_without_reuse(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    ttft_s=0.40))
    assert rec.estimated_cache_value_usd is None
    assert rec.value_quality_by_field["estimated_cache_value_usd"] == "missing"


# ────────────────────── 5. Migration / cold-start ──────────────────────


def test_migration_cost_missing_without_cache_loss_proxy(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    ttft_s=0.40))
    assert rec.estimated_migration_cost_usd is None
    assert rec.value_quality_by_field["estimated_migration_cost_usd"] == \
        "missing"


def test_migration_cost_derived_when_cache_loss_present(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    ttft_s=0.40, cache_loss_pct=0.50))
    p = rec.gpu_price_usd_per_hour
    expected = 0.50 * 0.40 * (p / 3600.0)
    assert abs(rec.estimated_migration_cost_usd - expected) < 1e-12


def test_cold_start_cost_derived_when_load_duration_present(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    model_load_duration_s=2.5,
                                    model_load_source="measured"))
    p = rec.gpu_price_usd_per_hour
    expected = 2.5 * (p / 3600.0)
    assert abs(rec.estimated_cold_start_cost_usd - expected) < 1e-12
    assert rec.value_quality_by_field["estimated_cold_start_cost_usd"] \
        == "derived"


def test_cold_start_proxy_labelled_when_source_unknown(builder):
    rec = builder.build_record(_op(e2e_latency_s=10.0, sla_s=20.0,
                                    model_load_duration_s=2.5))
    assert rec.value_quality_by_field["estimated_cold_start_cost_usd"] \
        == "proxy"


# ────────────────────── 6. SLA-safe goodput / $ ──────────────────────


def test_sla_met_binary(builder):
    rec_meets = builder.build_record(_op(e2e_latency_s=5.0, sla_s=10.0,
                                         output_tokens=50))
    rec_misses = builder.build_record(_op(e2e_latency_s=20.0, sla_s=10.0,
                                          output_tokens=50))
    assert rec_meets.sla_met is True
    assert rec_misses.sla_met is False
    assert rec_meets.sla_safe_goodput == 50.0
    assert rec_misses.sla_safe_goodput == 0.0


def test_goodput_per_dollar_excludes_carbon_under_default_policy(builder):
    rec = builder.build_record(_op(e2e_latency_s=5.0, sla_s=10.0,
                                   output_tokens=50, energy_kwh=0.001))
    assert rec.sla_safe_goodput_per_dollar is not None
    # carbon_cost is missing under default OperatorPricingPolicy, so the
    # formula in fm must not reference carbon_cost.
    fm = rec.formula_by_field["sla_safe_goodput_per_dollar"]
    assert "carbon" not in fm.lower()


# ────────────────────── 7. value_quality vocabulary ──────────────────────


def test_value_quality_labels_are_known(builder):
    rec = builder.build_record(_op(e2e_latency_s=5.0, sla_s=10.0,
                                   output_tokens=50, energy_kwh=0.001,
                                   ttft_s=0.4, tpot_s=0.02,
                                   cache_reuse_pct=0.3))
    extra_allowed = {
        "derived_from_power_prior",
        "measured_input_derived_formula",
        "prior_fuzzy_match",
    }
    for k, v in rec.value_quality_by_field.items():
        assert v in VALUE_QUALITY_LABELS or v in extra_allowed, (
            f"field {k} has unknown value_quality {v!r}")


def test_every_derived_field_has_a_formula(builder):
    rec = builder.build_record(_op(e2e_latency_s=5.0, sla_s=10.0,
                                   output_tokens=50, energy_kwh=0.001,
                                   ttft_s=0.4, tpot_s=0.02,
                                   cache_reuse_pct=0.3))
    # for every estimated_* field with vq in {derived, scenario_prior,
    # measured_input_derived_formula}, there must be a formula string.
    for k, v in rec.value_quality_by_field.items():
        if k.startswith("estimated_") and v in (
                "derived", "scenario_prior",
                "measured_input_derived_formula"):
            assert rec.formula_by_field.get(k), (
                f"field {k} (vq={v}) has no formula string")


# ────────────────────── 8. Overlay classification ──────────────────────


def test_overlay_class_when_gpu_prior_and_pjm_measured(builder):
    rec = builder.build_record(_op(e2e_latency_s=5.0, sla_s=10.0,
                                   output_tokens=50, energy_kwh=0.001))
    # GPU price = prior, energy price = measured PJM, energy_kwh = measured
    # → cross_dataset_joined (mixed input quality, no scenario)
    assert rec.overlay_class == "cross_dataset_joined"


def test_overlay_class_scenario_when_no_pjm():
    cfg = OverlayBuilderConfig(
        gpu_price_path=REPO_ROOT / "data" / "external" / "economic_overlay"
                       / "economic_overlay_samples"
                       / "gpu_price_overlay_2026-06-01.jsonl",
        pjm_path=None,
        use_live_pjm=False,
        energy_market="ercot_energy_overlay",
    )
    b = OverlayBuilder(cfg)
    rec = b.build_record(_op(e2e_latency_s=5.0, sla_s=10.0,
                              output_tokens=50, energy_kwh=0.001))
    assert rec.overlay_class == "scenario_prior"


def test_no_invented_constants_in_module():
    """Static check: the economic_overlay.py source must NOT contain
    hardcoded literals for GPU $/hr / electricity $/kWh / carbon $/kg."""
    body = (REPO_ROOT / "aurelius" / "forecasting"
                       / "economic_overlay.py").read_text()
    # We DO want the scenario_prior midpoints in SCENARIO_OVERLAYS — those
    # are explicit scenarios. The forbidden pattern is any other place that
    # uses a $/hr or $/kWh constant.
    import re
    forbidden_terms = [
        r"GPU_HOUR_PRICE\s*=", r"DEFAULT_GPU_HOUR\s*=",
        r"CACHE_VALUE_WEIGHT\s*=", r"CACHE_WEIGHT\s*=",
        r"MIGRATION_PENALTY\s*=", r"UTILITY_SCORE\s*=",
        r"COMPOSITE_WEIGHT\s*=",
    ]
    for pat in forbidden_terms:
        assert not re.search(pat, body), (
            f"economic_overlay.py contains forbidden constant pattern: {pat}")
