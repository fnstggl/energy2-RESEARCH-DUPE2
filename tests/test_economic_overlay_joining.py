"""Phase-4 / Phase-5 tests: join logic + scenario overlay safety.

Exercises:
  - GPU-price overlay (afhubbard) joins by gpu_type / region / provider /
    is_spot, with `prior_fuzzy_match` fallback when family is unknown;
  - PJM energy overlay measured-class lookup vs scenario_prior fallback;
  - WattTime/ERCOT/CAISO scenario_prior labelling — never measured;
  - operator-policy precedence over public overlays;
  - 3-class classification never mixes scenario_prior into a
    cross_dataset_joined record's headline.
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
    CarbonOverlay,
    EnergyPriceOverlay,
    GPUPriceOverlay,
    OperatorPricingPolicy,
    OverlayBuilder,
    OverlayBuilderConfig,
)

OVERLAY_DIR = REPO_ROOT / "data" / "external" / "economic_overlay"
SAMPLES_DIR = OVERLAY_DIR / "economic_overlay_samples"


@pytest.fixture(scope="module")
def gpu_overlay() -> GPUPriceOverlay:
    return GPUPriceOverlay.load(
        SAMPLES_DIR / "gpu_price_overlay_2026-06-01.jsonl")


# ────────────────────── 1. GPU price overlay joins ──────────────────────


def test_gpu_overlay_loads_from_committed_sample(gpu_overlay):
    assert len(gpu_overlay.rows) > 100
    assert gpu_overlay.source_dataset_id == "afhubbard/gpu-prices"
    assert gpu_overlay.source_license == "cc-by-4.0"


def test_gpu_overlay_exact_family_lookup(gpu_overlay):
    res = gpu_overlay.lookup(gpu_type="A100")
    assert res["price_per_gpu_hour_usd"] is not None
    assert res["value_quality"] in {"prior", "prior_fuzzy_match"}
    assert "afhubbard_gpu_prices" in res["formula"]


def test_gpu_overlay_fuzzy_match_when_unknown_family(gpu_overlay):
    res = gpu_overlay.lookup(gpu_type="some-exotic-XPU")
    # Either nearest-family fuzzy or missing — both honest.
    assert res["value_quality"] in {"prior_fuzzy_match", "missing"}
    if res["value_quality"] == "prior_fuzzy_match":
        assert res["match_kind"] == "nearest_family"


def test_gpu_overlay_missing_when_no_gpu_type(gpu_overlay):
    res = gpu_overlay.lookup(gpu_type=None)
    assert res["price_per_gpu_hour_usd"] is None
    assert res["value_quality"] == "missing"


# ────────────────────── 2. Operator policy precedence ──────────────────────


def test_operator_policy_overrides_public_gpu_overlay():
    cfg = OverlayBuilderConfig(
        gpu_price_path=SAMPLES_DIR / "gpu_price_overlay_2026-06-01.jsonl",
        operator_policy=OperatorPricingPolicy(
            gpu_hour_price_per_type={"a100": 0.99},
        ),
    )
    b = OverlayBuilder(cfg)
    rec = b.build_record({
        "source_trace_id": "t", "source_dataset_id": "ds",
        "gpu_type": "A100", "gpu_count": 1,
        "e2e_latency_s": 3.0, "sla_s": 10.0, "output_tokens": 50,
    })
    assert rec.gpu_price_usd_per_hour == 0.99
    assert rec.value_quality_by_field["gpu_price_usd_per_hour"] \
        == "measured"
    assert rec.formula_by_field["gpu_price_usd_per_hour"] \
        == "operator_policy.gpu_hour_price_per_type"


def test_operator_policy_overrides_energy_overlay():
    cfg = OverlayBuilderConfig(
        operator_policy=OperatorPricingPolicy(
            energy_price_per_kwh_usd=0.085,
        ),
    )
    b = OverlayBuilder(cfg)
    rec = b.build_record({
        "source_trace_id": "t", "source_dataset_id": "ds",
        "gpu_type": "A100", "gpu_count": 1,
        "e2e_latency_s": 3.0, "sla_s": 10.0, "output_tokens": 50,
        "energy_kwh": 0.001,
    })
    assert rec.electricity_price_usd_per_kwh == 0.085
    assert rec.value_quality_by_field["electricity_price_usd_per_kwh"] \
        == "measured"


# ────────────────────── 3. PJM measured vs scenario ──────────────────────


def test_pjm_measured_overlay_loads_live_rows():
    path = SAMPLES_DIR / "pjm_da_energy_price_7day.jsonl"
    if not path.exists():
        pytest.skip("PJM live fetch was not run")
    e = EnergyPriceOverlay.load_pjm(path)
    assert len(e.rows) >= 24, "PJM 7-day fetch should yield ≥24h rows"
    assert e.value_quality == "measured"
    r = e.lookup()
    assert r["value_quality"] == "measured"
    # PJM LMPs are $/MWh; /1000 gives $/kWh in 0.001 .. 0.50 range.
    assert 0.0 < r["price_per_kwh_usd"] < 0.5


def test_scenario_energy_overlays_have_explicit_scenario_label():
    for k in ("pjm_energy_overlay", "ercot_energy_overlay",
              "caiso_energy_overlay"):
        e = EnergyPriceOverlay.scenario(k)
        assert e.value_quality == "scenario_prior", (
            f"scenario {k} not labelled scenario_prior")
        r = e.lookup()
        assert r["value_quality"] == "scenario_prior"
        assert "scenario" in r["formula"]


def test_no_operator_policy_overlay_reports_missing():
    e = EnergyPriceOverlay.scenario("no_operator_policy_overlay")
    assert e.value_quality == "missing"
    r = e.lookup()
    assert r["value_quality"] == "missing"
    assert r["price_per_kwh_usd"] is None


# ────────────────────── 4. PJM/ERCOT/CAISO are NOT operator contract ─────


def test_pjm_measurement_not_treated_as_operator_contract():
    """The PJM overlay's value_quality must be `measured` (live market
    LMP), NEVER `operator_supplied`. This is the core test for the
    mission requirement: market price ≠ utility contract."""
    path = SAMPLES_DIR / "pjm_da_energy_price_7day.jsonl"
    if not path.exists():
        pytest.skip("PJM live fetch was not run")
    cfg = OverlayBuilderConfig(pjm_path=path)
    b = OverlayBuilder(cfg)
    rec = b.build_record({
        "source_trace_id": "t", "source_dataset_id": "ds",
        "gpu_type": "A100", "gpu_count": 1,
        "e2e_latency_s": 3.0, "sla_s": 10.0, "output_tokens": 50,
        "energy_kwh": 0.001,
    })
    # `measured` is correct (live LMP), `operator_supplied` would be wrong.
    assert rec.value_quality_by_field["electricity_price_usd_per_kwh"] \
        == "measured"
    assert "pjm" in rec.formula_by_field[
        "electricity_price_usd_per_kwh"].lower()


# ────────────────────── 5. Carbon overlay ──────────────────────


def test_watttime_carbon_is_physical_intensity_not_price():
    c = CarbonOverlay.scenario("watttime_carbon_overlay")
    r = c.lookup()
    # carbon_intensity_g_per_kwh is a physical quantity (g CO2 / kWh).
    assert r["carbon_intensity_g_per_kwh"] is not None
    assert 100 <= r["carbon_intensity_g_per_kwh"] <= 1000, (
        "carbon intensity midpoint should be in realistic 100-1000 g/kWh "
        f"range, got {r['carbon_intensity_g_per_kwh']}")
    assert r["value_quality"] == "scenario_prior"


def test_carbon_cost_requires_operator_carbon_price():
    """Carbon cost MUST require an operator-supplied carbon price; public
    data alone (intensity only) must never produce a $-denominated carbon
    cost."""
    cfg = OverlayBuilderConfig(
        gpu_price_path=SAMPLES_DIR / "gpu_price_overlay_2026-06-01.jsonl",
        pjm_path=SAMPLES_DIR / "pjm_da_energy_price_7day.jsonl",
        operator_policy=OperatorPricingPolicy(),  # no carbon price
    )
    b = OverlayBuilder(cfg)
    rec = b.build_record({
        "source_trace_id": "t", "source_dataset_id": "ds",
        "gpu_type": "A100", "gpu_count": 1,
        "e2e_latency_s": 5.0, "sla_s": 10.0, "output_tokens": 50,
        "energy_kwh": 0.001,
    })
    assert rec.estimated_carbon_kg is not None  # physical quantity present
    assert rec.estimated_carbon_cost_usd is None  # $-cost absent
    assert rec.value_quality_by_field[
        "estimated_carbon_cost_usd"] == "missing"


# ────────────────────── 6. Result-class purity ──────────────────────


def test_measured_same_record_only_when_both_inputs_measured():
    """measured_same_record requires both gpu_price AND energy_kwh
    measured. Otherwise the classification must drop a tier."""
    cfg = OverlayBuilderConfig(
        gpu_price_path=SAMPLES_DIR / "gpu_price_overlay_2026-06-01.jsonl",
        pjm_path=SAMPLES_DIR / "pjm_da_energy_price_7day.jsonl",
        operator_policy=OperatorPricingPolicy(
            gpu_hour_price_per_type={"a100": 1.50},
            energy_price_per_kwh_usd=0.08,
        ),
    )
    b = OverlayBuilder(cfg)
    # measured energy_kwh + operator-supplied gpu price + operator energy →
    # measured_same_record.
    rec = b.build_record({
        "source_trace_id": "t", "source_dataset_id": "ds",
        "gpu_type": "A100", "gpu_count": 1,
        "e2e_latency_s": 5.0, "sla_s": 10.0, "output_tokens": 50,
        "energy_kwh": 0.001,
    })
    assert rec.overlay_class == "measured_same_record"


def test_scenario_prior_class_when_only_scenario_inputs():
    cfg = OverlayBuilderConfig(
        gpu_price_path=None,
        pjm_path=None,
        use_live_pjm=False,
        energy_market="caiso_energy_overlay",
    )
    b = OverlayBuilder(cfg)
    rec = b.build_record({
        "source_trace_id": "t", "source_dataset_id": "ds",
        "gpu_type": "A100", "gpu_count": 1,
        "e2e_latency_s": 5.0, "sla_s": 10.0, "output_tokens": 50,
        "energy_kwh": 0.001,
    })
    assert rec.overlay_class == "scenario_prior"
