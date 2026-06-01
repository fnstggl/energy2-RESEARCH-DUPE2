"""Phase-1 / Phase-2 tests: source coverage matrix integrity.

Verifies that data/external/economic_overlay/source_coverage_matrix.json
records the binding contract from `docs/ECONOMIC_OVERLAY_LAYER_V1.md`:

- every source has its raw fields + join keys + per-axis coverage labels
- every term-computability entry names the inputs that drive it
- scenario_prior tables are explicitly labelled, never measured
- no raw secrets / API keys are committed under data/external/economic_overlay
- raw downloads remain gitignored
- normalized samples are bounded
- no oracle / FIFO is referenced as a headline
- the existing constraint scorer module is NOT modified by this PR
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OVERLAY_DIR = REPO_ROOT / "data" / "external" / "economic_overlay"
SAMPLES_DIR = OVERLAY_DIR / "economic_overlay_samples"
DOC = REPO_ROOT / "docs" / "ECONOMIC_OVERLAY_LAYER_V1.md"
MODULE = REPO_ROOT / "aurelius" / "forecasting" / "economic_overlay.py"

# Generic secret patterns. We never store any of these in committed
# artefacts. Tokens like the literal `hf_...` are flagged regardless of value.
SECRET_PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?i:password|secret|api_key)\s*[:=]\s*['\"][^'\"]{6,}"),
    re.compile(r"\b(WATTTIME_PASSWORD|ERCOT_PASSWORD|PJM_API_KEY)\s*=\s*[A-Za-z0-9]"),
]


@pytest.fixture(scope="module")
def matrix() -> dict:
    path = OVERLAY_DIR / "source_coverage_matrix.json"
    assert path.exists(), f"missing source coverage matrix: {path}"
    return json.loads(path.read_text())


# ────────────────────── 1. Matrix top-level schema ──────────────────────


def test_matrix_top_level_keys(matrix):
    required = {"doc_version", "production_claim", "shadow_only",
                "sources", "term_computability_matrix",
                "scenario_overlays_defined"}
    missing = required - set(matrix.keys())
    assert not missing, f"matrix missing keys: {missing}"
    assert matrix["production_claim"] is False
    assert matrix["shadow_only"] is True


def test_matrix_lists_all_required_sources(matrix):
    """The 5 operational + 1 economic + 4 scenario sources must be present."""
    ids = {s["id"] for s in matrix["sources"]}
    expected_operational = {
        "asdwb/cara_latency_prediction",
        "Qinghao/AcmeTrace",
        "optimum-benchmark/llm-perf-leaderboard",
        "eth-easl/swissai-serving-trace",
        "ejhusom/llm-inference-energy-consumption",
    }
    assert expected_operational <= ids, (
        f"matrix missing operational sources: {expected_operational - ids}")
    assert "afhubbard/gpu-prices" in ids, "GPU price economic overlay missing"
    pjm_ids = [i for i in ids if "PJM" in i.upper()]
    assert pjm_ids, "PJM live overlay missing from matrix"
    scenario_ids = [i for i in ids if "SCENARIO" in i.upper()]
    assert len(scenario_ids) >= 2, (
        f"expected ≥2 scenario_prior entries (ERCOT/CAISO/WattTime); "
        f"got: {scenario_ids}")


def test_every_source_has_required_fields(matrix):
    required = {"id", "role", "raw_fields", "join_keys", "timestamp_coverage",
                "region_coverage", "gpu_type_coverage", "model_coverage",
                "price_coverage", "energy_coverage", "carbon_coverage",
                "field_quality", "limitations", "prohibited_uses"}
    for s in matrix["sources"]:
        missing = required - set(s.keys())
        assert not missing, f"source {s.get('id')} missing keys: {missing}"


def test_field_quality_labels_are_known(matrix):
    allowed = {"measured", "derived", "prior", "scenario_prior", "missing"}
    for s in matrix["sources"]:
        assert s["field_quality"] in allowed, (
            f"{s['id']} has bad field_quality: {s['field_quality']}")


def test_scenario_sources_are_labelled_scenario_prior(matrix):
    """ERCOT / CAISO / WattTime entries must be field_quality=scenario_prior
    (this PR does not call live ERCOT / CAISO / WattTime APIs)."""
    for s in matrix["sources"]:
        if "SCENARIO" in s["id"].upper():
            assert s["field_quality"] == "scenario_prior", (
                f"{s['id']} should be scenario_prior, "
                f"got {s['field_quality']}")


def test_pjm_is_measured_when_live(matrix):
    """PJM Data Miner entry must be labelled measured (live fetch)."""
    pjm = next((s for s in matrix["sources"] if "PJM" in s["id"]), None)
    assert pjm is not None
    assert pjm["field_quality"] == "measured"


def test_afhubbard_is_measured_but_recorded_as_public_list_price(matrix):
    """GPU pricing dataset is measured (real listings) but its limitation
    section MUST note PUBLIC LIST PRICE != operator invoice."""
    af = next((s for s in matrix["sources"]
               if s["id"] == "afhubbard/gpu-prices"), None)
    assert af is not None
    assert af["field_quality"] == "measured"
    text = " ".join(af["limitations"]).lower()
    assert "list price" in text and "operator" in text, (
        "afhubbard limitations must call out the list-price vs operator-rate "
        "distinction explicitly")


# ────────────────────── 2. Term computability matrix ──────────────────────


def test_term_computability_covers_required_terms(matrix):
    required = {
        "estimated_gpu_cost_usd", "estimated_energy_cost_usd",
        "estimated_carbon_kg", "estimated_carbon_cost_usd",
        "estimated_cache_value_usd", "estimated_cold_start_cost_usd",
        "estimated_migration_cost_usd", "estimated_prefill_cost_usd",
        "estimated_decode_cost_usd", "estimated_memory_pressure_cost_usd",
        "estimated_sla_safe_goodput", "estimated_sla_safe_goodput_per_dollar",
    }
    actual = set(matrix["term_computability_matrix"].keys())
    missing = required - actual
    assert not missing, f"term computability missing: {missing}"


def test_carbon_cost_marked_operator_policy_only(matrix):
    e = matrix["term_computability_matrix"]["estimated_carbon_cost_usd"]
    assert "operator" in e["computable_when"].lower(), (
        "carbon cost must explicitly require operator carbon price")


def test_memory_pressure_marked_operator_policy_only(matrix):
    e = matrix["term_computability_matrix"][
        "estimated_memory_pressure_cost_usd"]
    assert "operator" in e["computable_when"].lower()


def test_every_term_records_inputs_and_sources(matrix):
    for term, e in matrix["term_computability_matrix"].items():
        assert e.get("inputs"), f"{term} has no inputs"
        assert e.get("sources"), f"{term} has no sources"
        assert e.get("computable_when"), f"{term} has no computable_when"


# ────────────────────── 3. Scenario overlays ──────────────────────


def test_scenario_overlays_match_module(matrix):
    """The list in the matrix must match the SCENARIO_OVERLAYS dict
    defined in `aurelius.forecasting.economic_overlay`."""
    from aurelius.forecasting.economic_overlay import SCENARIO_OVERLAYS
    expected = set(SCENARIO_OVERLAYS.keys())
    actual = set(matrix["scenario_overlays_defined"])
    assert expected == actual, (
        f"scenario overlays divergence: only-in-matrix="
        f"{actual - expected}, only-in-module={expected - actual}")


def test_required_scenarios_present(matrix):
    """Mission §5 names six scenario overlays. All must be present in the
    module (the matrix just mirrors it)."""
    from aurelius.forecasting.economic_overlay import SCENARIO_OVERLAYS
    required = {
        "pjm_energy_overlay", "ercot_energy_overlay",
        "caiso_energy_overlay", "watttime_carbon_overlay",
        "no_operator_policy_overlay",
    }
    missing = required - set(SCENARIO_OVERLAYS.keys())
    assert not missing, f"required scenarios missing: {missing}"


# ────────────────────── 4. Secret / raw-data safety ──────────────────────


def test_no_secrets_in_committed_overlay_dir():
    leaks = []
    for p in OVERLAY_DIR.rglob("*"):
        if not p.is_file():
            continue
        try:
            body = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(body):
                leaks.append((str(p.relative_to(REPO_ROOT)), pat.pattern))
                break
    assert not leaks, f"secret leak in overlay dir: {leaks}"


def test_no_secrets_in_module_or_scripts():
    files = [
        MODULE,
        REPO_ROOT / "scripts" / "build_economic_overlay_v1.py",
        REPO_ROOT / "scripts" / "run_economic_overlay_eval_v1.py",
        DOC,
    ]
    leaks = []
    for p in files:
        if not p.exists():
            continue
        body = p.read_text()
        for pat in SECRET_PATTERNS:
            if pat.search(body):
                leaks.append((str(p.relative_to(REPO_ROOT)), pat.pattern))
    assert not leaks, f"secret leak in code/doc files: {leaks}"


def test_no_raw_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf"], cwd=REPO_ROOT,
    ).decode().splitlines()
    raw = [p for p in out if "/raw/" in p]
    assert not raw, f"raw downloads committed (gitignore broken): {raw}"


def test_normalized_samples_bounded():
    """Each committed normalized sample under data/external/economic_overlay/
    must be ≤100 MB; total ≤300 MB."""
    total = 0
    for p in SAMPLES_DIR.rglob("*"):
        if not p.is_file():
            continue
        sz = p.stat().st_size
        assert sz <= 100 * 1024 * 1024, (
            f"{p.relative_to(REPO_ROOT)} = {sz}B > 100 MB cap")
        total += sz
    assert total <= 300 * 1024 * 1024, (
        f"total committed overlay samples = {total}B > 300 MB cap")


# ────────────────────── 5. No oracle / FIFO headline ──────────────────────


def test_doc_does_not_claim_oracle_or_fifo_as_headline():
    body = DOC.read_text().lower()
    for forbidden in ("oracle as headline", "fifo as headline",
                      "production savings claim",
                      "production truth"):
        assert forbidden not in body, (
            f"doc uses forbidden phrase: {forbidden!r}")


# ────────────────────── 6. Production scorer untouched ──────────────────────


def test_existing_constraint_scorer_untouched():
    """Verify this PR did not edit production scheduler / shadow scorer
    files. The economic overlay is additive only."""
    out = subprocess.check_output(
        ["git", "diff", "--name-only", "main...HEAD"], cwd=REPO_ROOT,
    ).decode().splitlines()
    forbidden_paths = (
        "aurelius/forecasting/constraint_shadow_scorer.py",
        "aurelius/forecasting/constraint_scorer_features.py",
        "aurelius/optimization/scheduler.py",
        "aurelius/optimization/objective.py",
        "aurelius/optimization/constraints.py",
        "aurelius/residency/decision.py",
        "aurelius/residency/sim.py",
        "aurelius/residency/shadow.py",
        "aurelius/residency/backtest.py",
        "aurelius/frontier/controller.py",
    )
    diffs = [p for p in out if p in forbidden_paths]
    assert not diffs, (
        "this PR must not modify production scheduler / scorer / residency "
        f"files: {diffs}")
