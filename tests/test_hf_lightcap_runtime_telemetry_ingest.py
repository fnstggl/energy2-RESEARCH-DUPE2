"""Tests for the Lightcap/agent-runtime-telemetry-small bounded ingest.

Covers:
- No raw / analysis_sample data committed.
- Per-config schema_profile + schema_mapping + summary + rollups present.
- Schema mapping classifies every accepted column, no unknown columns.
- Summary passes every promotion gate.
- ``operations`` config is promoted to backtest + constraint_aware_eval +
  training_priors (moderate strength, 2,262 rows).
- ``tool_summary`` config is promoted_for_schema_only (32 rows,
  fixture-only).
- ``tool_runtime_trace`` canonical type + ``ToolRuntimeRecord`` validated.
- Trust tier is ``tier_3_cluster_scheduler_traces`` (NOT Tier 1, NOT Tier 2).
- License is ``cc-by-4.0`` and gated=False.
- Signal coverage: routing / failure_timeout / cache-residency-proxy
  present; GPU / queue / replica / model / TTFT / TPOT absent.
- Limitations pin the closed-runtime-timing + no-LLM-serving caveats.
- Per-config normalized samples are committed under the 100-MiB cap.
- Per-config fixtures are committed and tiny.
- Canonical corpus registry knows about both configs.
- Candidates JSON records the focused_audit_2026_06_01c block.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces.hf_corpus import (
    promotion,  # noqa: E402
    schemas,  # noqa: E402
)

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_DATASET = DATASET_ID.replace("/", "__")
CONFIGS = ["operations", "tool_summary"]
PRIMARY_CONFIG = "operations"


def _processed_dir(config: str) -> Path:
    return HF_DIR / SAFE_DATASET / config / "processed"


def _fixture_path(config: str) -> Path:
    return FIXTURES_DIR / f"{SAFE_DATASET}__{config}_sample.jsonl"


def _summary(config: str) -> dict:
    with open(_processed_dir(config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw / analysis data committed ─────────────


def test_no_raw_files_tracked_by_git() -> None:
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_DATASET}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw_committed == [], (
        f"raw downloads committed (gitignore broken): {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed (gitignore broken): "
        f"{analysis_committed}"
    )


# ───────────────────────── 2. Fixture files ────────────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_fixture_files_are_committed_and_tiny(config: str) -> None:
    fixture = _fixture_path(config)
    assert fixture.exists(), f"missing fixture: {fixture}"
    size = fixture.stat().st_size
    assert 0 < size <= 16 * 1024, (
        f"fixture {fixture} size {size} outside 1-16KiB band"
    )
    # Deterministic JSONL: every line must parse as JSON.
    with open(fixture) as fh:
        lines = fh.readlines()
    assert len(lines) >= 1
    for ln in lines:
        json.loads(ln)


# ───────────────────────── 3. Processed artefacts ──────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_profile_present_and_well_formed(config: str) -> None:
    p = _processed_dir(config) / "schema_profile.json"
    assert p.exists(), f"missing schema_profile: {p}"
    with open(p) as fh:
        profile = json.load(fh)
    assert profile["dataset_id"] == DATASET_ID
    assert profile["config_name"] == config
    assert profile["inspected_row_count"] > 0
    assert isinstance(profile["raw_columns"], list)
    assert isinstance(profile["normalized_columns"], list)
    assert profile["raw_columns"]
    assert profile["normalized_columns"]


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_mapping_classifies_every_accepted_column(config: str) -> None:
    p = _processed_dir(config) / "schema_mapping.json"
    assert p.exists(), f"missing schema_mapping: {p}"
    with open(p) as fh:
        mapping = json.load(fh)
    assert mapping["dataset_id"] == DATASET_ID
    assert mapping["config_name"] == config
    assert mapping["accepted_columns"], (
        f"no accepted columns recorded for {config}"
    )
    assert mapping["rejected_columns"] == [], (
        f"{config} has rejected columns; tighten the mapping table: "
        f"{mapping['rejected_columns']}"
    )
    # Every column record must have non-null normalized_field, quality,
    # and aurelius_signal_category for accepted columns.
    accepted_set = set(mapping["accepted_columns"])
    for col in mapping["columns"]:
        if col["raw_column_name"] not in accepted_set:
            continue
        assert col["normalized_field"], (
            f"{config}: accepted column {col['raw_column_name']} has no "
            f"normalized_field"
        )
        assert col["field_quality"] in schemas.FIELD_QUALITY_VALUES, (
            f"{config}: bad field_quality {col['field_quality']} for "
            f"{col['raw_column_name']}"
        )
        assert col["aurelius_signal_category"], (
            f"{config}: accepted column {col['raw_column_name']} has no "
            f"aurelius_signal_category"
        )


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_present(config: str) -> None:
    s = _summary(config)
    assert s["dataset_id"] == DATASET_ID
    assert s["config_name"] == config
    assert s["canonical_trace_type"] == "tool_runtime_trace"
    assert s["license"] == "cc-by-4.0"
    assert s["gated"] is False
    assert s["source_url"] == f"https://huggingface.co/datasets/{DATASET_ID}"


@pytest.mark.parametrize("config", CONFIGS)
def test_statistical_rollups_present(config: str) -> None:
    p = _processed_dir(config) / "statistical_rollups.json"
    assert p.exists(), f"missing statistical_rollups: {p}"
    with open(p) as fh:
        rollups = json.load(fh)
    assert "subgroup_counts" in rollups
    if config == "operations":
        # Operations rollups must include duration_ms distribution +
        # per-tool failure rates.
        assert "numeric_distributions" in rollups
        assert "duration_ms" in rollups["numeric_distributions"]
        assert "per_tool_failure_rates" in rollups
        assert "overall_failure_rates" in rollups
        overall = rollups["overall_failure_rates"]
        # Sanity: error rate is in (0, 1) and matches subgroup counts.
        assert 0 < overall["error_rate"] < 0.5
        assert overall["count"] >= 1000


# ───────────────────────── 4. Promotion gates ─────────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_passes_every_promotion_gate(config: str) -> None:
    s = _summary(config)
    gates = promotion.gates(s)
    failed = [g for g in gates if not g["passed"]]
    assert failed == [], (
        f"{config}: {len(failed)} promotion gate(s) failed: {failed}"
    )


def test_operations_config_is_promoted_for_backtest_and_more() -> None:
    s = _summary("operations")
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_backtest"
    tags = set(decision["promotion_tags"])
    assert {
        "promoted_for_backtest",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_training_priors",
    }.issubset(tags), (
        f"operations missing expected promotion tags; got {sorted(tags)}"
    )


def test_tool_summary_config_is_promoted_for_schema_only() -> None:
    s = _summary("tool_summary")
    decision = promotion.evaluate_promotion(s)
    # 32 aggregated rows → fixture_only strength → schema-only promotion.
    assert decision["state"] == "promoted_for_schema_only", (
        f"unexpected tool_summary state: {decision['state']}"
    )


# ───────────────────────── 5. Canonical type + trust tier ─────────────────


def test_tool_runtime_trace_is_a_canonical_type() -> None:
    assert "tool_runtime_trace" in schemas.CANONICAL_TRACE_TYPES


def test_tool_runtime_record_class_is_registered() -> None:
    cls = schemas.TRACE_TYPE_TO_RECORD_CLASS["tool_runtime_trace"]
    assert cls is schemas.ToolRuntimeRecord
    # And its payload fields are non-empty and registered.
    fields = schemas.TRACE_TYPE_TO_PAYLOAD_FIELDS["tool_runtime_trace"]
    assert "operation_id" in fields
    assert "tool_name" in fields
    assert "duration_ms" in fields
    assert "status" in fields
    assert "error_type" in fields


def test_tool_runtime_record_validates_field_quality() -> None:
    # Smoke-test the dataclass validator: it must reject unknown fields
    # in field_quality and bad trace_type.
    with pytest.raises(schemas.HFCorpusSchemaError):
        schemas.ToolRuntimeRecord(
            source_dataset_id=DATASET_ID,
            trace_type="tool_runtime_trace",
            provenance="p",
            field_quality={"NOT_A_FIELD": "real"},
            operation_id="x",
        )
    with pytest.raises(schemas.HFCorpusSchemaError):
        schemas.ToolRuntimeRecord(
            source_dataset_id=DATASET_ID,
            trace_type="request_shape_trace",  # wrong type
            provenance="p",
            field_quality={"operation_id": "real"},
            operation_id="x",
        )
    # And a clean record builds.
    r = schemas.ToolRuntimeRecord(
        source_dataset_id=DATASET_ID,
        trace_type="tool_runtime_trace",
        provenance="p",
        field_quality={
            "operation_id": "real",
            "tool_name": "real",
            "duration_ms": "real",
            "status": "real",
        },
        operation_id="op-1",
        tool_name="surface_affinity",
        duration_ms=12.3,
        status="ok",
    )
    assert r.duration_ms == 12.3


def test_trust_tier_for_tool_runtime_trace_is_tier3() -> None:
    assert (schemas.CANONICAL_TRACE_TYPE_TO_TRUST_TIER["tool_runtime_trace"]
            == "tier_3_cluster_scheduler_traces")


@pytest.mark.parametrize("config", CONFIGS)
def test_registry_trust_tier_is_tier3_not_tier1(config: str) -> None:
    with open(DISC_DIR / "canonical_corpus_registry.json") as fh:
        reg = json.load(fh)
    entries = [e for e in reg["entries"]
               if e["dataset_id"] == DATASET_ID
               and e["config_name"] == config]
    assert len(entries) == 1, (
        f"expected exactly one canonical entry for {DATASET_ID}@{config}"
    )
    e = entries[0]
    assert e["trust_tier"] == "tier_3_cluster_scheduler_traces"
    # Pilot telemetry remains the only Tier-1 source. This dataset must
    # NEVER claim Tier 1 or Tier 2.
    assert e["trust_tier"] != "tier_1_real_pilot_telemetry"
    assert e["trust_tier"] != "tier_2_public_telemetry_traces"


# ───────────────────────── 6. Signal coverage ─────────────────────────────


def test_operations_signals_are_explicit_and_disjoint() -> None:
    s = _summary("operations")
    avail = set(s["available_signals"])
    miss = set(s["missing_signals"])
    assert avail.isdisjoint(miss), (
        f"available and missing signals overlap: {avail & miss}"
    )
    # The operations config MUST advertise these tool-runtime signals.
    expected_present = {
        "arrivals",
        "request_timestamps",
        "latency",
        "duration_measured",
        "tool_routing",
        "tool_failure_label",
        "tool_cancellation_label",
        "args_fingerprint_for_cache_reuse",
        "workload_shape",
    }
    missing_from_avail = expected_present - avail
    assert not missing_from_avail, (
        f"operations missing expected signals: {missing_from_avail}"
    )


def test_operations_does_not_claim_gpu_serving_signals() -> None:
    """No model_id / no input/output_tokens / no GPU type / no queue /
    no replica / no TTFT / no TPOT — Lightcap is tool-runtime telemetry,
    not LLM serving telemetry. These MUST live in missing_signals."""
    s = _summary("operations")
    miss = set(s["missing_signals"])
    forbidden_in_avail = {
        "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
        "model_load_event", "model_unload_event",
    }
    avail = set(s["available_signals"])
    leak = forbidden_in_avail & avail
    assert leak == set(), (
        f"operations falsely advertises serving signals it does NOT measure: "
        f"{leak}"
    )
    # And the absences are explicit.
    expected_missing = {
        "ttft", "tpot", "queue_state", "gpu_utilization", "replica_count",
    }
    not_recorded = expected_missing - miss
    assert not_recorded == set(), (
        f"operations did NOT record absences for: {not_recorded}"
    )


# ───────────────────────── 7. Limitations pinning ─────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_limitations_pin_no_llm_serving_signal(config: str) -> None:
    s = _summary(config)
    lims = s["limitations"]
    assert isinstance(lims, list) and lims, (
        f"{config}: limitations must be non-empty"
    )
    joined = " ".join(lims)
    # Must call out the closed-runtime / not-LLM-serving caveat somewhere.
    assert (
        "NOT GPU" in joined or "NOT LLM" in joined
        or "tool-runtime" in joined.lower() or "tool runtime" in joined.lower()
    ), (
        f"{config}: limitations do not pin the not-LLM-serving caveat: "
        f"{lims}"
    )


# ───────────────────────── 8. Bounded normalized sample ───────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_normalized_sample_is_committed_and_bounded(config: str) -> None:
    p = _processed_dir(config) / "normalized_sample.jsonl"
    assert p.exists(), f"missing normalized_sample: {p}"
    size = p.stat().st_size
    assert size > 0
    # 100 MiB cap from the policy.
    assert size <= 100 * 1024 * 1024, (
        f"{config}: normalized_sample.jsonl {size} bytes exceeds 100 MiB cap"
    )
    # The sha256 in the summary must match the actual file (or at least
    # be a 64-hex sha256 string).
    s = _summary(config)
    expected_sha = s.get("normalized_sample_sha256")
    assert isinstance(expected_sha, str) and len(expected_sha) == 64
    with open(p, "rb") as fh:
        actual_sha = hashlib.sha256(fh.read()).hexdigest()
    assert actual_sha == expected_sha, (
        f"{config}: normalized_sample.jsonl sha256 mismatch: "
        f"recorded={expected_sha} actual={actual_sha}"
    )


# ───────────────────────── 9. Registry consistency ────────────────────────


def test_canonical_registry_includes_both_configs() -> None:
    with open(DISC_DIR / "canonical_corpus_registry.json") as fh:
        reg = json.load(fh)
    ids = {(e["dataset_id"], e.get("config_name")) for e in reg["entries"]}
    for cfg in CONFIGS:
        assert (DATASET_ID, cfg) in ids, (
            f"canonical registry missing {DATASET_ID}@{cfg}"
        )


def test_candidates_json_records_focused_audit_2026_06_01c() -> None:
    with open(DISC_DIR / "hf_dataset_candidates.json") as fh:
        cands = json.load(fh)
    assert "focused_audit_2026_06_01c" in cands
    block = cands["focused_audit_2026_06_01c"]
    assert DATASET_ID in block["datasets"]
    assert block["new_canonical_type"] == "tool_runtime_trace"
    assert block["trust_tier"] == "tier_3_cluster_scheduler_traces"
    assert block["production_claim"] is False
    assert block["modifies_robust_energy_engine"] is False
    # And the candidate row itself reflects the ingest decision.
    candidates = cands["candidates"]
    light = [c for c in candidates if c.get("dataset_id") == DATASET_ID]
    assert len(light) == 1
    assert light[0]["recommended_action"] == "ingest_now_bounded"
    assert light[0]["audit_round"] == "focused_audit_2026_06_01c"


# ───────────────────────── 10. Promotion rules wiring ─────────────────────


def test_promotion_rules_include_tool_runtime_trace() -> None:
    allowed = promotion.TRACE_TYPE_TO_ALLOWED_PROMOTIONS["tool_runtime_trace"]
    assert "promoted_for_backtest" in allowed
    assert "promoted_for_constraint_aware_evaluation" in allowed
    assert "promoted_for_training_priors" in allowed
    # NOT dynamic_calibration — no queue / replica / GPU-util signal.
    assert "promoted_for_dynamic_calibration" not in allowed
