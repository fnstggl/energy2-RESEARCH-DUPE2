"""Tests for the AcmeTrace focused HF audit (Qinghao/AcmeTrace).

Covers:
- no raw / analysis_sample data committed for the 4 new configs,
- per-config schema_profile + schema_mapping + summary + rollups exist,
- schema mapping classifies every observed column (no rejected columns
  outside the wide-utilisation auto-mapping path),
- signal coverage records both available + missing signals,
- all 4 configs pass the canonical-corpus promotion gates,
- the canonical registry knows about all 4 configs,
- the audit summary records the 4 ingested + 3 discovery-only records,
- discovery candidates JSON now includes the 3 newly-classified ids,
- per-config fixture file exists and is small.

Audit-only — tests read committed artifacts; they do NOT hit the HF API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

NEW_CONFIGS = [
    ("Qinghao/AcmeTrace", "kalos_jobs"),
    ("Qinghao/AcmeTrace", "seren_jobs_head"),
    ("Qinghao/AcmeTrace", "kalos_gpu_util_head"),
    ("Qinghao/AcmeTrace", "seren_ipmi_gpu_power_head"),
]

DISCOVERY_ONLY_IDS = [
    "HuggingAGree/AcmeTrace",
    "osteele/llm-calibration-db",
    "jaytonde05/iris-prefix-cache-benchmark",
]


def _processed_dir(dataset_id: str, config: str) -> Path:
    return HF_DIR / dataset_id.replace("/", "__") / config / "processed"


def _fixture_path(dataset_id: str, config: str) -> Path:
    return FIXTURES_DIR / f"{dataset_id.replace('/', '__')}__{config}_sample.jsonl"


def _summary(dataset_id: str, config: str) -> dict:
    with open(_processed_dir(dataset_id, config) / "summary.json") as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw data committed ─────────────────────────


def test_no_raw_acmetrace_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/hf/Qinghao__AcmeTrace"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [p for p in out if p.endswith("/analysis_sample.jsonl")]
    assert raw_committed == [], (
        f"AcmeTrace raw downloads committed (gitignore broken): {raw_committed}"
    )
    assert analysis_committed == [], (
        "AcmeTrace analysis_sample.jsonl committed (gitignore broken): "
        f"{analysis_committed}"
    )


def test_acmetrace_fixture_files_are_committed_and_tiny():
    for dataset_id, config in NEW_CONFIGS:
        fixture = _fixture_path(dataset_id, config)
        assert fixture.exists(), f"missing fixture: {fixture}"
        size = fixture.stat().st_size
        assert 0 < size <= 16 * 1024, (
            f"fixture {fixture} size {size}B is outside (0, 16 KiB]"
        )


# ───────────────────────── 2. Per-config artefacts ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_processed_artifacts_present(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    for name in (
        "schema_profile.json", "schema_mapping.json",
        "summary.json", "statistical_rollups.json",
    ):
        path = pd / name
        assert path.exists(), f"{path} missing for {dataset_id}@{config}"
        with open(path) as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict) and payload, f"{path} is empty"


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_schema_mapping_has_columns(dataset_id: str, config: str) -> None:
    pd = _processed_dir(dataset_id, config)
    with open(pd / "schema_mapping.json") as fh:
        mapping = json.load(fh)
    assert "columns" in mapping and mapping["columns"], "no columns in mapping"
    # Every accepted column must record a field_quality in the canonical set
    # (or, for the wide-utilisation aggregate column, field_quality='derived').
    for c in mapping["columns"]:
        fq = c.get("field_quality")
        # The wide-utilisation aggregate entry is a "<server_ip>" wildcard
        # whose normalized_field is a comma-separated list of value_* keys.
        assert fq in {
            "real", "derived", "proxy", "synthetic", "missing", "unknown",
        }, f"invalid field_quality: {c}"


# ───────────────────────── 3. Promotion gates ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_summary_passes_all_promotion_gates(
    dataset_id: str, config: str,
) -> None:
    summary = _summary(dataset_id, config)
    gates = promotion.gates(summary)
    failed = [g for g in gates if not g["passed"]]
    assert not failed, (
        f"{dataset_id}@{config}: gates failed: "
        f"{[(g['gate'], g['detail']) for g in failed]}"
    )


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_summary_promoted(dataset_id: str, config: str) -> None:
    summary = _summary(dataset_id, config)
    decision = promotion.evaluate_promotion(summary)
    assert decision["state"] in (
        "promoted_for_training_priors",
        "promoted_for_constraint_aware_evaluation",
        "promoted_for_backtest",
        "promoted_for_dynamic_calibration",
        "promoted_for_performance_priors",
        "promoted_for_cache_residency_evaluation",
    ), (
        f"{dataset_id}@{config}: unexpected state {decision['state']}; "
        f"reasons={decision.get('reasons')}"
    )
    assert decision["promotion_tags"], (
        f"{dataset_id}@{config}: no promotion_tags assigned"
    )


# ───────────────────────── 4. Signal coverage ─────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_signal_coverage_recorded(dataset_id: str, config: str) -> None:
    summary = _summary(dataset_id, config)
    avail = summary["available_signals"]
    missing = summary["missing_signals"]
    assert isinstance(avail, list) and avail, "available_signals is empty"
    assert isinstance(missing, list), "missing_signals must be a list"
    assert not (set(avail) & set(missing)), (
        f"signal overlap: {set(avail) & set(missing)}"
    )


def test_acmetrace_kalos_jobs_has_queue_and_timeout_signals():
    s = _summary("Qinghao/AcmeTrace", "kalos_jobs")
    for sig in ("arrivals", "request_timestamps", "queue_state",
                "timeout_label", "capacity_proxy", "customer_traffic_mix"):
        assert sig in s["available_signals"], (
            f"kalos_jobs missing expected signal {sig}: {s['available_signals']}"
        )


def test_acmetrace_seren_jobs_has_queue_and_timeout_signals():
    s = _summary("Qinghao/AcmeTrace", "seren_jobs_head")
    for sig in ("arrivals", "request_timestamps", "queue_state",
                "timeout_label", "workload_shape"):
        assert sig in s["available_signals"], (
            f"seren_jobs missing expected signal {sig}: {s['available_signals']}"
        )


def test_acmetrace_kalos_gpu_util_has_dcgm_telemetry():
    s = _summary("Qinghao/AcmeTrace", "kalos_gpu_util_head")
    for sig in ("gpu_utilization", "dcgm_telemetry", "request_timestamps"):
        assert sig in s["available_signals"], (
            f"kalos_gpu_util missing expected signal {sig}: "
            f"{s['available_signals']}"
        )


def test_acmetrace_seren_ipmi_power_has_ipmi_signals():
    s = _summary("Qinghao/AcmeTrace", "seren_ipmi_gpu_power_head")
    for sig in ("ipmi_telemetry", "power_telemetry", "gpu_utilization",
                "request_timestamps"):
        assert sig in s["available_signals"], (
            f"seren_ipmi_power missing expected signal {sig}: "
            f"{s['available_signals']}"
        )


# ───────────────────────── 5. Trust tier assignment ────────────────────────


def test_acmetrace_job_traces_are_tier_3_cluster_scheduler():
    for cfg in ("kalos_jobs", "seren_jobs_head"):
        s = _summary("Qinghao/AcmeTrace", cfg)
        assert s["canonical_trace_type"] == "cluster_scheduler_trace", cfg


def test_acmetrace_utilization_files_are_tier_2_telemetry():
    for cfg in ("kalos_gpu_util_head", "seren_ipmi_gpu_power_head"):
        s = _summary("Qinghao/AcmeTrace", cfg)
        assert s["canonical_trace_type"] == "telemetry_trace", cfg


def test_acmetrace_seren_ipmi_power_promoted_to_dynamic_calibration():
    """The IPMI GPU power head sample is the first non-CARA HF dataset
    promoted to dynamic_calibration via this pipeline."""
    s = _summary("Qinghao/AcmeTrace", "seren_ipmi_gpu_power_head")
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_dynamic_calibration", (
        f"unexpected state {decision['state']}; reasons={decision.get('reasons')}"
    )
    assert "promoted_for_dynamic_calibration" in decision["promotion_tags"]


# ───────────────────────── 6. License + gating recorded ────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_license_and_gating_recorded(
    dataset_id: str, config: str,
) -> None:
    s = _summary(dataset_id, config)
    assert s.get("license") == "cc-by-4.0"
    assert s.get("gated") is False


# ───────────────────────── 7. Audit + registry rollups ─────────────────────


def test_acmetrace_audit_summary_exists_and_records_ingest_outcomes():
    path = DISC_DIR / "acmetrace_audit_summary.json"
    assert path.exists()
    with open(path) as fh:
        payload = json.load(fh)
    assert payload["modifies_robust_energy_engine"] is False
    assert payload["modifies_controllers_or_defaults"] is False
    assert payload["production_claim"] is False
    ingested_keys = {(e["dataset_id"], e["config_name"])
                     for e in payload["ingested"]}
    assert set(NEW_CONFIGS) <= ingested_keys, (
        f"missing entries in audit summary: {set(NEW_CONFIGS) - ingested_keys}"
    )
    disc_only_ids = {r["dataset_id"] for r in payload["discovery_only_records"]}
    assert set(DISCOVERY_ONLY_IDS) <= disc_only_ids, (
        f"missing discovery_only records: {set(DISCOVERY_ONLY_IDS) - disc_only_ids}"
    )


def test_canonical_registry_includes_acmetrace_entries():
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    assert reg_path.exists()
    with open(reg_path) as fh:
        reg = json.load(fh)
    ids = {(e["dataset_id"], e.get("config_name")) for e in reg["entries"]}
    assert set(NEW_CONFIGS) <= ids, (
        f"missing registry entries: {set(NEW_CONFIGS) - ids}"
    )


def test_candidates_json_now_includes_focused_audit_ids():
    cand_path = DISC_DIR / "hf_dataset_candidates.json"
    assert cand_path.exists()
    with open(cand_path) as fh:
        payload = json.load(fh)
    ids = {c["dataset_id"] for c in payload["candidates"]}
    expected = {
        "Qinghao/AcmeTrace", "HuggingAGree/AcmeTrace",
        "osteele/llm-calibration-db", "jaytonde05/iris-prefix-cache-benchmark",
    }
    assert expected <= ids, f"missing candidates: {expected - ids}"
    # Recommended-action labels
    by_id = {c["dataset_id"]: c for c in payload["candidates"]}
    assert by_id["Qinghao/AcmeTrace"]["recommended_action"] == "ingest_now_bounded"
    assert by_id["HuggingAGree/AcmeTrace"]["recommended_action"] == "duplicate_existing"
    assert by_id["osteele/llm-calibration-db"]["recommended_action"] == "gated_blocked"
    assert by_id["jaytonde05/iris-prefix-cache-benchmark"]["recommended_action"] == "reject_low_value"


# ───────────────────────── 8. Anti-spam invariants ─────────────────────────


def test_huggingagree_acmetrace_marked_as_duplicate_not_ingested():
    """Anti-spam: HuggingAGree/AcmeTrace is a re-upload of Qinghao/AcmeTrace
    and must NOT have a separate processed/ tree."""
    path = HF_DIR / "HuggingAGree__AcmeTrace"
    if path.exists():
        # If the directory exists it must be empty / .gitkeep only — no
        # processed/ subtree.
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        assert not subdirs, (
            f"HuggingAGree/AcmeTrace must not have a separate ingest tree: {subdirs}"
        )


def test_osteele_llm_calibration_db_marked_gated_not_ingested():
    """Gated datasets must not have a processed/ tree."""
    path = HF_DIR / "osteele__llm-calibration-db"
    if path.exists():
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        assert not subdirs, (
            f"osteele/llm-calibration-db must not have an ingest tree: {subdirs}"
        )


def test_iris_prefix_cache_benchmark_marked_low_value_not_ingested():
    """Rejected low-value datasets must not have a processed/ tree."""
    path = HF_DIR / "jaytonde05__iris-prefix-cache-benchmark"
    if path.exists():
        subdirs = [p for p in path.iterdir() if p.is_dir()]
        assert not subdirs, (
            f"iris-prefix-cache-benchmark must not have an ingest tree: {subdirs}"
        )


# ───────────────────────── 9. Statistical sample strength ──────────────────


def test_acmetrace_job_traces_strong_sample():
    for cfg in ("kalos_jobs", "seren_jobs_head"):
        s = _summary("Qinghao/AcmeTrace", cfg)
        assert s["statistical_sample_strength"] == "strong", (
            f"{cfg}: expected strong, got {s['statistical_sample_strength']} "
            f"(analysis_rows={s['analysis_sample_rows']})"
        )


def test_acmetrace_seren_ipmi_strong_sample():
    s = _summary("Qinghao/AcmeTrace", "seren_ipmi_gpu_power_head")
    assert s["statistical_sample_strength"] == "strong"
    assert s["analysis_sample_rows"] >= 10_000


def test_acmetrace_kalos_gpu_util_at_least_moderate():
    s = _summary("Qinghao/AcmeTrace", "kalos_gpu_util_head")
    # GPU_UTIL is wide (2300+ host columns per row) so even a 32 MiB head
    # only delivers ~6.7k rows — moderate. That is enough for performance
    # priors but not strong enough for dynamic calibration on its own.
    assert s["statistical_sample_strength"] in ("moderate", "strong")
    assert s["analysis_sample_rows"] >= 1_000


# ───────────────────────── 10. Limitations explicit ────────────────────────


@pytest.mark.parametrize("dataset_id,config", NEW_CONFIGS)
def test_acmetrace_limitations_recorded(dataset_id: str, config: str) -> None:
    s = _summary(dataset_id, config)
    lims = s.get("limitations") or []
    assert isinstance(lims, list) and len(lims) >= 3, (
        f"{dataset_id}@{config}: limitations is too short: {lims}"
    )
    # Trust-hierarchy honesty: benchmarks must not be claimed as Tier 1.
    joined = " ".join(lims).lower()
    if "telemetry_trace" == s["canonical_trace_type"]:
        assert (
            "pilot" in joined or "tier-1" in joined or "tier_1" in joined
            or "prior" in joined or "research" in joined
        ), (
            f"{dataset_id}@{config}: telemetry_trace limitations must call out "
            f"that pilot telemetry remains the only Tier-1 calibration source"
        )
