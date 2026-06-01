"""Tests for the Round-3 HF discovery ingest of ``ssong1/llmperf-bedrock``.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.

Covered:

* one new ``latency_benchmark_trace`` config
  (``ssong1/llmperf-bedrock`` → ``bedrock_claude_instant_v1``)
* the Round-3 discovery-only / negative-result records
  (``DistServe/2025-05-06T14-…``, ``deepanjalimishra99/datacenter-traces``,
  ``intellistream/sage-control-plane-llm-workloads``, etc.)
* the closed-API caveat (must be pinned in ``limitations`` to prevent
  any downstream user from treating Bedrock TTFT as GPU-direct serving
  latency)
* the corpus invariant: no raw / analysis sample committed; license is
  declared and matches the dataset card (Apache-2.0); fixture is tiny
  + deterministic.
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

from aurelius.traces.hf_corpus import promotion  # noqa: E402

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "ssong1/llmperf-bedrock"
SAFE_NAME = "ssong1__llmperf-bedrock"
CONFIG = "bedrock_claude_instant_v1"

PROC_DIR = HF_DIR / SAFE_NAME / CONFIG / "processed"
FIXTURE = FIXTURES_DIR / f"{SAFE_NAME}__{CONFIG}_sample.jsonl"
AUDIT_PATH = DISC_DIR / "round3_broadened_discovery_audit_summary.json"

ROUND3_DISCOVERY_IDS = [
    "DistServe/2025-05-06T14-automatic-profiling",
    "DistServe/test-amd-ci-profiler",
    "DistServe/test-sample",
    "deepanjalimishra99/datacenter-traces",
    "intellistream/sage-control-plane-llm-workloads",
    "Nathan-Maine/dgx-spark-kv-cache-benchmark",
    "hlarcher/inference-benchmarker",
    "kshitijthakkar/moe-inference-benchmark",
]


def _summary() -> dict:
    with open(PROC_DIR / "summary.json") as fh:
        return json.load(fh)


def _audit() -> dict:
    with open(AUDIT_PATH) as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw / analysis files committed ─────────


def test_no_raw_or_analysis_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    analysis_committed = [
        p for p in out if p.endswith("/analysis_sample.jsonl")
    ]
    assert raw_committed == [], (
        f"Raw downloads committed for {SAFE_NAME}: {raw_committed}"
    )
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed for {SAFE_NAME}: "
        f"{analysis_committed}"
    )


def test_fixture_is_committed_and_tiny():
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    size = FIXTURE.stat().st_size
    assert 0 < size <= 16 * 1024, (
        f"fixture {FIXTURE} size {size}B outside (0, 16 KiB]"
    )


def test_committed_normalized_sample_bounded_under_512_kib():
    """Committed normalized sample must be ≤ 100 MB (corpus policy) AND in
    this dataset's case ≤ 512 KiB (per script budget)."""
    s = _summary()
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    assert path.exists(), f"missing committed normalised sample {path}"
    sz = path.stat().st_size
    assert 0 < sz <= 100 * 1024 * 1024, (
        f"committed normalised sample {path} > 100 MB cap (size {sz})"
    )
    # Hard local bound — we expect ~ 380 KB; allow up to 512 KB.
    assert sz <= 512 * 1024, (
        f"committed normalised sample {path} unexpectedly large ({sz}B); "
        "tighten COMMITTED_NORMALIZED_MAX_BYTES if intentional."
    )


# ───────────────────────── 2. Processed artefacts present ───────────────


def test_processed_artifacts_present():
    for name in (
        "schema_profile.json", "schema_mapping.json", "summary.json",
        "statistical_rollups.json", "committed_normalized_sample.jsonl",
    ):
        path = PROC_DIR / name
        assert path.exists(), f"{path} missing"
        if path.suffix == ".json":
            with open(path) as fh:
                payload = json.load(fh)
            assert isinstance(payload, dict) and payload, f"{path} empty"


# ───────────────────────── 3. Schema profile + mapping ─────────────────


def test_schema_profile_records_dtypes_and_examples():
    with open(PROC_DIR / "schema_profile.json") as fh:
        prof = json.load(fh)
    assert prof["dataset_id"] == DATASET_ID
    assert prof["config_name"] == CONFIG
    assert prof["inspected_row_count"] == 350
    assert prof["unknown_columns"] == []
    expected_raw = {
        "end_to_end_latency_s", "error_code", "error_msg",
        "inter_token_latency_s", "number_input_tokens",
        "number_output_tokens", "number_total_tokens",
        "request_output_throughput_token_per_s", "ttft_s",
    }
    assert expected_raw.issubset(set(prof["raw_columns"])), (
        f"missing raw columns: "
        f"{expected_raw - set(prof['raw_columns'])}"
    )
    # Every populated column has at least one dtype recorded.
    for col in prof["raw_columns"]:
        assert prof["dtypes"][col], f"empty dtypes for {col}"


def test_schema_mapping_field_quality_and_signal_category():
    with open(PROC_DIR / "schema_mapping.json") as fh:
        m = json.load(fh)
    assert m["dataset_id"] == DATASET_ID
    assert m["columns"], "no columns in mapping"
    allowed_quality = {"real", "derived", "proxy", "synthetic",
                       "missing", "unknown"}
    allowed_categories = {
        "request_arrival", "request_completion", "latency", "queue",
        "throughput", "tokens", "gpu_resource", "memory", "autoscaling",
        "routing", "cache_residency", "failure_timeout", "scheduler_state",
        "cost_energy_carbon", "metadata_only", "irrelevant",
    }
    for c in m["columns"]:
        assert c["field_quality"] in allowed_quality, c
        assert c["aurelius_signal_category"] in allowed_categories, c
        assert c["raw_column_name"]
        assert c["normalized_field"]
    # At least one latency-category mapping and one throughput-category.
    cats = {c["aurelius_signal_category"] for c in m["columns"]}
    assert "latency" in cats
    assert "throughput" in cats


# ───────────────────────── 4. Promotion gates pass ─────────────────────


def test_promotion_gates_all_pass():
    s = _summary()
    gate_log = promotion.gates(s)
    failed = [g for g in gate_log if not g["passed"]]
    assert not failed, f"failed gates: {[g['gate'] for g in failed]}"


def test_promotion_state_is_performance_priors():
    s = _summary()
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] == "promoted_for_performance_priors"
    assert "promoted_for_performance_priors" in decision["promotion_tags"]


def test_canonical_trace_type_is_latency_benchmark():
    s = _summary()
    assert s["canonical_trace_type"] == "latency_benchmark_trace"


def test_statistical_sample_strength_is_moderate():
    """350 rows × 4 stratified runs → moderate (not strong, not weak).

    This must be preserved so the promotion harness allows
    ``promoted_for_performance_priors`` (min=moderate) but blocks
    ``promoted_for_dynamic_calibration`` (min=strong).
    """
    s = _summary()
    assert s["statistical_sample_strength"] == "moderate"


# ───────────────────────── 5. Registry + audit ──────────────────────────


def test_registry_contains_new_entry():
    reg = promotion.load_canonical_registry(
        str(DISC_DIR / "canonical_corpus_registry.json")
    )
    assert reg, "canonical registry missing"
    matches = [
        e for e in reg["entries"]
        if e["dataset_id"] == DATASET_ID and e["config_name"] == CONFIG
    ]
    assert len(matches) == 1, (
        f"expected exactly one entry for {DATASET_ID}/{CONFIG}, "
        f"got {len(matches)}"
    )
    entry = matches[0]
    assert entry["trust_tier"] == "tier_4_latency_benchmark_traces"
    assert entry["license"] == "apache-2.0"
    assert entry["gated"] is False
    assert entry["promotion_state"] == "promoted_for_performance_priors"


def test_audit_summary_records_ingest_and_discovery_records():
    a = _audit()
    assert a["production_claim"] is False
    assert a["modifies_robust_energy_engine"] is False
    assert a["modifies_controllers_or_defaults"] is False
    ingested_ids = {x["dataset_id"] for x in a["ingested"]}
    assert DATASET_ID in ingested_ids
    discovery_ids = {r["dataset_id"] for r in a["discovery_only_records"]}
    missing = set(ROUND3_DISCOVERY_IDS) - discovery_ids
    assert not missing, f"audit missing discovery records: {missing}"


def test_candidate_registry_has_ingest_recommendation_for_bedrock():
    candidates_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(candidates_path) as fh:
        d = json.load(fh)
    for c in d.get("candidates", []):
        if c["dataset_id"] == DATASET_ID:
            assert c["recommended_action"] == "ingest_now_bounded"
            assert c["candidate_trace_type"] == "latency_benchmark_trace"
            assert c["license"] == "apache-2.0"
            assert c["gated_status"] == "public"
            return
    pytest.fail(
        f"{DATASET_ID} not in hf_dataset_candidates.json"
    )


def test_candidate_registry_has_discovery_only_round3_records():
    candidates_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(candidates_path) as fh:
        d = json.load(fh)
    cand_ids = {c["dataset_id"] for c in d.get("candidates", [])}
    missing = set(ROUND3_DISCOVERY_IDS) - cand_ids
    assert not missing, (
        f"candidate registry missing Round-3 discovery records: {missing}"
    )


# ───────────────────────── 6. Closed-API caveat enforcement ─────────────


def test_limitations_pin_closed_api_caveat():
    """Must explicitly call out the closed-managed-API caveat so no
    downstream user treats Bedrock latency as GPU-direct serving."""
    s = _summary()
    text = " ".join(s["limitations"]).lower()
    assert "closed" in text and "managed" in text, (
        f"closed-API caveat missing from limitations: {s['limitations']}"
    )
    assert "tier 4" in text, "tier-4 note missing"
    assert "not pilot telemetry" in text or "not pilot" in text, (
        f"pilot-telemetry disclaimer missing: {s['limitations']}"
    )


def test_concurrency_is_constant_and_recorded():
    """All 4 runs in this snapshot use num_concurrent_requests=1; this
    must be reflected in the limitations + missing_signals so the
    Aurelius queue-risk module never consumes this dataset."""
    s = _summary()
    text = " ".join(s["limitations"]).lower()
    assert "concurrency" in text and "1" in text, (
        f"concurrency=1 caveat missing: {s['limitations']}"
    )
    # The queue-state signal must be marked missing.
    assert "queue_state" in s["missing_signals"]
    assert "queue_wait" in s["missing_signals"]


def test_no_gpu_serving_signals_claimed():
    """The dataset is API-only; no GPU type / GPU utilisation / batch_size
    must appear in available_signals."""
    s = _summary()
    avail = set(s["available_signals"])
    forbidden_in_avail = {
        "gpu_type", "gpu_utilization", "gpu_memory",
        "batch_size", "kernel_duration", "kv_cache_size",
    }
    leaked = avail & forbidden_in_avail
    assert not leaked, (
        f"closed-API dataset claims GPU/serving-internal signals: {leaked}"
    )


# ───────────────────────── 7. Signal taxonomy ──────────────────────────


def test_available_signals_include_ttft_itl_e2e_and_throughput():
    s = _summary()
    avail = set(s["available_signals"])
    required = {"ttft", "itl", "e2e_latency", "throughput"}
    assert required.issubset(avail), (
        f"required latency signals missing: {required - avail}"
    )


# ───────────────────────── 8. Fixture determinism ──────────────────────


def test_fixture_sha256_matches_summary():
    s = _summary()
    fx = FIXTURE
    h = hashlib.sha256()
    with open(fx, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    assert h.hexdigest() == s["sample_sha256"], (
        f"fixture sha256 drift: file={h.hexdigest()} vs summary="
        f"{s['sample_sha256']}"
    )


def test_committed_normalized_sample_sha256_matches_summary():
    s = _summary()
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    assert h.hexdigest() == s["committed_normalized_sample_sha256"]


# ───────────────────────── 9. Statistical rollups ──────────────────────


def test_statistical_rollups_has_per_run_strata():
    """Rollups must record per-run subgroup counts AND per-run percentile
    summaries — required for the per-run stratification claim in the
    summary."""
    with open(PROC_DIR / "statistical_rollups.json") as fh:
        r = json.load(fh)
    counts = r["subgroup_counts"]
    assert len(counts) == 4, (
        f"expected 4 run strata, got {len(counts)}: {counts}"
    )
    # Each run must have >= 50 rows (strong enough for per-run quantiles).
    for label, n in counts.items():
        assert n >= 50, f"subgroup {label} only has {n} rows"
    # by_strata must contain per-run latency rollups.
    assert len(r["by_strata"]) == 4, (
        f"per-strata rollups incomplete: {list(r['by_strata'].keys())}"
    )
    for label, stats in r["by_strata"].items():
        assert "ttft_ms" in stats, (
            f"missing TTFT rollup for {label}"
        )
        assert stats["ttft_ms"]["p95"] > 0
        assert stats["ttft_ms"]["p50"] > 0


def test_overall_rollup_records_p95_p99_for_ttft():
    with open(PROC_DIR / "statistical_rollups.json") as fh:
        r = json.load(fh)
    o = r["overall"]
    for k in ("ttft_ms", "itl_ms", "e2e_latency_ms"):
        assert k in o, f"missing overall rollup for {k}"
        assert o[k]["n"] == 350
        assert o[k]["p95"] >= o[k]["p50"]
        assert o[k]["p99"] >= o[k]["p95"]


# ───────────────────────── 10. License + redistribution ────────────────


def test_license_is_apache2_and_dataset_is_public():
    s = _summary()
    assert s["license"] == "apache-2.0"
    assert s["gated"] is False
    assert s["committed_normalized_sample_reason_skipped"] is None
    assert s["committed_normalized_sample_rows"] > 0


def test_provenance_pins_dataset_and_git_sha():
    s = _summary()
    prov = s["provenance"]
    assert DATASET_ID in prov, f"provenance lacks dataset id: {prov}"
    assert CONFIG in prov, f"provenance lacks config name: {prov}"
    assert "git=" in prov, f"provenance lacks git sha: {prov}"
