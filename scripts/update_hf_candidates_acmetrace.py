#!/usr/bin/env python3
"""Add Qinghao/AcmeTrace + HuggingAGree/AcmeTrace + osteele/llm-calibration-db
to ``data/external/hf_discovery/hf_dataset_candidates.json``.

The previous discovery run (PR #129) missed these IDs because the keyword
groups did not match — AcmeTrace's HF card uses "Acme" / "Shanghai AI Lab"
phrasing rather than the canonical DCGM / scheduler keywords. This script
seeds the four short-term-mission dataset ids directly using the same
classification/scoring functions, so the candidate registry keeps a single
authoritative memory of what has been evaluated.

Audit-only. Does not modify scheduler / controllers / robust energy engine.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDIDATES_PATH = (
    REPO_ROOT / "data" / "external" / "hf_discovery" / "hf_dataset_candidates.json"
)

# Manually-classified records from the focused audit (matches the audit
# summary at data/external/hf_discovery/acmetrace_audit_summary.json).
FOCUSED_AUDIT_RECORDS = [
    {
        "dataset_id": "Qinghao/AcmeTrace",
        "dataset_url": "https://huggingface.co/datasets/Qinghao/AcmeTrace",
        "candidate_trace_type": "cluster_scheduler_trace",
        "trust_level": "tier_3_cluster_scheduler_traces",
        "license": "cc-by-4.0",
        "gated_status": "public",
        "downloads": 212,
        "likes": 9,
        "last_modified": "2024-03-12T19:03:08.000Z",
        "estimated_size": ["80GB_full_with_utilization"],
        "configs": [
            "kalos_jobs", "seren_jobs_head",
            "kalos_gpu_util_head", "seren_ipmi_gpu_power_head",
        ],
        "feature_names": [
            "job_id", "user", "node_num", "gpu_num", "cpu_num", "type",
            "state", "submit_time", "start_time", "end_time", "duration",
            "queue", "gpu_time",
            "mem_per_pod_GB", "shared_mem_per_pod", "fail_time", "stop_time",
            "Time", "<per_server_ip_utilization_or_power>",
        ],
        "available_splits": [
            {"config": "kalos_jobs", "split": "csv_full",
             "approx_rows": 62413},
            {"config": "seren_jobs_head", "split": "csv_head",
             "approx_rows": 79999},
            {"config": "kalos_gpu_util_head", "split": "dcgm_15s_head",
             "approx_rows": 6680},
            {"config": "seren_ipmi_gpu_power_head", "split": "ipmi_15s_head",
             "approx_rows": 79999},
        ],
        "schema_available": True,
        "matched_keywords": [
            "manual_seed::nsdi24_llm_datacenter_characterization",
            "manual_seed::shanghai_ai_lab_acme",
            "schema::queue_wait_real",
            "schema::dcgm_gpu_utilization_per_host",
            "schema::ipmi_power_per_host",
        ],
        "available_signals": [
            "request_timestamps", "arrivals", "queue_state", "timeout_label",
            "capacity_proxy", "customer_traffic_mix", "workload_shape",
            "latency", "gpu_utilization", "dcgm_telemetry", "ipmi_telemetry",
            "power_telemetry",
        ],
        "missing_signals": [
            "ttft", "tpot", "cache_reuse", "prefix_reuse", "kv_block_hashes",
            "sla_label", "model_load_event", "model_unload_event",
            "replica_count", "cost_or_region",
        ],
        "classification_evidence": {
            "cluster_scheduler_trace": [
                "scheduler_trace", "job_queue", "queue_wait", "gpu_count",
                "submit_time",
            ],
            "telemetry_trace": [
                "dcgm", "prometheus_export", "ipmi_power",
            ],
        },
        "frontier_value_score": 5,
        "ingestion_feasibility_score": 4,
        "schema_quality_score": 5,
        "production_similarity_score": 4,
        "overall_priority_score": 9.5,
        "recommended_action": "ingest_now_bounded",
        "aurelius_use_case": (
            "Real Shanghai AI Lab Kalos + Seren cluster scheduler trace "
            "(NSDI'24 'Characterization of LLM Development in the Datacenter'). "
            "Job-level: real queue_wait, real timeout/failure labels, GPU/CPU "
            "request counts, workload type. DCGM-collected per-host GPU "
            "utilisation + IPMI per-host GPU power telemetry. First HF "
            "Tier 3 cluster_scheduler_trace AND first HF Tier 2 GPU power "
            "telemetry trace promoted to dynamic_calibration."
        ),
        "not_recommended_uses": [
            "TTFT/TPOT calibration (no per-token timing)",
            "Cache-hit / prefix-cache calibration (not measured)",
            "Production-truth SLA calibration (still benchmark/research-class)",
        ],
        "discovery_timestamp_s": time.time(),
        "focused_audit_2026_06_01": True,
    },
    {
        "dataset_id": "HuggingAGree/AcmeTrace",
        "dataset_url": "https://huggingface.co/datasets/HuggingAGree/AcmeTrace",
        "candidate_trace_type": "cluster_scheduler_trace",
        "trust_level": "tier_3_cluster_scheduler_traces",
        "license": "cc-by-4.0",
        "gated_status": "public",
        "downloads": 169,
        "likes": 0,
        "last_modified": "2026-04-27T03:56:58.000Z",
        "estimated_size": ["mirror_of_Qinghao_AcmeTrace"],
        "configs": ["mirror"],
        "feature_names": [],
        "available_splits": [],
        "schema_available": True,
        "matched_keywords": ["manual_seed::mirror_of_qinghao_acmetrace"],
        "available_signals": [],
        "missing_signals": [],
        "classification_evidence": {
            "cluster_scheduler_trace": [
                "scheduler_trace", "submit_time",
            ],
        },
        "frontier_value_score": 1,
        "ingestion_feasibility_score": 4,
        "schema_quality_score": 5,
        "production_similarity_score": 1,
        "overall_priority_score": 1.0,
        "recommended_action": "duplicate_existing",
        "aurelius_use_case": (
            "Re-upload of Qinghao/AcmeTrace with the same 75 files. "
            "Discovery-only — no separate ingest. Qinghao/AcmeTrace is the "
            "canonical entry."
        ),
        "not_recommended_uses": ["re-ingestion (use Qinghao/AcmeTrace)"],
        "discovery_timestamp_s": time.time(),
        "focused_audit_2026_06_01": True,
    },
    {
        "dataset_id": "osteele/llm-calibration-db",
        "dataset_url": "https://huggingface.co/datasets/osteele/llm-calibration-db",
        "candidate_trace_type": "latency_benchmark_trace",
        "trust_level": "tier_4_latency_benchmark_traces",
        "license": "mit",
        "gated_status": "gated_manual",
        "downloads": 18,
        "likes": 0,
        "last_modified": "2026-03-11T11:10:35.000Z",
        "estimated_size": ["n<1K"],
        "configs": [
            "calibration_runs", "calibration_stats", "dtype_calibration",
            "inference_overhead", "inference_overhead_measurements",
            "layer_timing", "memory_calibration", "system_load_snapshots",
            "telemetry_samples",
        ],
        "feature_names": [
            "<not_inspectable_until_manual_gate_approved>",
        ],
        "available_splits": [],
        "schema_available": False,
        "matched_keywords": [
            "manual_seed::llm_training_calibration",
            "tag::gpu", "tag::calibration", "tag::performance",
        ],
        "available_signals": [],
        "missing_signals": [],
        "classification_evidence": {
            "latency_benchmark_trace": [
                "tag::benchmarking", "tag::performance", "tag::calibration",
            ],
            "telemetry_trace": [
                "siblings::telemetry_samples.parquet",
                "siblings::system_load_snapshots.parquet",
            ],
        },
        "frontier_value_score": 3,
        "ingestion_feasibility_score": 1,
        "schema_quality_score": 3,
        "production_similarity_score": 3,
        "overall_priority_score": 2.0,
        "recommended_action": "gated_blocked",
        "aurelius_use_case": (
            "Empirical GPU training timing measurements + telemetry samples "
            "across GPU architectures + model families (per HF card). Would "
            "be a Tier 4 latency benchmark + Tier 2 telemetry candidate IF "
            "the manual gate is approved. HF_TOKEN is not authorised for "
            "this dataset. Revisit if/when access is granted."
        ),
        "not_recommended_uses": ["ingest until manual gate approval recorded"],
        "discovery_timestamp_s": time.time(),
        "focused_audit_2026_06_01": True,
    },
]


def main() -> int:
    if not CANDIDATES_PATH.exists():
        print(f"candidates file not found at {CANDIDATES_PATH}", file=sys.stderr)
        return 1
    with open(CANDIDATES_PATH) as fh:
        payload = json.load(fh)

    candidates = payload.get("candidates", [])
    by_id = {c["dataset_id"]: c for c in candidates}

    # 1. Insert / overwrite the 3 newly-classified candidates.
    appended = 0
    for rec in FOCUSED_AUDIT_RECORDS:
        previous = by_id.get(rec["dataset_id"])
        if previous is None:
            appended += 1
        by_id[rec["dataset_id"]] = rec

    # 2. Stamp the focused-audit confirmation on the already-rejected iris-prefix
    iris_id = "jaytonde05/iris-prefix-cache-benchmark"
    if iris_id in by_id:
        by_id[iris_id]["focused_audit_2026_06_01"] = True
        by_id[iris_id]["focused_audit_decision"] = (
            "reject_low_value confirmed — 20 prompts only, no measured "
            "telemetry. Existing jaytonde05/prefixbench already covers the "
            "synthetic prefix-cache role with 4 jsonl files."
        )

    new_candidates = sorted(by_id.values(),
                            key=lambda c: -float(c.get("overall_priority_score", 0)))

    payload["candidates"] = new_candidates
    payload["candidate_count"] = len(new_candidates)

    # Track the focused audit run.
    fa_list = payload.get("focused_audit_2026_06_01", [])
    fa_list = list(set(fa_list)
                   | {rec["dataset_id"] for rec in FOCUSED_AUDIT_RECORDS}
                   | {iris_id})
    payload["focused_audit_2026_06_01"] = sorted(fa_list)
    payload["last_updated_at_s"] = time.time()

    with open(CANDIDATES_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(
        f"updated {CANDIDATES_PATH} "
        f"(+{appended} new candidates, total={len(new_candidates)}, "
        f"focused_audit_2026_06_01={len(payload['focused_audit_2026_06_01'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
