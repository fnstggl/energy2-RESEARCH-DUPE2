#!/usr/bin/env python3
"""Register the Lightcap/agent-runtime-telemetry-small bounded ingest into
the canonical corpus registry + update the candidates JSON entry to
reflect that the Round-5 ``defer_high_value_different_trace_class`` block
has been cleared by the introduction of the ``tool_runtime_trace``
canonical type.

Reads the per-config summary.json under
``data/external/hf/Lightcap__agent-runtime-telemetry-small/<config>/processed/``
and re-writes
``data/external/hf_discovery/canonical_corpus_registry.json`` via
``aurelius.traces.hf_corpus.promotion``. Also stamps a
``focused_audit_2026_06_01c`` block on the candidates JSON.

Audit-only. No production claim. No scheduler / controller / robust
energy engine modified.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
REGISTRY_PATH = DISC_DIR / "canonical_corpus_registry.json"
CANDIDATES_PATH = DISC_DIR / "hf_dataset_candidates.json"

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_DATASET = DATASET_ID.replace("/", "__")

NEW_ENTRIES = [
    (DATASET_ID, "operations"),
    (DATASET_ID, "tool_summary"),
]

AUDIT_BLOCK_KEY = "focused_audit_2026_06_01c"


def _register_canonical_corpus() -> int:
    if not REGISTRY_PATH.exists():
        print(f"registry not found at {REGISTRY_PATH}", file=sys.stderr)
        return 1
    with open(REGISTRY_PATH) as fh:
        reg = json.load(fh)

    existing = {(e["dataset_id"], e.get("config_name")): e
                for e in reg["entries"]}

    appended = 0
    for dataset_id, config in NEW_ENTRIES:
        safe = dataset_id.replace("/", "__")
        summary_path = HF_DIR / safe / config / "processed" / "summary.json"
        if not summary_path.exists():
            print(f"missing summary: {summary_path}", file=sys.stderr)
            continue
        with open(summary_path) as fh:
            summary = json.load(fh)
        decision = promotion.evaluate_promotion(summary)
        entry = promotion.build_registry_entry(summary, decision)
        existing[(dataset_id, config)] = entry
        appended += 1
        print(f"  registered {dataset_id}@{config} "
              f"state={decision['state']} tags={decision['promotion_tags']}")

    entries = sorted(existing.values(),
                     key=lambda e: (e["dataset_id"], e.get("config_name") or ""))
    payload = {
        "doc_version": reg.get("doc_version",
                                "hf_corpus_canonical_registry_v1"),
        "stage": reg.get("stage", "federated_benchmark_corpus_v1"),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "trust_hierarchy_note": reg.get(
            "trust_hierarchy_note",
            "Tier 1 (real pilot telemetry) remains the only production "
            "calibration source. Promotion here is research-class only.",
        ),
        "written_at_s": time.time(),
        "entry_count": len(entries),
        "entries": entries,
    }
    with open(REGISTRY_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nWrote {REGISTRY_PATH} ({len(entries)} entries, +{appended} new)")
    return 0


def _update_candidates() -> int:
    if not CANDIDATES_PATH.exists():
        print(f"candidates not found at {CANDIDATES_PATH}", file=sys.stderr)
        return 1
    with open(CANDIDATES_PATH) as fh:
        cands = json.load(fh)

    audit_note = (
        "Round-5 defer cleared: Lightcap ingested 2026-06-01 as the inaugural "
        "tool_runtime_trace canonical type (new). operations config (2,262 "
        "rows × 33 cols) -> promoted_for_backtest + "
        "promoted_for_constraint_aware_evaluation + "
        "promoted_for_training_priors. tool_summary config (32 aggregated "
        "rows) -> promoted_for_schema_only. Trust tier: Tier 3 "
        "(tier_3_cluster_scheduler_traces — real measured execution "
        "telemetry, job-trace shape, but the 'jobs' are MCP tool calls, "
        "not GPU jobs). Routing-quality + failure-rate + tail-latency "
        "priors for agent workloads. NO GPU / NO model / NO LLM-serving "
        "signal — closed tool-runtime e2e timing only."
    )

    candidates = cands.get("candidates") or []
    updated = 0
    for c in candidates:
        if c.get("dataset_id") == DATASET_ID:
            c["recommended_action"] = "ingest_now_bounded"
            c["audit_round"] = "focused_audit_2026_06_01c"
            c["audit_decision"] = "ingested_as_tool_runtime_trace"
            c["audit_note_2026_06_01c"] = audit_note
            c["canonical_trace_type"] = "tool_runtime_trace"
            c["trust_level"] = "tier_3_cluster_scheduler_traces"
            updated += 1

    cands["candidates"] = candidates
    cands["last_updated_at_s"] = time.time()
    cands[AUDIT_BLOCK_KEY] = {
        "ran_at_s": time.time(),
        "scope": (
            "Inaugural tool_runtime_trace canonical-type ingest "
            "(Lightcap/agent-runtime-telemetry-small). Clears Round-5 "
            "'defer_high_value_different_trace_class' block by adding the "
            "new canonical type to aurelius/traces/hf_corpus/schemas.py + "
            "promotion.py."
        ),
        "datasets": [DATASET_ID],
        "configs_ingested": [
            f"{DATASET_ID}@operations (2,262 rows, moderate strength)",
            f"{DATASET_ID}@tool_summary (32 aggregated rows, fixture_only "
            "strength → promoted_for_schema_only)",
        ],
        "new_canonical_type": "tool_runtime_trace",
        "trust_tier": "tier_3_cluster_scheduler_traces",
        "license": "cc-by-4.0",
        "headline_promotion_state": "promoted_for_backtest",
        "headline_promotion_tags": [
            "promoted_for_backtest",
            "promoted_for_constraint_aware_evaluation",
            "promoted_for_training_priors",
        ],
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
    }
    with open(CANDIDATES_PATH, "w") as fh:
        json.dump(cands, fh, indent=2, sort_keys=True)
    print(f"Updated {CANDIDATES_PATH} ({updated} entries touched, +1 audit block)")
    return 0


def main() -> int:
    rc = _register_canonical_corpus()
    if rc != 0:
        return rc
    return _update_candidates()


if __name__ == "__main__":
    sys.exit(main())
