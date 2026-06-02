#!/usr/bin/env python3
"""Refresh the gate-derived metadata on every committed Lightcap JSON.

In-place maintenance helper that updates the four already-committed
``data/external/hf/Lightcap__agent-runtime-telemetry-small/<config>/
processed/summary.json`` files and the rollup
``data/external/hf_discovery/lightcap_runtime_telemetry_ingest_summary.json``
so they carry the new redistribution-gate fields the ninth gate-consumer
wiring of ``scripts/ingest_hf_lightcap_runtime_telemetry.py`` introduces.

The script avoids re-downloading the Lightcap parquet files (raw is
gitignored under ``data/external/hf/Lightcap__.../raw/``) — the gate
decision is a pure function of the recorded license tag, so the
verdict is computed from the existing committed JSONs alone.

Pure-Python; no third-party deps; no HF API call; no HF_TOKEN read.

Invariants this script preserves:

* Every other field in summary.json is byte-for-byte unchanged.
* Field ordering is preserved by writing with ``sort_keys=True`` (the
  v1 writer used the same convention).
* The fixture files on disk are NOT touched.
* The audit summary's ``configs`` row order matches the per-config
  ingest order embedded in ``TARGETS`` in the main ingest script.

Run once after wiring the gate; the next live ingest writes the same
fields directly through the ninth-consumer code path.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"

LIGHTCAP_ROOT = HF_DIR / "Lightcap__agent-runtime-telemetry-small"


def _load_ingest_module():
    spec = importlib.util.spec_from_file_location(
        "ingest_hf_lightcap_runtime_telemetry_refresh",
        REPO_ROOT / "scripts" / "ingest_hf_lightcap_runtime_telemetry.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_hf_lightcap_runtime_telemetry_refresh"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ingest = _load_ingest_module()
    ledger = ingest._load_ledger()

    updated_configs: list[dict] = []
    for target in ingest.TARGETS:
        config = target["config_name"]
        processed = LIGHTCAP_ROOT / config / "processed"
        summary_path = processed / "summary.json"
        if not summary_path.exists():
            print(f"skip {config}: no summary.json on disk")
            continue
        s = json.loads(summary_path.read_text())

        decision = ingest.evaluate_redistribution(
            ledger=ledger,
            license_tag=ingest.LICENSE_TAG,
            dataset_id=ingest.DATASET_ID,
        )

        # Preserve the existing license value (it was already
        # ``"cc-by-4.0"`` literal in the v1 writer); the new fields are
        # additive.
        s["license"] = ingest.LICENSE_TAG
        s["license_redistribution_status"] = decision.license_status
        s["license_redistribution_source"] = ingest.LICENSE_SOURCE
        s["redistribution_gate_reason_code"] = decision.reason_code
        s["redistribution_gate_reason_detail"] = decision.reason_detail
        s["redistribution_gate_permitted"] = decision.permitted
        s["redistribution_gate_operator_grant_dataset_id"] = (
            decision.operator_grant_dataset_id
        )
        s["redistribution_gate_scope"] = ingest.GATE_SCOPE

        summary_path.write_text(
            json.dumps(s, indent=2, default=str, sort_keys=True) + "\n"
        )

        updated_configs.append({
            "config_name": config,
            "license": s["license"],
            "license_redistribution_status": decision.license_status,
            "redistribution_gate_reason_code": decision.reason_code,
            "redistribution_gate_permitted": decision.permitted,
            "redistribution_gate_operator_grant_dataset_id":
                decision.operator_grant_dataset_id,
            "canonical_trace_type": s.get("canonical_trace_type"),
            "available_signals": s.get("available_signals"),
            "missing_signals": s.get("missing_signals"),
            "analysis_sample_rows": s.get("analysis_sample_rows"),
            "statistical_sample_strength": s.get(
                "statistical_sample_strength"
            ),
        })

    audit_path = (
        DISC_DIR / "lightcap_runtime_telemetry_ingest_summary.json"
    )
    prev = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    # Merge: for every config we updated, REPLACE the existing
    # configs row in-place; preserve the rest of the v1 row keys (like
    # ``decision_state`` / ``decision_tags`` / ``manifest``) so the
    # refresh stays additive.
    prev_configs = prev.get("configs", [])
    new_by_key = {e["config_name"]: e for e in updated_configs}
    merged_configs: list[dict] = []
    for row in prev_configs:
        key = row.get("config")  # v1 used "config", we add "config_name"
        if key in new_by_key:
            updated = dict(row)
            updated.update(new_by_key[key])
            merged_configs.append(updated)
            del new_by_key[key]
        else:
            merged_configs.append(row)
    # Append any new rows (none expected; all four configs already exist).
    for k in list(new_by_key.keys()):
        merged_configs.append(new_by_key[k])

    payload = {
        "doc_version": (
            "lightcap_runtime_telemetry_ingest_summary_v2"
        ),
        "dataset_id": ingest.DATASET_ID,
        "wrote_at_s": time.time(),
        "git_sha": _git_sha(),
        "redistribution_gate_scope": ingest.GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "configs": merged_configs,
    }
    audit_path.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )
    print(f"wrote {audit_path} ({len(merged_configs)} config rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
