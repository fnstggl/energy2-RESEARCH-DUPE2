#!/usr/bin/env python3
"""Run the full canonical-environment ValidationSuite and print the report.

Calibrates the canonical multi-plane environment (Azure serving + Mooncake KV +
Alibaba v2026 fleet + ISO electricity), then validates every plane's distribution
against its held-out reference, printing PASS / WARN / FAIL / SKIPPED + the metrics
and the honesty-gate verdict. With the FULL_TRACE_EXACT v2026 artifacts present
(``--processed-dir``), the fleet distributions anchor to the real 6.5 B-row
marginals; without them the env falls back to the committed sample.

Usage:
  python -m scripts.run_canonical_validation
  python -m scripts.run_canonical_validation --json
  V2026_PROCESSED_DIR=... python -m scripts.run_canonical_validation
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.canonical import CanonicalMultiPlaneEnvironment
from aurelius.environment.ingestion.azure import ingest_azure, to_serving_raw

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MOONCAKE = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_sample.csv")
_DEFAULT_PROCESSED = os.environ.get(
    "V2026_PROCESSED_DIR", os.path.join(_REPO, "data", "external", "alibaba_gpu_v2026", "processed"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=_DEFAULT_PROCESSED,
                    help="v2026 FULL_TRACE_EXACT artifact dir (anchors fleet marginals)")
    ap.add_argument("--mooncake", default=_DEFAULT_MOONCAKE)
    ap.add_argument("--limit", type=int, default=5880, help="cap Azure requests")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    reqs, status = ingest_azure(limit=args.limit)
    azure_raw = to_serving_raw(reqs)
    env = CanonicalMultiPlaneEnvironment(
        mooncake_path=args.mooncake, processed_dir=args.processed_dir)
    env.calibrate(azure_raw)
    report = env.validate().to_dict()
    manifest = env.manifest().to_dict()

    if args.json:
        print(json.dumps({
            "azure_tier": status.tier, "n_azure_requests": len(azure_raw),
            "validation": report,
            "manifest_is_production_grade": manifest["is_production_grade"],
            "headline_safe_signals": manifest["headline_safe_signals"],
        }, indent=2))
        return

    print(f"overall: {report['overall_verdict']}   counts: {report['counts']}")
    print(f"azure source: {status.tier} ({len(azure_raw)} requests)\n")
    for c in report["checks"]:
        print(f"  {c['verdict']:7} {c['kind']:34} {c['metric_name']}={c['metric']}  [{c['ref_tier']}]")
        if c["verdict"] == "SKIPPED":
            print(f"          ↳ {c['detail']}")
    print(f"\nhonesty gate — production_grade: {manifest['is_production_grade']} "
          f"(never True while any signal < TRACE_DERIVED or the ABSENT proprietary tier is unfilled)")
    print(f"headline-safe signals: {len(manifest['headline_safe_signals'])}")


if __name__ == "__main__":
    main()
