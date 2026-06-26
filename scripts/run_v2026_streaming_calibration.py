#!/usr/bin/env python3
"""Run v2026 incremental FULL_TRACE_EXACT calibration by streaming the OSS archives.

Streams one v2026 fact table partition-by-partition (bounded disk, resumable) and
writes its exact calibration artifact. Re-run to resume from the last checkpoint;
a completed table is labeled FULL_TRACE_EXACT, a partial run SUBSET_TRACE (still
exact over the partitions processed).

Env:
  V2026_WORK_DIR       bounded temp dir for one partition at a time (default /tmp/v2026_work)
  V2026_PROCESSED_DIR  where artifacts + manifests are written
                       (default data/external/alibaba_gpu_v2026/processed)

Usage:
  python -m scripts.run_v2026_streaming_calibration network_hourly
  python -m scripts.run_v2026_streaming_calibration pod_hourly --max-partitions 50
  # resume: just run the same command again.
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.ingestion.v2026_calibration import calibrate_table

DEFAULT_WORK = os.environ.get("V2026_WORK_DIR", "/tmp/v2026_work")
DEFAULT_PROCESSED = os.environ.get(
    "V2026_PROCESSED_DIR", "data/external/alibaba_gpu_v2026/processed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["pod_hourly", "server_hourly", "network_hourly",
                                       "job_execution_summary"])
    ap.add_argument("--max-partitions", type=int, default=None,
                    help="cap partitions this run (resumable; omit for full FULL_TRACE_EXACT)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("V2026_WORKERS", "8")),
                    help="parallel partition fetchers for range-streamed archives (pod_hourly)")
    ap.add_argument("--work-dir", default=DEFAULT_WORK)
    ap.add_argument("--processed-dir", default=DEFAULT_PROCESSED)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    manifest = os.path.join(args.processed_dir, f"{args.table}_manifest.json")
    artifact = os.path.join(args.processed_dir, f"{args.table}_calibration.json")

    print(f"streaming v2026 {args.table} (work={args.work_dir}, "
          f"workers={args.workers}, resumable) ...")
    res = calibrate_table(
        args.table, work_dir=args.work_dir, manifest_path=manifest,
        max_partitions=args.max_partitions, workers=args.workers)
    d = res.to_dict()
    with open(artifact, "w") as f:
        json.dump(d, f, indent=2)

    print(f"  partitions {d['n_partitions_done']}/{d['n_partitions_total']} | "
          f"label={d['label']} | streamed={d['bytes_streamed'] / 1e9:.2f} GB")
    print(f"  artifact: {artifact}")
    if not d["complete"]:
        print("  (partial — re-run to resume from the checkpoint toward FULL_TRACE_EXACT)")


if __name__ == "__main__":
    main()
