#!/usr/bin/env python3
"""Generate research/V2026_FULL_TRACE_ARTIFACT_STATUS.md from the live artifacts.

Reads the committed calibration artifacts + manifests and emits the honest
per-table status (partitions completed/total, rows, fidelity label, resume
command). Run after a streaming pass to refresh the status doc.
"""

from __future__ import annotations

import json
import os

PROCESSED = os.environ.get(
    "V2026_PROCESSED_DIR", "data/external/alibaba_gpu_v2026/processed")
TABLES = ("pod_hourly", "server_hourly", "network_hourly", "job_execution_summary")
DOC = "research/V2026_FULL_TRACE_ARTIFACT_STATUS.md"


def _rows(art: dict) -> int:
    # best row proxy: the largest aggregator n across categories
    best = 0
    for a in (art.get("artifacts") or {}).values():
        if isinstance(a, dict) and "n" in a:
            best = max(best, a["n"])
    return best


def main() -> None:
    lines = ["# v2026 FULL_TRACE artifact status — auto-generated\n",
             "Streamed incrementally from Aliyun OSS with bounded disk "
             "(`aurelius/environment/ingestion/v2026_stream.py`). FULL_TRACE_EXACT "
             "= every partition processed; SUBSET_TRACE = resumable partial run "
             "(exact over the partitions processed). Raw data is never committed.\n",
             "| table | partitions | rows | streamed | label | categories | artifact |",
             "|---|---|---|---|---|---|---|"]
    for t in TABLES:
        p = os.path.join(PROCESSED, f"{t}_calibration.json")
        if not os.path.exists(p):
            lines.append(f"| {t} | — | — | — | (not run) | — | — |")
            continue
        a = json.load(open(p))
        cats = len(a.get("artifacts") or {})
        lines.append(
            f"| {t} | {a['n_partitions_done']}/{a['n_partitions_total']} | "
            f"{_rows(a):,} | {a['bytes_streamed'] / 1e9:.2f} GB | **{a['label']}** | "
            f"{cats} | `{os.path.relpath(p)}` |")

    lines += [
        "\n## Resume / complete commands\n",
        "```bash",
        "export V2026_PROCESSED_DIR=data/external/alibaba_gpu_v2026/processed",
        "# small archives: whole-download then local stream (fast)",
        "python -m scripts.run_v2026_streaming_calibration network_hourly",
        "python -m scripts.run_v2026_streaming_calibration job_execution_summary",
        "python -m scripts.run_v2026_streaming_calibration server_hourly",
        "# pod_hourly (351 GB): range-streamed, resumable — re-run to continue",
        "python -m scripts.run_v2026_streaming_calibration pod_hourly  # full FULL_TRACE_EXACT",
        "python -m scripts.run_v2026_streaming_calibration pod_hourly --max-partitions 200  # a batch",
        "```",
        "\n## Notes",
        "- Percentiles are FULL_TRACE_APPROX (fixed-bin histograms over every row; "
        "bins documented per category). Exact stats (count/sum/mean/variance/min/"
        "max/category mixes) are FULL_TRACE_EXACT.",
        "- pod_hourly is the only table that must be range-streamed (351 GB > 10 GB "
        "prefetch cap); a full pass transfers 351 GB once (time-bound, resumable).",
    ]
    with open(DOC, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {DOC}")
    print("\n".join(lines[2:4 + len(TABLES)]))


if __name__ == "__main__":
    main()
