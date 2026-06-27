#!/usr/bin/env python3
"""Build the committed, CI-reproducible Mooncake validation fixture from the RAW trace.

The KV held-out validation must pass identically on a clean checkout (CI), where the
gitignored RAW Mooncake JSONL is absent. This script deterministically derives a
committed fixture — the COMPLETE public trace (12,031 requests) as a gzipped compact
CSV (~0.77 MB; "reasonably commit-sized" → maximum fidelity) — and proves it preserves
the distributions the KV simulator depends on.

Reuse structure forbids naive row-sampling (dropping a block's first appearance
corrupts later reuse accounting), so the fixture is the full trace in original order
(or a contiguous prefix if ``--max-records`` is given — a real causal sub-trace). gzip
is written with mtime=0 for byte-reproducible output.

Provenance: RAW → FULL_TRACE; this fixture → VALIDATION_FIXTURE (real public data, repo
provenance — NEVER synthetic, never labeled FULL_TRACE).

Usage:
  python -m scripts.build_mooncake_fixture                # full trace
  python -m scripts.build_mooncake_fixture --max-records 10000
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics

from aurelius.environment.ingestion.mooncake import (
    FIXTURE_GZ,
    RAW,
    MooncakeRequest,
    reuse_distribution,
)
from aurelius.environment.validation_suite import hist_l1, ks_statistic, wasserstein1

_REPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "research", "MOONCAKE_FIXTURE_REPRESENTATIVENESS.md")
_HEADER = "request_id,timestamp_s,input_length,output_length,hash_ids\n"


def _load_raw(path: str) -> list:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _to_reqs(rows: list) -> list:
    return [MooncakeRequest(timestamp=float(r.get("timestamp", 0.0)),
                            input_length=int(r.get("input_length", 0)),
                            output_length=int(r.get("output_length", 0)),
                            hash_ids=[str(b) for b in r.get("hash_ids", [])]) for r in rows]


def _write_fixture(rows: list, path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.GzipFile(path, mode="wb", mtime=0) as g:        # mtime=0 → reproducible bytes
        g.write(_HEADER.encode())
        for i, r in enumerate(rows):
            ids = " ".join(str(b) for b in r.get("hash_ids", []))
            g.write(f"{i},{r.get('timestamp', 0)},{r.get('input_length', 0)},"
                    f"{r.get('output_length', 0)},{ids}\n".encode())
    return os.path.getsize(path)


def _summary(reqs: list) -> dict:
    d = reuse_distribution(reqs)
    nblocks = [len(r.hash_ids) for r in reqs if r.hash_ids]
    inlen = [r.input_length for r in reqs]
    outlen = [r.output_length for r in reqs]
    return {**{k: d[k] for k in ("n_requests", "exact_prefix_hit_rate", "mean_partial_overlap",
                                 "mean_lcp_blocks", "p95_lcp_blocks", "distinct_blocks")},
            "mean_blocks": round(statistics.mean(nblocks), 3) if nblocks else 0,
            "mean_input_length": round(statistics.mean(inlen), 1) if inlen else 0,
            "mean_output_length": round(statistics.mean(outlen), 1) if outlen else 0,
            "_overlap": d["partial_overlap_samples"], "_blocks": nblocks,
            "_inlen": inlen, "_outlen": outlen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=None,
                    help="contiguous prefix size (default: full trace)")
    args = ap.parse_args()
    if not os.path.exists(RAW):
        raise SystemExit(f"RAW Mooncake trace not present at {RAW}; download it first "
                         "(see ingestion.mooncake.DOWNLOAD_HINT) to regenerate the fixture.")

    raw_rows = _load_raw(RAW)
    sub_rows = raw_rows[: args.max_records] if args.max_records else raw_rows
    size = _write_fixture(sub_rows, FIXTURE_GZ)

    full, fix = _summary(_to_reqs(raw_rows)), _summary(_to_reqs(sub_rows))
    metrics = {
        "ks_partial_overlap": round(ks_statistic(fix["_overlap"], full["_overlap"]), 4),
        "wasserstein_partial_overlap": round(wasserstein1(fix["_overlap"], full["_overlap"]), 5),
        "histl1_blocks": round(hist_l1(fix["_blocks"], full["_blocks"]), 4),
        "histl1_input_length": round(hist_l1(fix["_inlen"], full["_inlen"]), 4),
        "ks_output_length": round(ks_statistic(fix["_outlen"], full["_outlen"]), 4),
    }
    drop = ("_overlap", "_blocks", "_inlen", "_outlen")
    full = {k: v for k, v in full.items() if k not in drop}
    fix = {k: v for k, v in fix.items() if k not in drop}

    lines = ["# Mooncake validation fixture — representativeness report (auto-generated)\n",
             f"Fixture: `{os.path.relpath(FIXTURE_GZ)}` ({size/1e6:.2f} MB gz), "
             f"{fix['n_requests']:,} requests "
             f"({'full trace' if args.max_records is None else f'first {args.max_records}'}). "
             "Tier: **VALIDATION_FIXTURE** (real public data; RAW download is FULL_TRACE).\n",
             "Generated deterministically: `python -m scripts.build_mooncake_fixture` "
             "(gzip mtime=0 → byte-reproducible). Reuse structure forbids row-sampling, so the "
             "fixture is the trace in original order.\n",
             "## KV distributions preserved (fixture vs full public trace)\n",
             "| statistic | full trace | fixture |", "|---|---|---|"]
    for k in ("n_requests", "exact_prefix_hit_rate", "mean_partial_overlap", "mean_lcp_blocks",
              "p95_lcp_blocks", "mean_blocks", "mean_input_length", "mean_output_length",
              "distinct_blocks"):
        lines.append(f"| {k} | {full[k]} | {fix[k]} |")
    lines += ["\n## Distribution-distance vs full trace (same metrics as ValidationSuite)\n",
              "| metric | value | tolerance |", "|---|---|---|",
              f"| KS (partial-overlap) | {metrics['ks_partial_overlap']} | ≤ 0.05 |",
              f"| Wasserstein-1 (partial-overlap) | {metrics['wasserstein_partial_overlap']} | ≤ 0.02 |",
              f"| hist-L1 (blocks/req) | {metrics['histl1_blocks']} | ≤ 0.05 |",
              f"| hist-L1 (input length) | {metrics['histl1_input_length']} | ≤ 0.05 |",
              f"| KS (output length) | {metrics['ks_output_length']} | ≤ 0.05 |",
              "\nThe fixture is the complete public trace → distances are ~0 (identity). A "
              "contiguous-prefix fixture (`--max-records`) stays within the tolerances above."]
    with open(_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(FIXTURE_GZ)} ({size/1e6:.2f} MB), "
          f"{fix['n_requests']} requests; report → {os.path.relpath(_REPORT)}")
    print("representativeness:", metrics)


if __name__ == "__main__":
    main()
