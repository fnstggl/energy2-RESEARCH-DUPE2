"""Mooncake trace — FULL_TRACE ingestion (KV plane).

Ingests the complete public Mooncake trace (FAST'25 release, JSONL with
``hash_ids`` = block-level prefix hashes) and computes the real prefix-reuse
distribution that parameterizes the stateful KV cache model — exact-prefix reuse,
partial-prefix overlap, and longest-common-prefix length. Falls back, with an
EXPLICIT status, to the committed sample.

Each record: ``{timestamp, input_length, output_length, hash_ids:[block,...]}``.
Two requests sharing a leading sub-list of ``hash_ids`` shared that KV prefix.
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
from dataclasses import dataclass

from ..data_tier import FULL_TRACE, SAMPLE_FIXTURE, VALIDATION_FIXTURE, SourceStatus

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(_REPO, "data", "external", "mooncake", "raw", "conversation_trace.jsonl")
# Committed, CI-reproducible fixture (the complete public trace, gzipped compact CSV).
# Built by scripts/build_mooncake_fixture.py; tier VALIDATION_FIXTURE (real public data).
FIXTURE_GZ = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_validation.csv.gz")
SAMPLE = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_sample.csv")

DOWNLOAD_HINT = (
    "curl -o data/external/mooncake/raw/conversation_trace.jsonl "
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/"
    "traces/conversation_trace.jsonl")


@dataclass
class MooncakeRequest:
    timestamp: float
    input_length: int
    output_length: int
    hash_ids: list


def _load_jsonl(path: str) -> list:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(MooncakeRequest(
                timestamp=float(d.get("timestamp", 0.0)),
                input_length=int(d.get("input_length", 0)),
                output_length=int(d.get("output_length", 0)),
                hash_ids=[str(b) for b in d.get("hash_ids", [])]))
    return out


def _load_csv_rows(rows) -> list:
    out = []
    for i, r in enumerate(rows):
        out.append(MooncakeRequest(
            timestamp=float(r.get("timestamp_s") or i),
            input_length=int(float(r.get("input_length") or 0)),
            output_length=int(float(r.get("output_length") or 0)),
            hash_ids=(r.get("hash_ids") or "").split()))
    return out


def _load_sample_csv(path: str) -> list:
    import csv
    with open(path, newline="") as fh:
        return _load_csv_rows(csv.DictReader(fh))


def _load_fixture_gz(path: str) -> list:
    """Load the committed gzipped compact-CSV validation fixture."""
    import csv
    with gzip.open(path, "rt", newline="") as fh:
        return _load_csv_rows(csv.DictReader(fh))


def ingest_mooncake() -> tuple:
    """Return ``(requests, SourceStatus)``.

    Source preference (so validation is CI-reproducible without local raw files):
    RAW JSONL → FULL_TRACE; else the committed gz fixture → VALIDATION_FIXTURE (the
    complete public trace, real data); else the 8-row sample → SAMPLE_FIXTURE. Both
    real paths sort by the same key, so the fixture reproduces the RAW reuse stats.
    """
    if os.path.exists(RAW):
        reqs = sorted(_load_jsonl(RAW), key=lambda r: r.timestamp)
        return reqs, SourceStatus(
            source="mooncake", tier=FULL_TRACE, path=RAW, n_records=len(reqs),
            trace_version="Mooncake/FAST25/conversation_trace")
    if os.path.exists(FIXTURE_GZ):
        reqs = sorted(_load_fixture_gz(FIXTURE_GZ), key=lambda r: r.timestamp)
        return reqs, SourceStatus(
            source="mooncake", tier=VALIDATION_FIXTURE, path=FIXTURE_GZ, n_records=len(reqs),
            trace_version="Mooncake/FAST25 (committed validation fixture)")
    reqs = sorted(_load_sample_csv(SAMPLE), key=lambda r: r.timestamp)
    return reqs, SourceStatus(
        source="mooncake", tier=SAMPLE_FIXTURE, path=SAMPLE, n_records=len(reqs),
        trace_version="sample", blocked_reason="full JSONL + validation fixture not present",
        manual_step=DOWNLOAD_HINT)


def reuse_distribution(requests: list) -> dict:
    """Compute the real prefix-reuse distributions from the trace (causal — each
    request only sees blocks from EARLIER requests; never future)."""
    seen: set = set()
    exact_hits = 0          # leading block already cached
    partial_overlaps = []   # fraction of a request's blocks already cached
    lcp_lengths = []        # longest cached leading-prefix length (blocks)
    for r in requests:
        h = r.hash_ids
        if not h:
            continue
        if h[0] in seen:
            exact_hits += 1
        partial_overlaps.append(sum(1 for b in h if b in seen) / len(h))
        lcp = 0
        for b in h:
            if b in seen:
                lcp += 1
            else:
                break
        lcp_lengths.append(lcp)
        seen.update(h)
    n = sum(1 for r in requests if r.hash_ids)
    return {
        "n_requests": n,
        "exact_prefix_hit_rate": round(exact_hits / n, 4) if n else 0.0,
        "mean_partial_overlap": round(statistics.mean(partial_overlaps), 4) if partial_overlaps else 0.0,
        "mean_lcp_blocks": round(statistics.mean(lcp_lengths), 4) if lcp_lengths else 0.0,
        "p95_lcp_blocks": (sorted(lcp_lengths)[int(len(lcp_lengths) * 0.95)] if lcp_lengths else 0),
        "distinct_blocks": len(seen),
        "partial_overlap_samples": partial_overlaps,   # for held-out validation
    }


def split_reuse(requests: list, holdout_frac: float = 0.3) -> tuple:
    """Time-split into (train, holdout) reuse distributions for validation."""
    cut = int(len(requests) * (1.0 - holdout_frac))
    return reuse_distribution(requests[:cut]), reuse_distribution(requests[cut:])


__all__ = ["MooncakeRequest", "ingest_mooncake", "reuse_distribution",
           "split_reuse", "DOWNLOAD_HINT"]
