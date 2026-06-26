"""Azure LLM trace — FULL_TRACE ingestion (serving plane).

Ingests the complete public Azure LLM Inference trace (``AzureLLMInferenceTrace_
conv.csv`` + ``_code.csv`` from Azure/AzurePublicDataset) into relative-time
serving requests, computes a time-ordered train/holdout split, and (separately)
emits a deterministic fixture for tests. Falls back, with an EXPLICIT
``SAMPLE_FIXTURE`` status (never silent), to the committed sample when the full
CSVs are not present.

Columns: ``TIMESTAMP`` (wall-clock), ``ContextTokens`` (prompt), ``GeneratedTokens``
(output). Service time / goodput derive from output tokens; context tokens feed
the KV/prefill model.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from ..data_tier import FULL_TRACE, SAMPLE_FIXTURE, SourceStatus

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(_REPO, "data", "external", "azure_llm_2024", "raw")
FULL_CONV = os.path.join(RAW_DIR, "AzureLLMInferenceTrace_conv.csv")
FULL_CODE = os.path.join(RAW_DIR, "AzureLLMInferenceTrace_code.csv")
SAMPLE_FIXTURE_PATH = os.path.join(_REPO, "tests", "fixtures", "azure_llm_2024_sample.csv")

DOWNLOAD_HINT = (
    "curl -o data/external/azure_llm_2024/raw/AzureLLMInferenceTrace_conv.csv "
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"
    "AzureLLMInferenceTrace_conv.csv (and _code.csv)")


def _parse_ts(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    # "2023-11-16 18:15:46.6805900" — trim sub-microsecond digits for fromisoformat
    if "." in s:
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _load_csv(path: str) -> list:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            ts = _parse_ts(r.get("TIMESTAMP"))
            try:
                ctx = int(float(r.get("ContextTokens") or 0))
                gen = int(float(r.get("GeneratedTokens") or 0))
            except ValueError:
                continue
            if ts is not None and gen > 0:
                out.append((ts, ctx, gen))
    return out


def ingest_azure(*, limit: int | None = None) -> tuple:
    """Return ``(requests, SourceStatus)`` where requests are
    ``(arrival_s, context_tokens, output_tokens)`` relative to the first arrival.

    FULL_TRACE if the raw conv/code CSVs are present (merged + time-sorted), else
    an explicit SAMPLE_FIXTURE fall-back (not a silent downgrade).
    """
    if os.path.exists(FULL_CONV):
        rows = _load_csv(FULL_CONV)
        if os.path.exists(FULL_CODE):
            rows += _load_csv(FULL_CODE)
        rows.sort(key=lambda r: r[0])
        t0 = rows[0][0]
        reqs = [(t - t0, ctx, gen) for (t, ctx, gen) in rows]
        if limit:
            reqs = reqs[:limit]
        return reqs, SourceStatus(
            source="azure_llm", tier=FULL_TRACE, path=RAW_DIR, n_records=len(reqs),
            trace_version="AzurePublicDataset/AzureLLMInferenceTrace")
    # explicit sample fall-back
    reqs = []
    with open(SAMPLE_FIXTURE_PATH, newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        for i, row in enumerate(rdr):
            try:
                reqs.append((float(i), 0, int(float(row[-1]))))
            except (ValueError, IndexError):
                continue
    return reqs, SourceStatus(
        source="azure_llm", tier=SAMPLE_FIXTURE, path=SAMPLE_FIXTURE_PATH,
        n_records=len(reqs), trace_version="sample",
        blocked_reason="full conv/code CSVs not downloaded",
        manual_step=DOWNLOAD_HINT)


def to_serving_raw(requests: list) -> list:
    """Project ``(arrival_s, ctx, gen)`` → the ``(arrival_s, output_tokens)`` the
    serving plane consumes (output tokens drive service time + goodput)."""
    return [(arr, gen) for (arr, _ctx, gen) in requests]


def context_tokens(requests: list) -> list:
    return [ctx for (_a, ctx, _g) in requests]


__all__ = ["ingest_azure", "to_serving_raw", "context_tokens",
           "FULL_CONV", "FULL_CODE", "DOWNLOAD_HINT"]
