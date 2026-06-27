"""Azure LLM trace — FULL_TRACE ingestion (serving plane).

Ingests the public Azure LLM Inference trace (Azure/AzurePublicDataset). Prefers the
**2024 one-week** trace (``AzureLLMInferenceTrace_{conv,code}_1week.csv`` — 168 h,
~44 M requests, the longest public sample) over the 2023 one-hour Splitwise files, then
the committed sample (explicit ``SAMPLE_FIXTURE``, never a silent downgrade). The audit
that found the one-week files exist but were unwired is
``research/AZURE_TRACE_COVERAGE_AUDIT.md``.

Two entry points:

- ``ingest_azure(limit)`` — a relative-time serving-request **list** of
  ``(arrival_s, context_tokens, output_tokens)``. The one-week trace has ~44 M rows,
  far too many to hold as a list, so an unlimited call is capped (``DEFAULT_FULL_CAP``)
  and the status says so; pass an explicit ``limit`` for a bounded serving slice.
- ``azure_period_frames(...)`` — a **bounded-memory streaming binner** for the full week.
  One pass over a single clean service (conv): it keeps the EXACT per-bin arrival count
  (every row) and a deterministic **stride sample** of the requests per bin. The sample is
  proportional to the true count, so the diurnal SHAPE is preserved; building period frames
  from the sample keeps the forecast arrival-rate and the controller's replayed load at the
  SAME (1/stride) scale — internally consistent and fair (every policy sees the same load),
  while never materialising the 44 M-row trace.

The conv week (May 12–18) and code week (May 10–16) are offset two days, so a naive union
would inject artificial arrival-rate steps at the availability boundaries; the binner uses
the larger, cleaner conv service, and ``ingest_azure`` draws its bounded slice from conv first.

Columns: ``TIMESTAMP`` (wall-clock), ``ContextTokens`` (prompt), ``GeneratedTokens``
(output). Service time / goodput derive from output tokens; context feeds the KV model.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from ..data_tier import FULL_TRACE, SAMPLE_FIXTURE, SourceStatus

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(_REPO, "data", "external", "azure_llm_2024", "raw")
# 2024 one-week trace (preferred — the longest public Azure LLM sample).
FULL_1WEEK_CONV = os.path.join(RAW_DIR, "AzureLLMInferenceTrace_conv_1week.csv")
FULL_1WEEK_CODE = os.path.join(RAW_DIR, "AzureLLMInferenceTrace_code_1week.csv")
# 2023 one-hour Splitwise trace (fallback).
FULL_CONV = os.path.join(RAW_DIR, "AzureLLMInferenceTrace_conv.csv")
FULL_CODE = os.path.join(RAW_DIR, "AzureLLMInferenceTrace_code.csv")
SAMPLE_FIXTURE_PATH = os.path.join(_REPO, "tests", "fixtures", "azure_llm_2024_sample.csv")

# When the (huge) one-week trace feeds the request-LIST API with no explicit limit, cap it
# so callers never materialise ~44 M rows. `azure_period_frames` is the bounded entry point.
DEFAULT_FULL_CAP = 200_000

DOWNLOAD_HINT = (
    "curl -L -o data/external/azure_llm_2024/raw/AzureLLMInferenceTrace_conv_1week.csv "
    "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
    "AzureLLMInferenceTrace_conv_1week.csv  (and _code_1week.csv); or the 2023 one-hour "
    "AzureLLMInferenceTrace_conv.csv / _code.csv from .../master/data/")


def _parse_ts(s: str) -> float | None:
    """Parse a TIMESTAMP into an epoch float. Tolerates the 2024 tz-aware form
    (``...+00:00``) and >6-digit fractional seconds; only deltas are ever used, so the
    naive/UTC interpretation is irrelevant (every row in a file is treated alike)."""
    s = (s or "").strip()
    if not s:
        return None
    if "+" in s:                       # drop a trailing tz offset ("...+00:00")
        s = s.split("+", 1)[0]
    if "." in s:                       # trim sub-microsecond digits for fromisoformat
        head, frac = s.split(".", 1)
        s = f"{head}.{frac[:6]}"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _active_files() -> tuple:
    """``(files, trace_version, tier)`` for the serving-request list — prefer the 1-week
    pair (conv first), then the 1-hour pair, then none (sample)."""
    if os.path.exists(FULL_1WEEK_CONV):
        files = [FULL_1WEEK_CONV] + ([FULL_1WEEK_CODE] if os.path.exists(FULL_1WEEK_CODE) else [])
        return files, "AzureLLMInferenceDataset2024/1week", FULL_TRACE
    if os.path.exists(FULL_CONV):
        files = [FULL_CONV] + ([FULL_CODE] if os.path.exists(FULL_CODE) else [])
        return files, "AzureLLMInferenceTrace2023/1hour", FULL_TRACE
    return [], "", SAMPLE_FIXTURE


def _primary_file() -> tuple:
    """``(path, trace_version, tier)`` for the single clean service binned into periods
    (conv preferred — the larger, cleaner diurnal series). ``None`` path = sample only."""
    if os.path.exists(FULL_1WEEK_CONV):
        return FULL_1WEEK_CONV, "AzureLLMInferenceDataset2024/1week/conv", FULL_TRACE
    if os.path.exists(FULL_CONV):
        return FULL_CONV, "AzureLLMInferenceTrace2023/1hour/conv", FULL_TRACE
    return None, "", SAMPLE_FIXTURE


def _iter_rows(path: str):
    """Yield ``(ts, ctx, gen)`` for valid rows, streaming (bounded memory)."""
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)                # header
        for row in rdr:
            if len(row) < 3:
                continue
            ts = _parse_ts(row[0])
            if ts is None:
                continue
            try:
                ctx, gen = int(float(row[1])), int(float(row[2]))
            except ValueError:
                continue
            if gen > 0:
                yield ts, ctx, gen


def ingest_azure(*, limit: int | None = None) -> tuple:
    """Return ``(requests, SourceStatus)`` with ``(arrival_s, context_tokens, output_tokens)``
    relative to the first arrival. Streams with an early stop so the one-week trace never
    materialises in full; an unlimited one-week call is capped at ``DEFAULT_FULL_CAP``."""
    files, version, tier = _active_files()
    if files:
        cap = limit or (DEFAULT_FULL_CAP if "1week" in version else None)
        rows: list = []
        for f in files:                # gather enough from the primary service to sort + slice
            for rec in _iter_rows(f):
                rows.append(rec)
                if cap and len(rows) >= cap * 2:
                    break
            if cap and len(rows) >= cap * 2:
                break
        rows.sort(key=lambda r: r[0])
        if cap:
            rows = rows[:cap]
        t0 = rows[0][0] if rows else 0.0
        reqs = [(t - t0, ctx, gen) for (t, ctx, gen) in rows]
        capped = (not limit) and ("1week" in version)
        return reqs, SourceStatus(
            source="azure_llm", tier=tier, path=RAW_DIR, n_records=len(reqs),
            trace_version=version,
            blocked_reason=("request-list capped at DEFAULT_FULL_CAP; use "
                            "azure_period_frames for the full week" if capped else ""))
    reqs = []
    with open(SAMPLE_FIXTURE_PATH, newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        for i, row in enumerate(rdr):
            if limit and len(reqs) >= limit:    # honour limit on the sample path too
                break
            try:
                reqs.append((float(i), 0, int(float(row[-1]))))
            except (ValueError, IndexError):
                continue
    return reqs, SourceStatus(
        source="azure_llm", tier=SAMPLE_FIXTURE, path=SAMPLE_FIXTURE_PATH,
        n_records=len(reqs), trace_version="sample",
        blocked_reason="no 1-week or 1-hour CSVs present", manual_step=DOWNLOAD_HINT)


def _bin_stream(rows, *, bin_seconds: float, sample_stride: int) -> tuple:
    """Bin a ``(ts, ctx, gen)`` stream → ``(per_period, exact_counts)`` (bounded memory).

    Pure over the iterable (no file I/O), so it is unit-testable on synthetic rows. The
    stream is assumed timestamp-sorted (the Azure files are): the first row sets ``t0``.
    ``exact_counts[b]`` counts every row; ``per_period[b]`` keeps every ``sample_stride``-th
    request as ``(arrival_s, out_tok, in_tok)`` relative to the trace start."""
    per_period: dict = {}
    exact: dict = {}
    t0 = None
    gidx = 0
    for ts, ctx, gen in rows:
        if t0 is None:
            t0 = ts
        b = int((ts - t0) // bin_seconds)
        exact[b] = exact.get(b, 0) + 1
        if gidx % sample_stride == 0:
            per_period.setdefault(b, []).append((ts - t0, gen, ctx))
        gidx += 1
    return per_period, exact


def azure_period_frames(*, bin_seconds: float = 3600.0, sample_stride: int = 24) -> dict | None:
    """Stream the primary Azure service once → bounded-memory per-period inputs.

    Returns ``None`` when no full trace is present (caller falls back to the per-minute
    sample path). Otherwise a dict with:

    - ``per_period``: ``{bin_index: [(arrival_s, out_tok, in_tok), ...]}`` — every
      ``sample_stride``-th request, arrival relative to the trace start. This is the
      load the forecaster's frames and the controller's replay BOTH consume, so they
      share one (1/stride) scale. The sample is proportional to the true per-bin count,
      so the diurnal shape is preserved.
    - ``exact_counts``: ``{bin_index: true_count}`` — every row counted (for reporting /
      the coverage claim), independent of the sample.
    - ``trace_version`` / ``tier`` / ``n_bins`` / ``total_requests`` (exact) /
      ``sample_stride`` / ``bin_seconds`` / ``service``.

    The file is timestamp-sorted, so the first row sets ``t0``; a single pass suffices.
    """
    path, version, tier = _primary_file()
    if path is None:
        return None
    per_period, exact = _bin_stream(_iter_rows(path), bin_seconds=bin_seconds,
                                    sample_stride=sample_stride)
    if not exact:
        return None
    return {"per_period": per_period, "exact_counts": exact, "trace_version": version,
            "tier": tier, "n_bins": len(exact), "total_requests": sum(exact.values()),
            "sample_stride": sample_stride, "bin_seconds": bin_seconds, "service": "conv"}


def to_serving_raw(requests: list) -> list:
    """Project ``(arrival_s, ctx, gen)`` → the ``(arrival_s, output_tokens)`` the serving
    plane consumes (output tokens drive service time + goodput)."""
    return [(arr, gen) for (arr, _ctx, gen) in requests]


def context_tokens(requests: list) -> list:
    return [ctx for (_a, ctx, _g) in requests]


__all__ = ["ingest_azure", "azure_period_frames", "to_serving_raw", "context_tokens",
           "FULL_1WEEK_CONV", "FULL_1WEEK_CODE", "FULL_CONV", "FULL_CODE", "DOWNLOAD_HINT"]
