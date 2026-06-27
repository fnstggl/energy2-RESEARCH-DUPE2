"""Azure LLM trace — FULL_TRACE ingestion (serving plane).

Ingests the public Azure LLM Inference trace (Azure/AzurePublicDataset). Prefers the
**2024 one-week** trace (``AzureLLMInferenceTrace_{conv,code}_1week.csv`` — 168 h,
~44 M requests, the longest public sample) over the 2023 one-hour Splitwise files, then
the committed sample (explicit ``SAMPLE_FIXTURE``, never a silent downgrade). The audit
that established the one-week files exist but were unwired is
``research/AZURE_TRACE_COVERAGE_AUDIT.md``.

Two entry points:

- ``ingest_azure(limit)`` — a relative-time serving-request **list** of
  ``(arrival_s, context_tokens, output_tokens)``. The one-week trace has ~44 M rows,
  far too many to hold as a list, so an unlimited call is capped (``DEFAULT_FULL_CAP``)
  and the status says so; pass an explicit ``limit`` for a bounded serving slice (every
  canonical-env caller does).
- ``hourly_arrival_frames(...)`` — a **bounded-memory streaming binner** that turns the
  full week into per-hour records (exact arrival counts + a stride-sampled token subset)
  *without* materialising the trace. This is what makes the 168 clean hourly periods
  usable for hourly forecasting/evaluation.

The conv week (May 12–18) and code week (May 10–16) are offset by two days, so naively
merging them would inject artificial arrival-rate steps at the availability boundaries.
``hourly_arrival_frames`` therefore bins a **single clean service** (conv preferred) to
keep the arrival-rate series honest; ``ingest_azure`` likewise draws its bounded slice
from the primary (conv) service first.

Columns: ``TIMESTAMP`` (wall-clock), ``ContextTokens`` (prompt), ``GeneratedTokens``
(output). Service time / goodput derive from output tokens; context feeds the KV model.
"""

from __future__ import annotations

import csv
import os
import statistics
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

# When the (huge) one-week trace feeds the request-LIST API with no explicit limit,
# cap it so callers never accidentally materialise ~44 M rows. `hourly_arrival_frames`
# is the bounded-memory entry point for the full week.
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
    """``(path, trace_version, tier)`` for the single clean service binned into hourly
    frames (conv preferred — the larger, cleaner diurnal series). ``None`` path = sample."""
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
                            "hourly_arrival_frames for the full week" if capped else ""))
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


def _pctl(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


def _bin_rows(rows, *, bin_seconds: float, sample_stride: int, cycle_len: int) -> list:
    """Bin a ``(ts, ctx, gen)`` stream into per-bin frames — bounded memory.

    Pure over the iterable (no file I/O), so it is unit-testable on synthetic rows. The
    stream is assumed timestamp-sorted (the Azure files are): the first row sets ``t0``.
    Keeps a ~``n_bins``-key exact count dict (every row) + a stride sample of tokens.
    """
    counts: dict = {}
    samples: dict = {}                 # bin -> list of (ts_rel, output, context) from the stride sample
    t0 = None
    gidx = 0
    for ts, ctx, gen in rows:
        if t0 is None:
            t0 = ts
        b = int((ts - t0) // bin_seconds)
        counts[b] = counts.get(b, 0) + 1
        if gidx % sample_stride == 0:
            samples.setdefault(b, []).append((ts - t0, gen, ctx))
        gidx += 1
    frames = []
    for b in sorted(counts):
        recs = samples.get(b, [])      # already in arrival order (file is sorted)
        out = [r[1] for r in recs]
        inp = [r[2] for r in recs]
        gaps = [recs[i + 1][0] - recs[i][0] for i in range(len(recs) - 1)]
        mg = statistics.mean(gaps) if gaps else 0.0
        cv = (statistics.pstdev(gaps) / mg) if (gaps and mg > 0) else 0.0
        frames.append({
            "index": b, "hour_of_day": b % cycle_len, "n_requests": counts[b],
            "arrival_rate_per_s": counts[b] / bin_seconds,
            "output_token_mean": (statistics.mean(out) if out else 0.0),
            "output_token_p95": _pctl(out, 0.95),
            "input_token_mean": (statistics.mean(inp) if inp else 0.0),
            "interarrival_cv": cv, "n_sampled": len(recs),
        })
    return frames


def hourly_arrival_frames(*, bin_seconds: float = 3600.0, sample_stride: int = 100,
                          cycle_len: int = 24) -> dict | None:
    """Stream the primary Azure service once → bounded-memory per-bin arrival frames.

    Single pass over a single clean service (conv preferred); returns ``None`` when no
    full trace is present (caller falls back to the sample). The file is timestamp-sorted
    so the first row is ``t0`` — no second pass needed.

    Each frame: ``index`` (bin), ``hour_of_day`` (= index % cycle_len; the 2024 week
    starts 00:00 UTC so this is the true UTC hour), ``n_requests`` (exact),
    ``arrival_rate_per_s``, ``output_token_mean/p95``, ``input_token_mean``,
    ``interarrival_cv`` (from the sample)."""
    path, version, tier = _primary_file()
    if path is None:
        return None
    frames = _bin_rows(_iter_rows(path), bin_seconds=bin_seconds,
                       sample_stride=sample_stride, cycle_len=cycle_len)
    if not frames:
        return None
    return {"frames": frames, "trace_version": version, "tier": tier,
            "n_bins": len(frames), "total_requests": sum(f["n_requests"] for f in frames),
            "sample_stride": sample_stride, "bin_seconds": bin_seconds, "service": "conv"}


def to_serving_raw(requests: list) -> list:
    return [(arr, gen) for (arr, _ctx, gen) in requests]


def context_tokens(requests: list) -> list:
    return [ctx for (_a, ctx, _g) in requests]


__all__ = ["ingest_azure", "hourly_arrival_frames", "to_serving_raw", "context_tokens",
           "FULL_1WEEK_CONV", "FULL_1WEEK_CODE", "FULL_CONV", "FULL_CODE", "DOWNLOAD_HINT"]
