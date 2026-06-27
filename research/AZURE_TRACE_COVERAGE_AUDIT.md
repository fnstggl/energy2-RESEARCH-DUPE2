# Azure Trace Coverage Audit

**Verdict: B — INGESTION GAP (not a dataset limitation).** The ~1-hour span the
canonical environment ingested was the **2023 one-hour Splitwise sample**; the full
available public Azure LLM inference trace is **one week (168 hours, ~44 M requests)**
and was simply not wired. This audit verifies that from the raw files and corrects the
ingestion.

Every number below was re-verified by streaming the actual files in this environment
(`scripts`-free check, one pass per file), **not** taken on faith from any prior note.

## 1. Source + files ingested *before* this fix

`ingest_azure()` (`aurelius/environment/ingestion/azure.py`) loaded, from
`data/external/azure_llm_2024/raw/` (gitignored):

| file | rows (verified) | span | dates | dataset |
|---|---|---|---|---|
| AzureLLMInferenceTrace_conv.csv | 19,366 | 0.97 h | 2023-11-16 18:15–19:14 | AzureLLMInferenceDataset**2023** (Splitwise, ISCA'24) |
| AzureLLMInferenceTrace_code.csv | 8,819 | 0.95 h | 2023-11-16 18:17–19:14 | same |

Both cover the **same** 2023-11-16 production hour, so merging them is still **~1 hour**
of wall-clock — it does not extend duration. That one hour was the ceiling on any
hourly analysis.

## 2. Files available in the public source

[Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) publishes
**three** LLM/LMM inference datasets:

| dataset | files | span | status |
|---|---|---|---|
| **2023** | `AzureLLMInferenceTrace_{conv,code}.csv` | ~1 hour (Nov 2023) | **was wired** (the gap) |
| **2024** | `AzureLLMInferenceTrace_{conv,code}_1week.csv` | **1 week** (May 2024) | **available, was NOT wired** ← the gap |
| 2025 | LMM multimodal (images + tokens) | 1 week (Oct 2024) | out of scope (no token-only serving schema) |

The 2024 one-week files are GitHub-release assets under
`.../releases/download/dataset-llm-2024/AzureLLMInferenceTrace_{conv,code}_1week.csv`.

## 3. Coverage comparison (downloaded + stream-verified, this audit)

| file | bytes | rows (verified) | first TS | last TS | span | sorted? |
|---|---|---|---|---|---|---|
| conv_1week.csv | 1.13 GB | **27,303,999** | 2024-05-12 00:00:00Z | 2024-05-18 23:59:59Z | **168.00 h / 7.00 d** | ascending ✓ |
| code_1week.csv | 692 MB | **16,803,695** | 2024-05-10 00:00:00Z | 2024-05-16 23:59:59Z | **168.00 h / 7.00 d** | ascending ✓ |

Same schema as 2023 (`TIMESTAMP, ContextTokens, GeneratedTokens`). Each file is a clean,
gap-free, ascending 7-day week. **The two weeks are offset by two days** (code 05-10→05-16,
conv 05-12→05-18), overlapping ~120 h.

## 4. Additional days/hours not wired?

Yes — the entire 2024 one-week trace (168 h **per service**). That is ~168× the wired
2023 hour and is more than enough for held-out **hourly** forecasting and evaluation
(e.g. 5 train days / 1 val day / 1 eval day).

## 5. Decision + what this PR changes (item 5: more data exists)

`ingest_azure()` now **prefers the one-week trace** (`FULL_TRACE`,
`trace_version = AzureLLMInferenceDataset2024/1week`), falling back to the 2023 one-hour
files, then the committed sample — never a silent downgrade; the tier is reported.

The 168-hour series is exposed to consumers via a new **bounded-memory** binner,
`hourly_arrival_frames()`: one streaming pass over a **single clean service**, holding
only a ~168-key exact-count dict plus a stride sample of tokens — it never materialises
the 27 M-row list. Verified on the real trace: **168 hourly bins, 27,303,999 requests
counted exactly, a real diurnal signal** (peak hour-of-day ≈ 14 at ~59 req/s, trough
≈ 22 at ~33 req/s).

**Why a single service, not the merged 44 M:** the conv and code weeks are offset by two
days, so a naive union would inject *artificial* arrival-rate steps at the
data-availability boundaries (code-only May 10–12, both May 12–16, conv-only May 16–18) —
a forecaster would learn an artefact, not real demand. The binner therefore uses the
larger, cleaner **conv** week (full 7 diurnal cycles). The request-LIST API
(`ingest_azure`) likewise draws its bounded slice from the primary (conv) service.
A future consumer that wants true *combined* load can sum the two services over their
~120 h overlap; that option is left explicit rather than silently merged.

Memory: the raw multi-GB CSVs stay gitignored; only this small audit + the derived
hourly frames (if a downstream step persists them) are committed.

## 6. Scope boundary — what this PR does *not* claim

This is an **ingestion/coverage** fix. The repository does **not** currently contain a
committed Azure forecasting model or model-predictive controller, so there are **no
forecasting artifacts to "regenerate"** here, and none are fabricated. What changes is
that the data layer now serves **168 clean hourly periods instead of one** — so when a
forecasting/evaluation layer is built on top, it trains and validates on a week of real
diurnal structure, not a single unrepresentative hour. The forecasting/controller
architecture is intentionally left untouched (no such code exists to change).

## 7. Hourly-validity gate

With 168 contiguous hourly periods now loadable in bounded memory, hourly Azure serving
analysis is **no longer degenerate** (non-trivial train/val/eval hourly splits exist).
The prior ~1-hour ceiling was an artefact of the ingestion gap, now corrected. Until a
forecaster is actually fit and shown to beat its naive baseline on a held-out hourly
split, no hourly forecasting *accuracy* claim is made — only that the data now supports
one.

Sources: [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset),
[AzureLLMInferenceDataset2024.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md).
Verified counts/spans produced by a one-pass stream of each raw file in this environment.
