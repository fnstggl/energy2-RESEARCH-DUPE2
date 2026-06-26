# v2026 FULL_TRACE artifact status — auto-generated

Streamed incrementally from Aliyun OSS with bounded disk (`aurelius/environment/ingestion/v2026_stream.py`). FULL_TRACE_EXACT = every partition processed; SUBSET_TRACE = resumable partial run (exact over the partitions processed). Raw data is never committed.

| table | partitions | rows | streamed | label | categories | artifact |
|---|---|---|---|---|---|---|
| pod_hourly | — | — | — | (not run) | — | — |
| server_hourly | — | — | — | (not run) | — | — |
| network_hourly | 168/168 | 5,390,218 | 0.20 GB | **FULL_TRACE_EXACT** | 4 | `data/external/alibaba_gpu_v2026/processed/network_hourly_calibration.json` |
| job_execution_summary | 1/1 | 40,522,321 | 1.19 GB | **FULL_TRACE_EXACT** | 12 | `data/external/alibaba_gpu_v2026/processed/job_execution_summary_calibration.json` |

## Resume / complete commands

```bash
export V2026_PROCESSED_DIR=data/external/alibaba_gpu_v2026/processed
# small archives: whole-download then local stream (fast)
python -m scripts.run_v2026_streaming_calibration network_hourly
python -m scripts.run_v2026_streaming_calibration job_execution_summary
python -m scripts.run_v2026_streaming_calibration server_hourly
# pod_hourly (351 GB): range-streamed, resumable — re-run to continue
python -m scripts.run_v2026_streaming_calibration pod_hourly  # full FULL_TRACE_EXACT
python -m scripts.run_v2026_streaming_calibration pod_hourly --max-partitions 200  # a batch
```

## Notes
- Percentiles are FULL_TRACE_APPROX (fixed-bin histograms over every row; bins documented per category). Exact stats (count/sum/mean/variance/min/max/category mixes) are FULL_TRACE_EXACT.
- pod_hourly is the only table that must be range-streamed (351 GB > 10 GB prefetch cap); a full pass transfers 351 GB once (time-bound, resumable).
