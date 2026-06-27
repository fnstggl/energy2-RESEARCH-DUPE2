# Mooncake validation fixture — representativeness report (auto-generated)

Fixture: `tests/fixtures/mooncake/mooncake_validation.csv.gz` (0.77 MB gz), 12,031 requests (full trace). Tier: **VALIDATION_FIXTURE** (real public data; RAW download is FULL_TRACE).

Generated deterministically: `python -m scripts.build_mooncake_fixture` (gzip mtime=0 → byte-reproducible). Reuse structure forbids row-sampling, so the fixture is the trace in original order.

## KV distributions preserved (fixture vs full public trace)

| statistic | full trace | fixture |
|---|---|---|
| n_requests | 12031 | 12031 |
| exact_prefix_hit_rate | 0.9999 | 0.9999 |
| mean_partial_overlap | 0.3843 | 0.3843 |
| mean_lcp_blocks | 8.7865 | 8.7865 |
| p95_lcp_blocks | 43 | 43 |
| mean_blocks | 23.98 | 23.98 |
| mean_input_length | 12035.1 | 12035.1 |
| mean_output_length | 342.6 | 342.6 |
| distinct_blocks | 182790 | 182790 |

## Distribution-distance vs full trace (same metrics as ValidationSuite)

| metric | value | tolerance |
|---|---|---|
| KS (partial-overlap) | 0.0 | ≤ 0.05 |
| Wasserstein-1 (partial-overlap) | 0.0 | ≤ 0.02 |
| hist-L1 (blocks/req) | 0.0 | ≤ 0.05 |
| hist-L1 (input length) | 0.0 | ≤ 0.05 |
| KS (output length) | 0.0 | ≤ 0.05 |

The fixture is the complete public trace → distances are ~0 (identity). A contiguous-prefix fixture (`--max-records`) stays within the tolerances above.
