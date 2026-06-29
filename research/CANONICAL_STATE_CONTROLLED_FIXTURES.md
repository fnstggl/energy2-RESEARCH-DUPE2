# Canonical State — Controlled Fixtures (Phase 11)

The properties this PR claims, each proven by an executable fixture (`tests/test_canonical_state.py`) or the
checkpointed runner. Every fixture is deterministic (seed-0, no RNG). PASS = the assertion holds.

| # | property | fixture | proof |
|--|--|--|--|
| 1 | **request lifecycle persists** across periods (no request is lost) | `test_request_lifecycle_persists_and_conserves` | 8 + 4 ingested → `arrived == 12`; all retained |
| 2 | **queue evolution** is consolidated from RequestState | `test_queue_summary_consolidation` | `queue_summary()` backlog/class-mix/completion-rate derived from RequestState |
| 3 | **queue pressure affects SLA** (missed requests counted) | `test_request_lifecycle_persists_and_conserves` | `missed_sla` tracks the realised SLA-safe fraction |
| 4 | **forecast error is measured causally** (only after realization) | `test_forecast_error_only_after_realization` | error is `None` before `record_realized`; correct after |
| 5 | **placement persists and releases** | `RequestState.placement` + `test_placement_ref_validation_catches_dangling` | placement recorded per request; dangling refs caught |
| 6 | **roofline state changes with GPU/model/batch** | `test_roofline_record_from_diag` | `RooflineRecord.from_diag` captures regime/precision/timing |
| 7 | **no request disappears** | `test_no_request_disappears_and_none_double_counted` | `arrived == running + completed + dropped`; `len(requests)==arrived` |
| 8 | **no queued request is also completed** | `test_completed_request_not_in_backlog` | completed → empty backlog; `validate_no_completed_in_queue` PASS |
| 9 | **no placed request references a nonexistent replica** | `test_placement_ref_validation_catches_dangling` | `validate_placement_refs` FAILs on a ghost replica, PASSes on a real one |
| 10 | **all-knobs runner checkpoints and resumes** | `scripts/run_checkpointed_all_knobs_backtest.py` | artifact written after every cell; `--resume` skips COMPLETED/TIMEOUT/SKIPPED_TOO_HEAVY |

## Extra honesty checks (in `state_validation.py`)

- **forecast no-future-leakage** (`test_forecast_made_after_target_is_flagged`): a belief whose `made_at_period`
  is after its `target_period` is FLAGGED FAIL — the planner cannot believe about the past.
- **forecast error correctness** (`test_forecast_error_summary_mae_mape`): MAE/MAPE computed only from
  belief − realized; no fabricated values.
- **clone isolation** (`test_clone_isolation_request_and_forecast`): mutating a deepcopy never touches the
  original — the MPC-search isolation guarantee, now extended to the new states.
- **legacy preserved**: the new states are opt-in (`getattr`-guarded controller hook; `forecast_state=None`
  default in `run_period_episode`); with them unattached the reward/cost/SLA path is byte-identical (the 36
  controller/world/electricity/mpc-training tests pass unchanged).

## Reproduce

```
python -m pytest tests/test_canonical_state.py -q          # 11 fixtures
python -m scripts.run_checkpointed_all_knobs_backtest --quick   # checkpoint/resume + state capture
```
