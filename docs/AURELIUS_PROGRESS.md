# Aurelius Progress Tracker

## Current Status
- Phase: PHASE_3 → PHASE_4 ready
- Milestone: Phase 3 — Production-Like Simulation Environment
- Status: MERGED

## Last Run
- Date: 2026-04-25
- Branch: claude/bold-dirac-z8f0z
- PR URL: https://github.com/fnstggl/energy2/pull/8
- PR Status: MERGED (squash)
- Merge Status: MERGED
- Main Commit SHA: 2c093db8bdc260711e367b0327f28a2eb65aa431

## Tests
- Unit: 370 passed, 0 failed (full suite)
- Phase 3 new tests: 60 (test_phase3_workloads.py)
- Pre-existing tests: 310 (all still pass)
- Result: ALL PASSING

## Phase 3 Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| load_workload_csv() → list[Job] with GPU/SLA/PUE fields | DONE | aurelius/ingestion/workload_traces.py |
| Job model: workload_type, gpu_type, gpu_count, sla_penalty_per_hour, data_transfer_gb, pue | DONE | aurelius/models.py |
| Job model: interruptible, preemptible, checkpointable, sla_class, allowed_regions, forbidden_regions | DONE | aurelius/models.py |
| Data residency enforced in Job.__post_init__ | DONE | allowed_regions narrows region_options, forbidden_regions removed |
| Existing tests still pass with defaults | DONE | 310 pre-existing tests pass |
| Objective: sla_penalty_cost, data_transfer_cost, pue_factor | DONE | aurelius/optimization/objective.py |
| Integration test: high-penalty job not scheduled past deadline | DONE | test_phase3_workloads.py |
| WorkloadSimulator.generate() — statistically distinct per type | DONE | aurelius/simulation/workload_simulator.py |
| 7 workload types × 8 GPU types, seeded reproducibility | DONE | _PROFILES + _GPU_BASE_POWER_KW |
| ShadowRunner.run() — decisions vs realized prices | DONE | aurelius/execution/shadow_runner.py |
| Costs from real_prices/real_carbon only (not forecasts) | DONE | tested + verified adversarially |
| JSONL persistence of shadow results | DONE | ShadowRunner.output_path |
| docker/Dockerfile: multi-stage, Python 3.11, non-root | DONE | docker/Dockerfile |
| docker/docker-compose.yml: api + postgres + redis | DONE | docker/docker-compose.yml |
| .github/workflows/ci.yml: ruff + mypy + pytest + Docker | DONE | .github/workflows/ci.yml |

## What Was Completed This Run

### New Files
- aurelius/ingestion/workload_traces.py — load_workload_csv() with full validation
- aurelius/simulation/workload_simulator.py — WorkloadSimulator.generate() + generate_mixed()
- aurelius/execution/shadow_runner.py — ShadowRunner + ShadowResult + ShadowDecisionRecord
- docker/Dockerfile — multi-stage Python 3.11, non-root user
- docker/docker-compose.yml — aurelius-api + postgres:15 + redis:7
- .github/workflows/ci.yml — ruff + mypy + pytest + docker-build jobs
- tests/test_phase3_workloads.py — 60 Phase 3 tests

### Modified Files
- aurelius/models.py — Extended Job (14 new fields) and OptimizationConfig (6 new fields)
- aurelius/optimization/objective.py — PUE, SLA penalty, data transfer cost terms
- .env.example — Added Phase 3 env vars (AURELIUS_API_KEY, REDIS_URL, ML_ARTIFACTS_DIR)

## Adversarial Review Findings and Fixes

| Issue Found | Fix Applied |
|-------------|-------------|
| Test used floor(earliest_start) which could be < earliest_start | Fixed test to use ceil(earliest_start) |
| ruff: 10 import/unused-var issues in new files | Auto-fixed with ruff --fix |
| PUE must scale both energy cost AND carbon (not just cost) | Verified in objective.py + adversarial test |
| Shadow runner must not use forecast data for cost computation | Verified: real_prices/real_carbon only |
| WorkloadSimulator: deadline must always be >= earliest_start + runtime | Tested across all 7 types × 100 jobs |

## Known Risks for Phase 4

- Docker build not verified locally (no Docker daemon) — CI will verify on GitHub Actions
- sla_class stored in Job but safety gate SLA thresholds not yet automatically loaded from workload_type (requires Phase 4 wiring)
- Shadow runner uses default price when real_prices dict is missing hour keys (graceful, logged)
- No HTML report generator yet (Phase 4)
- No API auth middleware yet (Phase 4)
- No GET /simulations or GET /simulations/{id} endpoints yet (Phase 4)
- No confidence intervals on savings metrics yet (Phase 4)

## What Remains for Phase 3
NONE — all Phase 3 acceptance criteria are met.

## Phase 4 Next Steps
Phase 4: Reporting and Pilot Readiness requires:
1. aurelius/reporting/savings_report.py — SavingsReport.generate(backtest_result) with 95% bootstrap CIs
2. aurelius/reporting/html_report.py — render_html_report() → self-contained HTML with embedded charts
3. aurelius/api/app.py — Implement GET /simulations + GET /simulations/{run_id} + API key auth middleware
4. aurelius/validation/leakage_audit.py — assert_no_leakage() raising DataLeakageError on overlap

All reports must include:
- Cost savings with 95% CI (bootstrap)
- Carbon reduction in tonnes
- Methodology section proving leakage-free computation
- Independent reproducibility instructions

## Next Task
Start Phase 4 sprint. Exact scope:
- Implement SavingsReport.generate() with bootstrap confidence intervals
- Implement HTML report generation (Jinja2 + matplotlib embedded as base64)
- Add GET /simulations + GET /simulations/{run_id} to API
- Add API key auth middleware (AURELIUS_API_KEY env var)
- Add DataLeakageError and assert_no_leakage() to validation
- Write unit + integration + E2E tests for all reporting
