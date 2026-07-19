# Aurelius Replay — Self-Serve Local Trace Analyzer (POC)

> **Status: proof of concept.** Read-only tooling only. No optimizer,
> simulator, policy, baseline, or economics change — the replay engines and
> KPI are imported from the locked evaluation stack and never re-implemented
> or tuned here. Every rendered report carries the `docs/RESULTS.md` §8
> allowed wording and is scanned for the forbidden claim substrings before
> it is written. **Directional only — not production savings.**

## 1. What this is

A local-first, self-serve version of the pilot guide's **offline replay**
phase (`enterprisedocs/pilot-guide.md`): a prospect points the tool at a
historical scheduler / serving log export and gets

1. a **schema mapping plan** — how their columns map onto the canonical
   trace contracts (`aurelius/traces/schema.py`),
2. a **readiness verdict** — `ready | degraded | insufficient`, with named
   levers on/off and what unlocks each,
3. a **descriptive analysis** measured purely from their own log,
4. a **counterfactual replay** of the same history through the locked
   harnesses — `aurelius/traces/backtest.py` (LLM serving) or
   `aurelius/traces/gpu_scheduling.py` (GPU jobs) — scored on the canonical
   KPI, and
5. a single-file HTML report + an **anonymized schema fingerprint** they can
   choose to share.

Everything runs on their machine. The tool makes no network calls and
transmits nothing; the fingerprint is the only artifact designed to be
shared, and sharing it is a manual user action.

```
python -m aurelius.selfserve demo                 # zero-input first run
python -m aurelius.selfserve inspect sacct.csv    # mapping plan only
python -m aurelius.selfserve run sacct.csv --fleet-gpus 512 --gpu-type h100
python -m aurelius.selfserve serve                # local web UI on 127.0.0.1
```

## 2. Why local-first (the distribution decision)

The strategic goal is to cut the trust required to *experience* Aurelius
from "send a startup your scheduler logs" to "run a read-only binary and
look at a report." The failure mode that matters — *self-serve user feeds
it a schema we never saw, it breaks or (worse) prints a wrong number* — is
a property of the **importer**, not of where the compute runs. A hosted
version would break on the same schemas; it would just add a data-custody
ask on top. So the architecture is:

- **local-first execution** (trust story: nothing leaves the machine;
  matches the security posture already promised in
  `enterprisedocs/pilot-guide.md` and the site's evaluation narrative);
- **fingerprint-mediated learning loop** (calibration story: the schema
  fingerprint carries column names, shapes, coverage, and parse failures —
  enough to write a mapping for a new export format — without log rows or
  identifier values).

What would have required seeing their data becomes: they run it → readiness
report names what didn't map → they send the fingerprint → we ship a
synonym-table/adapter update → their next run works. Every fingerprint
widens `aurelius/selfserve/fields.py`; the importer is the compounding
asset.

## 3. Never-break contract

The tool has three terminal states and no fourth:

| state | when | output |
|---|---|---|
| replay ran | required fields mapped, checks pass | full report (tier `ready`/`degraded`, every assumption labelled) |
| replay refused | required fields missing, too few rows, span too short, mapping suspect | diagnosis + fingerprint — **never a number the data can't support** |
| file unreadable | binary/empty/headerless | a plain-language error naming supported formats |

Mechanisms: strict preflight validation (`validate.py`) before any engine
call; per-row error ledger instead of exceptions (`extract.py`); explicit
unit inference with recorded evidence (epoch s/ms/us, ISO-8601,
`[D-]HH:MM:SS`, `_ms`/`_hours` name hints — never silent); fuzzy column
matches labelled for confirmation; a demo mode whose sample logs are messy
on purpose so the failure paths are exercised on every demo.

## 4. Honesty posture (inherited, enforced)

- Headline = `constraint_aware` vs the **strongest realistic safe
  baseline** chosen by the locked classifiers (`sla_aware` family for
  serving, packing baselines for jobs). FIFO is rendered as a sanity floor
  and marked as such.
- Aurelius-side variants (`safe_high_utilization`, `min_cost_safe`) are
  labelled *aurelius variant* and never the quoted result; a safety column
  applies the documented gates (serving timeout ≤ 10 %; job starvation).
- Levers the data cannot evidence are OFF and say so (no session key → zero
  simulated cache benefit — mirroring `aurelius/traces/replay.py`).
- Cost basis is the public default from `aurelius/benchmarks/economics.py`
  and the report says so; the customer's procurement rate is a pilot-stage
  input (`docs/RESULTS.md` §8 gate).
- Inferred fleet (peak concurrent GPU demand) is labelled an ASSUMPTION
  with the flag to replace it.
- `report.write_report` raises `ClaimRuleViolation` on any forbidden
  phrase; `tests/test_selfserve_pipeline.py` covers it.
- Demo output is synthetic and never quotable — demo numbers exist to show
  the report shape, not to be cited.

## 5. What it does NOT do (deliberately)

- **No per-customer optimizer fitting.** Self-serve maps data *into* the
  canonical contracts; it does not tune constants, priors, or gates against
  the customer's workload. Calibration against live telemetry is exactly
  what the shadow-pilot phase is for — the report's closing section says
  so. A self-serve flow that silently "fits" to arbitrary data is how a
  confidently wrong number ships.
- **No uploads, no telemetry, no phone-home.** Not even opt-out analytics.
- **No energy-arbitrage replay yet** (System A, `python -m aurelius
  backtest`): it needs region/price context that a cold prospect won't
  have on first touch. Roadmap, behind the same validator.

## 6. Rollout stages (recommended)

1. **Concierge (now, first 5–10 prospects):** send the repo/wheel; they run
   `demo`, then `run` on a real export while we're on a call. We never see
   logs; we see the fingerprint. Every session extends the synonym table.
2. **Gated download:** email-gated artifact so we know who ran it and can
   follow up when a fingerprint shows unmapped fields. Importer hardened by
   stage-1 formats (Slurm `sacct` ✓, K8s job dumps, Run:ai, vLLM/gateway
   logs ✓, Kueue/Volcano).
3. **Public self-serve:** the funnel the site's evaluation section already
   describes — download → replay → report → shadow-pilot conversation.
   Prerequisite: IP decision on shipping policy logic in a public artifact
   (compiled wheel vs. open-core vs. keeping variants server-side).

## 7. Files

| path | role |
|---|---|
| `aurelius/selfserve/fields.py` | canonical fields, synonym tables, unit coercion (the compounding asset) |
| `aurelius/selfserve/sniff.py` | format sniffing (CSV/TSV/JSONL, gzip) + mapping plan |
| `aurelius/selfserve/extract.py` | streaming row extraction + error ledger |
| `aurelius/selfserve/validate.py` | readiness tiers, checks, lever availability |
| `aurelius/selfserve/analyze.py` | descriptive stats measured from the log |
| `aurelius/selfserve/pipeline.py` | orchestration; bridges to the locked replay engines |
| `aurelius/selfserve/report.py` | single-file HTML report + claim-rule guard |
| `aurelius/selfserve/fingerprint.py` | anonymized shareable diagnostics |
| `aurelius/selfserve/demo.py` | deterministic messy sample generators |
| `aurelius/selfserve/cli.py` | `inspect / run / demo / serve` |
| `aurelius/selfserve/webapp.py` | localhost web UI (FastAPI, no external assets) |
| `tests/test_selfserve_importer.py` | mapping, coercion, never-crash contract |
| `tests/test_selfserve_pipeline.py` | end-to-end, claim rules, fingerprint privacy |

## 8. Known limitations (POC)

- Serving replay requires per-request token counts (by design — no invented
  token distributions). Gateways that don't log tokens need a vLLM/router
  export; the readiness report says exactly that.
- Jobs replay prices the fleet with default rates and (absent `--fleet-*`)
  an inferred fleet; both are labelled assumptions in the report.
- Pure-Python row parsing: ~0.5–1M rows in seconds-to-tens-of-seconds;
  larger exports should be windowed (`--max-rows` caps with a warning).
- Web UI is single-user local; no auth (binds 127.0.0.1 by default).
- The marketing site's headline claim predates the benchmark rollup's
  claim tiers; before this tool is put in front of prospects, site copy and
  report numbers must tell the same story
  (`docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md` §10 has the approved
  tiers).
