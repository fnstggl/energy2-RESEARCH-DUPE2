"""Preflight validation: can this mapped data support a replay, and how well?

Produces a :class:`ReadinessReport` with a three-tier verdict, in the spirit
of the linkage-quality grading in ``docs/PILOT_TELEMETRY_CONTRACT.md`` §4:

- ``ready``        — required fields mapped and clean; replay runs with the
                     full applicable lever set.
- ``degraded``     — replay runs, but named levers are disabled or named
                     assumptions substituted; the report labels every one.
- ``insufficient`` — replay is refused. The output is the diagnosis + the
                     fingerprint, never a number computed on data that cannot
                     support it. (A wrong number costs more trust than no
                     number.)

Lever availability follows the no-invention rule from the trace harnesses:
e.g. without a session/prefix signal the cache-affinity lever is OFF — the
replay never simulates a cache benefit the data cannot evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extract import ExtractionResult
from .fields import GPU_JOBS, SERVING
from .sniff import MappingPlan

MIN_USABLE_ROWS = 100
MIN_SPAN_SERVING_S = 15 * 60          # 15 minutes of arrivals
MIN_SPAN_JOBS_S = 6 * 3600            # 6 hours of submissions
PARSE_RATE_WARN = 0.99
PARSE_RATE_FAIL = 0.90

TIER_READY = "ready"
TIER_DEGRADED = "degraded"
TIER_INSUFFICIENT = "insufficient"


@dataclass
class Check:
    check_id: str
    level: str          # pass | warn | fail
    message: str

    def to_dict(self) -> dict:
        return {"check_id": self.check_id, "level": self.level,
                "message": self.message}


@dataclass
class Lever:
    name: str
    available: bool
    reason: str
    unlock: str = ""    # what field/export would enable it

    def to_dict(self) -> dict:
        return {"name": self.name, "available": self.available,
                "reason": self.reason, "unlock": self.unlock}


@dataclass
class ReadinessReport:
    tier: str
    workload_class: str
    checks: list[Check] = field(default_factory=list)
    levers: list[Lever] = field(default_factory=list)
    field_coverage: dict = field(default_factory=dict)
    rows_read: int = 0
    rows_usable: int = 0
    refusal_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "workload_class": self.workload_class,
            "rows_read": self.rows_read,
            "rows_usable": self.rows_usable,
            "checks": [c.to_dict() for c in self.checks],
            "levers": [lv.to_dict() for lv in self.levers],
            "field_coverage": self.field_coverage,
            "refusal_reasons": self.refusal_reasons,
        }


def _span_s(records: list[dict], key: str) -> float:
    vals = [r[key] for r in records if r.get(key) is not None]
    if len(vals) < 2:
        return 0.0
    return max(vals) - min(vals)


def validate(plan: MappingPlan, ext: ExtractionResult) -> ReadinessReport:
    cls = ext.workload_class
    checks: list[Check] = []
    levers: list[Lever] = []
    refusals: list[str] = []

    def check(check_id: str, ok: bool, msg_ok: str, msg_bad: str,
              *, fail: bool = False) -> None:
        if ok:
            checks.append(Check(check_id, "pass", msg_ok))
        else:
            level = "fail" if fail else "warn"
            checks.append(Check(check_id, level, msg_bad))
            if fail:
                refusals.append(msg_bad)

    # ------------------------------------------------------------------ class
    if cls not in (SERVING, GPU_JOBS):
        checks.append(Check(
            "workload_class", "fail",
            "could not classify the log as LLM-serving requests or GPU jobs: "
            + plan.class_evidence,
        ))
        return ReadinessReport(
            tier=TIER_INSUFFICIENT, workload_class=cls, checks=checks,
            rows_read=ext.rows_read, rows_usable=ext.rows_usable,
            field_coverage=ext.coverage(),
            refusal_reasons=[
                "workload class unknown — map the required fields explicitly "
                "(--class and --map) or share the schema fingerprint for help",
            ],
        )

    cov = ext.coverage()

    # ------------------------------------------------------------ row volume
    check(
        "usable_rows", ext.rows_usable >= MIN_USABLE_ROWS,
        f"{ext.rows_usable:,} usable rows of {ext.rows_read:,} read",
        f"only {ext.rows_usable:,} usable rows of {ext.rows_read:,} read "
        f"(minimum {MIN_USABLE_ROWS}) — see the parse-error ledger",
        fail=True,
    )

    # ------------------------------------------------------------ parse rate
    if ext.rows_read:
        rate = ext.rows_usable / ext.rows_read
        if rate < PARSE_RATE_FAIL:
            check("parse_rate", False, "",
                  f"only {rate:.0%} of rows were usable — the mapping is "
                  "probably wrong for this export; review the mapping plan",
                  fail=True)
        elif rate < PARSE_RATE_WARN:
            check("parse_rate", False, "",
                  f"{rate:.1%} of rows usable; dropped rows are itemized in "
                  "the parse-error ledger")
        else:
            check("parse_rate", True,
                  f"{rate:.1%} of rows parsed cleanly", "")

    # ---------------------------------------------------------------- span
    span_key = "timestamp" if cls == SERVING else "submit_time"
    span = _span_s(ext.records, span_key)
    min_span = MIN_SPAN_SERVING_S if cls == SERVING else MIN_SPAN_JOBS_S
    hours = span / 3600.0
    check(
        "time_span", span >= min_span,
        f"log spans {hours:,.1f} hours",
        f"log spans only {hours:,.2f} hours — too short for a meaningful "
        f"replay (minimum {min_span / 3600:.1f}h)",
        fail=True,
    )

    # ------------------------------------------------------- class-specific
    if cls == SERVING:
        both_tokens = min(cov.get("prompt_tokens", 0.0),
                          cov.get("output_tokens", 0.0))
        check(
            "token_fields", both_tokens >= 0.9,
            "prompt and output token counts present",
            "token counts are partial "
            f"(prompt {cov.get('prompt_tokens', 0):.0%}, "
            f"output {cov.get('output_tokens', 0):.0%}) — the serving replay "
            "needs per-request token counts; it will not substitute invented "
            "token distributions",
            fail=both_tokens < 0.5,
        )
        neg = sum(1 for r in ext.records
                  if (r.get("prompt_tokens") or 0) < 0
                  or (r.get("output_tokens") or 0) < 0)
        check("token_sanity", neg == 0,
              "no negative token counts",
              f"{neg} rows have negative token counts (dropped from replay)")

        has_session = cov.get("session_id", 0.0) >= 0.3
        levers.append(Lever(
            "cache_affinity", has_session,
            "session/conversation key present — prefix-locality proxy enabled"
            if has_session else
            "no session/conversation key — the replay simulates ZERO cache "
            "benefit rather than inventing one",
            unlock="export a session_id / conversation_id column",
        ))
        has_model = cov.get("model", 0.0) >= 0.5
        levers.append(Lever(
            "model_mix_routing", has_model,
            "per-request model labels present" if has_model else
            "no model column — replay treats traffic as a single pooled model",
            unlock="export the model/deployment name per request",
        ))
        has_status = cov.get("status", 0.0) >= 0.5
        levers.append(Lever(
            "failure_accounting", has_status,
            "status column present — failures excluded from goodput"
            if has_status else
            "no status column — all requests scored as successful "
            "(goodput may be overstated)",
            unlock="export a status / http_status column",
        ))
        if ext.status_values_unrecognized:
            top = ", ".join(f"{k} ({v})" for k, v in
                            list(ext.status_values_unrecognized.items())[:5])
            checks.append(Check(
                "status_values", "warn",
                f"unrecognized status values treated as success: {top} — "
                "share the fingerprint (or map them) to classify",
            ))

    else:  # GPU_JOBS
        gpu_vals = [r["gpu_count"] for r in ext.records if r.get("gpu_count")]
        big = sum(1 for g in gpu_vals if g > 4096)
        check("gpu_count_sanity", big == 0,
              "GPU counts in a plausible range",
              f"{big} jobs request >4096 GPUs — check the gpu_count mapping")
        negdur = sum(1 for r in ext.records
                     if r.get("duration") is not None and r["duration"] <= 0)
        check("duration_sanity", negdur == 0,
              "all durations positive",
              f"{negdur} rows have non-positive durations (dropped)")
        skew = sum(
            1 for r in ext.records
            if r.get("start_time") is not None
            and r.get("submit_time") is not None
            and r["start_time"] < r["submit_time"]
        )
        check("clock_order", skew == 0,
              "start times are never before submit times",
              f"{skew} jobs start before they were submitted — clock skew or "
              "timezone mismatch between columns; queue-wait stats are "
              "unreliable for those rows")

        has_wait = cov.get("queue_wait", 0.0) >= 0.5
        levers.append(Lever(
            "measured_queue_wait", has_wait,
            "measured queue waits present — as-is waits are your real ones"
            if has_wait else
            "no start_time / queue_wait — as-is queue behavior cannot be "
            "shown from your data (replay still compares policies on the "
            "same arrivals)",
            unlock="export job start times (or an explicit wait column)",
        ))
        has_type = cov.get("gpu_type", 0.0) >= 0.5
        levers.append(Lever(
            "heterogeneous_gpu_pricing", has_type,
            "GPU types present — price-aware routing evaluated across types"
            if has_type else
            "no GPU type column — fleet priced as a single GPU class",
            unlock="export the GPU model per job (or per node)",
        ))
        has_status = cov.get("status", 0.0) >= 0.5
        levers.append(Lever(
            "failure_accounting", has_status,
            "status column present — failed jobs excluded from goodput"
            if has_status else
            "no status column — all jobs scored as completed "
            "(goodput may be overstated)",
            unlock="export the terminal job state",
        ))
        levers.append(Lever(
            "fleet_specification", False,
            "fleet size is inferred from peak concurrent GPU demand unless "
            "provided — an ASSUMPTION, labelled in the report",
            unlock="pass --fleet-gpus / --gpus-per-node with your real fleet",
        ))

    # -------------------------------------------------------------- fuzzy
    fuzzy = [m for m in plan.mappings if m.matched_by == "fuzzy"]
    if fuzzy:
        names = ", ".join(f"{m.canonical}←{m.source!r}" for m in fuzzy)
        checks.append(Check(
            "fuzzy_mappings", "warn",
            f"fuzzy-matched mappings need confirmation: {names}",
        ))
    if ext.rows_capped:
        checks.append(Check(
            "row_cap", "warn",
            "input larger than the row cap — replay uses the capped prefix "
            "(pass --max-rows to raise it)",
        ))

    # ---------------------------------------------------------------- tier
    if any(c.level == "fail" for c in checks):
        tier = TIER_INSUFFICIENT
    elif any(c.level == "warn" for c in checks) or \
            any(not lv.available for lv in levers):
        tier = TIER_DEGRADED
    else:
        tier = TIER_READY

    return ReadinessReport(
        tier=tier, workload_class=cls, checks=checks, levers=levers,
        field_coverage=cov, rows_read=ext.rows_read,
        rows_usable=ext.rows_usable, refusal_reasons=refusals,
    )
