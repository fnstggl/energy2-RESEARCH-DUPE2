"""Row extraction: mapped source rows → canonical records + error ledger.

Extraction never throws on a bad row. Every row either becomes a canonical
record or increments a named error counter (with a bounded set of examples,
values-scrubbed) — the ledger is what the validator and the fingerprint
report. This is the "fail legibly" half of the never-break contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .fields import (
    GPU_JOBS,
    SERVING,
    coerce_duration,
    coerce_int,
    coerce_str,
    coerce_timestamp,
    parse_gres_gpus,
    status_to_failure,
)
from .sniff import MappingPlan, RawTable

_MAX_ROWS_DEFAULT = 2_000_000


@dataclass
class ErrorLedger:
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, key: str, example: str = "") -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        if example:
            ex = self.examples.setdefault(key, [])
            if len(ex) < 3:
                ex.append(example[:160])

    def to_dict(self) -> dict:
        return {"counts": dict(sorted(self.counts.items())),
                "examples": self.examples}


@dataclass
class ExtractionResult:
    workload_class: str
    rows_read: int
    rows_usable: int
    rows_capped: bool
    records: list[dict]                 # canonical field name -> parsed value
    errors: ErrorLedger
    field_nonnull: dict[str, int]       # canonical field -> non-null count
    status_values_unrecognized: dict[str, int]

    def coverage(self) -> dict[str, float]:
        if not self.rows_usable:
            return {k: 0.0 for k in self.field_nonnull}
        return {
            k: round(v / self.rows_usable, 4)
            for k, v in sorted(self.field_nonnull.items())
        }


def extract(
    table: RawTable,
    plan: MappingPlan,
    *,
    max_rows: int = _MAX_ROWS_DEFAULT,
) -> ExtractionResult:
    """Stream the table through the mapping plan into canonical records."""
    cls = plan.workload_class
    getters = {m.canonical: m for m in plan.mappings if m.source is not None}
    errors = ErrorLedger()
    records: list[dict] = []
    nonnull: dict[str, int] = {}
    unrecognized_status: dict[str, int] = {}
    rows_read = 0
    capped = False

    for row in table.iter_rows():
        if rows_read >= max_rows:
            capped = True
            break
        rows_read += 1
        if "__parse_error__" in row:
            errors.add("row_unparseable")
            continue

        rec: dict = {}
        row_bad: Optional[str] = None

        for canonical, m in getters.items():
            raw_val = row.get(m.source, "")
            if raw_val is None or not str(raw_val).strip():
                rec[canonical] = None
                continue
            raw_str = str(raw_val)
            if canonical in ("timestamp", "submit_time", "start_time", "end_time"):
                v = coerce_timestamp(raw_str, m.transform)
                if v is None:
                    errors.add(f"{canonical}_unparseable", raw_str)
                    if canonical in ("timestamp", "submit_time"):
                        row_bad = f"{canonical}_unparseable"
                rec[canonical] = v
            elif canonical in ("duration", "queue_wait", "elapsed"):
                v = coerce_duration(raw_str, m.transform)
                if v is None:
                    errors.add(f"{canonical}_unparseable", raw_str)
                rec[canonical] = v
            elif canonical in ("prompt_tokens", "output_tokens", "gpu_count"):
                v = coerce_int(raw_str)
                if v is None:
                    errors.add(f"{canonical}_unparseable", raw_str)
                    if cls == SERVING and canonical in ("prompt_tokens",
                                                        "output_tokens"):
                        row_bad = f"{canonical}_unparseable"
                    if cls == GPU_JOBS and canonical == "gpu_count":
                        row_bad = "gpu_count_unparseable"
                rec[canonical] = v
            elif canonical == "gres":
                n, gtype = parse_gres_gpus(raw_str)
                rec["gres"] = raw_str
                if n is not None and rec.get("gpu_count") is None:
                    rec["gpu_count"] = n
                if gtype and not rec.get("gpu_type"):
                    rec["gpu_type"] = gtype
            elif canonical == "status":
                s = coerce_str(raw_str)
                rec["status"] = s
                if s is not None:
                    failed = status_to_failure(s)
                    rec["is_failure"] = failed
                    if failed is None:
                        key = s.lower()[:40]
                        unrecognized_status[key] = (
                            unrecognized_status.get(key, 0) + 1
                        )
            else:
                rec[canonical] = coerce_str(raw_str)

        # Derivations ------------------------------------------------------
        if cls == GPU_JOBS:
            if rec.get("duration") is None:
                st, en = rec.get("start_time"), rec.get("end_time")
                if st is not None and en is not None and en >= st:
                    rec["duration"] = en - st
            if rec.get("queue_wait") is None:
                sub, st = rec.get("submit_time"), rec.get("start_time")
                if sub is not None and st is not None and st >= sub:
                    rec["queue_wait"] = st - sub

        # Row-level acceptance --------------------------------------------
        if row_bad:
            errors.add("row_dropped_" + row_bad)
            continue
        if cls == SERVING:
            if rec.get("timestamp") is None:
                errors.add("row_dropped_no_timestamp")
                continue
            if rec.get("prompt_tokens") is None and rec.get("output_tokens") is None:
                errors.add("row_dropped_no_tokens")
                continue
        elif cls == GPU_JOBS:
            if rec.get("submit_time") is None:
                errors.add("row_dropped_no_submit_time")
                continue
            if not rec.get("gpu_count"):
                errors.add("row_dropped_no_gpu_count")
                continue
            if rec.get("duration") is None or rec["duration"] <= 0:
                errors.add("row_dropped_no_duration")
                continue

        for k, v in rec.items():
            if v is not None:
                nonnull[k] = nonnull.get(k, 0) + 1
        records.append(rec)

    return ExtractionResult(
        workload_class=cls,
        rows_read=rows_read,
        rows_usable=len(records),
        rows_capped=capped,
        records=records,
        errors=errors,
        field_nonnull=nonnull,
        status_values_unrecognized=dict(sorted(
            unrecognized_status.items(), key=lambda kv: -kv[1])[:10]),
    )
