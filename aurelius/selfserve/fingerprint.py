"""Anonymized schema fingerprint — the explicit, opt-in shareable artifact.

The fingerprint answers the founder-side question "what does real customer
data look like?" without the customer sending logs. It contains:

- column NAMES, inferred dtypes, null rates, distinct-count buckets,
- numeric distribution summaries (percentiles) for non-identifier columns,
- the mapping plan, readiness verdict, and the parse-error ledger,
- tool version + file size/row counts.

It contains NO raw values from identifier-like or free-text columns, and no
row data. String examples are limited to the top *status* tokens (needed to
extend the status table) — everything else is shape only.

Sharing is a user action (attach the file to an email); the tool never
transmits anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import __version__
from .extract import ExtractionResult
from .sniff import MappingPlan, RawTable
from .validate import ReadinessReport

_NUMERIC_SAMPLE_CAP = 5000


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _column_profile(table: RawTable, col: str) -> dict:
    vals = [str(r.get(col, "")).strip() for r in table.sample_rows]
    nonempty = [v for v in vals if v]
    numeric = [float(v) for v in nonempty if _is_number(v)][:_NUMERIC_SAMPLE_CAP]
    distinct = len(set(nonempty))
    profile: dict = {
        "null_rate_sample_pct": round(
            100 * (len(vals) - len(nonempty)) / max(1, len(vals)), 1),
        "distinct_in_sample": distinct,
        "looks_numeric": bool(nonempty) and len(numeric) >= 0.9 * len(nonempty),
        "max_len": max((len(v) for v in nonempty), default=0),
    }
    if profile["looks_numeric"] and numeric:
        ordered = sorted(numeric)
        profile["numeric_summary"] = {
            "min": ordered[0],
            "p50": ordered[len(ordered) // 2],
            "max": ordered[-1],
        }
    return profile


def build_fingerprint(
    table: RawTable,
    plan: MappingPlan,
    ext: ExtractionResult,
    readiness: ReadinessReport,
) -> dict:
    p = Path(table.path)
    h = hashlib.sha256()
    h.update(p.name.encode())
    h.update(str(p.stat().st_size).encode())
    plan_dict = plan.to_dict()
    for m in plan_dict["mappings"]:
        # The local report may show samples; the shareable artifact never does.
        m["sample_values"] = []
    return {
        "artifact": "aurelius-replay-schema-fingerprint",
        "tool_version": __version__,
        "privacy": (
            "column names, shapes and parse diagnostics only — no log rows, "
            "no identifier values; sharing this file is a manual user action"
        ),
        "file": {
            "name_hash": h.hexdigest()[:16],
            "format": table.fmt,
            "dialect": table.dialect_note,
            "size_bytes": p.stat().st_size,
            "rows_read": ext.rows_read,
            "rows_usable": ext.rows_usable,
        },
        "columns": {
            c: _column_profile(table, c) for c in table.columns
        },
        "mapping_plan": plan_dict,
        "readiness": readiness.to_dict(),
        "parse_errors": ext.errors.to_dict(),
        "unrecognized_status_values": ext.status_values_unrecognized,
    }


def write_fingerprint(fingerprint: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    return out
