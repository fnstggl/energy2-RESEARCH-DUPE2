"""File sniffing + column mapping plan for the self-serve importer.

The importer's job is to turn *any* tabular log export into either

- a confident, reviewable :class:`MappingPlan` onto the canonical fields
  (``fields.py``), or
- a legible explanation of what could not be mapped — never a stack trace.

Matching is deliberately conservative and transparent:

1. exact synonym match on the normalized column name (high confidence),
2. ``difflib`` fuzzy match (adopted only above a strict cutoff, and always
   labelled ``fuzzy`` so the UI/CLI shows it for confirmation),
3. user overrides (``--map canonical=source``) always win,
4. everything else is listed as unmapped — the collaborative-debugging
   surface, not a failure.

Stdlib only, deterministic.
"""

from __future__ import annotations

import csv
import difflib
import gzip
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .fields import (
    FIELDS,
    GPU_JOBS,
    KIND_DURATION,
    KIND_GRES,
    KIND_TIMESTAMP,
    SERVING,
    FieldSpec,
    infer_duration_unit,
    infer_timestamp_unit,
    normalize_column_name,
)

_FUZZY_CUTOFF = 0.88
_SAMPLE_ROWS = 200
_MAX_FIELD_LEN = 4096


class TableReadError(ValueError):
    """Raised when a file cannot be read as a table at all (I/O-level)."""


# ---------------------------------------------------------------------------
# Table reading (CSV / TSV / JSONL, optionally gzipped)
# ---------------------------------------------------------------------------

@dataclass
class RawTable:
    """A lazily-consumable tabular view of the source file."""

    path: Path
    fmt: str                      # "csv" | "tsv" | "jsonl"
    columns: list[str]
    sample_rows: list[dict]       # first _SAMPLE_ROWS rows (str values)
    encoding: str
    dialect_note: str

    def iter_rows(self) -> Iterable[dict]:
        """Stream every row as a {column: str} dict (values stringified)."""
        opener = gzip.open if str(self.path).endswith(".gz") else open
        with opener(self.path, "rt", encoding=self.encoding, errors="replace",
                    newline="") as fh:
            if self.fmt == "jsonl":
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        yield {"__parse_error__": line[:_MAX_FIELD_LEN]}
                        continue
                    if isinstance(obj, dict):
                        yield {str(k): _stringify(v) for k, v in obj.items()}
                    else:
                        yield {"__parse_error__": str(obj)[:_MAX_FIELD_LEN]}
            else:
                delim = "\t" if self.fmt == "tsv" else ","
                reader = csv.DictReader(fh, delimiter=delim)
                for row in reader:
                    yield {
                        (k if k is not None else "__extra__"): _stringify(v)
                        for k, v in row.items()
                    }


def _stringify(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def read_table(path: str | Path) -> RawTable:
    """Open a log export and sniff its format. Raises TableReadError on failure."""
    p = Path(path)
    if not p.exists():
        raise TableReadError(f"file not found: {p}")
    if p.stat().st_size == 0:
        raise TableReadError(f"file is empty: {p}")

    opener = gzip.open if p.name.endswith(".gz") else open
    stem = p.name[:-3] if p.name.endswith(".gz") else p.name
    try:
        with opener(p, "rb") as fb:
            raw_head = fb.read(8192)
        if b"\x00" in raw_head:
            raise TableReadError(
                f"{p} looks like a binary file, not a tabular log export "
                "(CSV / TSV / JSON-lines, optionally gzipped)"
            )
        with opener(p, "rt", encoding="utf-8", errors="replace", newline="") as fh:
            head = fh.read(65536)
    except (OSError, gzip.BadGzipFile) as exc:
        raise TableReadError(f"cannot read {p}: {exc}") from exc
    if not head.strip():
        raise TableReadError(f"file has no content: {p}")

    first_line = head.lstrip().splitlines()[0]
    if stem.endswith((".jsonl", ".ndjson")) or first_line.startswith("{"):
        fmt = "jsonl"
        columns: list[str] = []
        sample: list[dict] = []
        for line in head.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                row = {str(k): _stringify(v) for k, v in obj.items()}
                for k in row:
                    if k not in columns:
                        columns.append(k)
                if len(sample) < _SAMPLE_ROWS:
                    sample.append(row)
        if not columns:
            raise TableReadError(
                f"{p}: looks like JSON lines but no parseable object rows found"
            )
        return RawTable(p, fmt, columns, sample, "utf-8", "json-lines")

    # CSV / TSV
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",\t;|")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if "\t" in first_line else ","
    fmt = "tsv" if delim == "\t" else "csv"
    reader = csv.DictReader(io.StringIO(head), delimiter=delim)
    columns = [c for c in (reader.fieldnames or []) if c and c.strip()]
    if not columns:
        raise TableReadError(f"{p}: could not detect a header row")
    if len(columns) == 1 and fmt == "csv" and ";" in columns[0]:
        # European-style semicolon CSV that the sniffer missed.
        reader = csv.DictReader(io.StringIO(head), delimiter=";")
        columns = [c for c in (reader.fieldnames or []) if c and c.strip()]
        delim = ";"
    sample = []
    for row in reader:
        if len(sample) >= _SAMPLE_ROWS:
            break
        sample.append({(k or "__extra__"): _stringify(v) for k, v in row.items()})
    table = RawTable(p, fmt, columns, sample, "utf-8", f"delimiter={delim!r}")
    if delim == ";":
        table.fmt = "csv"

        def _iter_semicolon(self=table):
            opener2 = gzip.open if str(self.path).endswith(".gz") else open
            with opener2(self.path, "rt", encoding=self.encoding,
                         errors="replace", newline="") as fh:
                for row in csv.DictReader(fh, delimiter=";"):
                    yield {(k or "__extra__"): _stringify(v) for k, v in row.items()}

        table.iter_rows = _iter_semicolon  # type: ignore[method-assign]
    return table


# ---------------------------------------------------------------------------
# Mapping plan
# ---------------------------------------------------------------------------

@dataclass
class ColumnMapping:
    canonical: str
    source: Optional[str]         # raw source column name (None = unmapped)
    matched_by: str               # exact_synonym | fuzzy | user | absent
    confidence: float             # 0..1
    transform: str                # identity | iso8601 | epoch_ms | hms | gres | ...
    evidence: str
    sample_values: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "source": self.source,
            "matched_by": self.matched_by,
            "confidence": round(self.confidence, 3),
            "transform": self.transform,
            "evidence": self.evidence,
            "sample_values": self.sample_values[:5],
        }


@dataclass
class MappingPlan:
    workload_class: str           # serving | gpu_jobs | unknown
    class_confidence: float
    class_evidence: str
    mappings: list[ColumnMapping]
    unmapped_source_columns: list[str]
    warnings: list[str]

    def get(self, canonical: str) -> Optional[ColumnMapping]:
        for m in self.mappings:
            if m.canonical == canonical and m.source is not None:
                return m
        return None

    def to_dict(self) -> dict:
        return {
            "workload_class": self.workload_class,
            "class_confidence": round(self.class_confidence, 3),
            "class_evidence": self.class_evidence,
            "mappings": [m.to_dict() for m in self.mappings],
            "unmapped_source_columns": self.unmapped_source_columns,
            "warnings": self.warnings,
        }


def _samples_for(table: RawTable, source_col: str, n: int = 40) -> list[str]:
    out = []
    for row in table.sample_rows:
        v = row.get(source_col, "")
        if str(v).strip():
            out.append(str(v)[:_MAX_FIELD_LEN])
        if len(out) >= n:
            break
    return out


def _candidate_fields(spec: FieldSpec, workload_class: Optional[str]) -> bool:
    if workload_class is None:
        return True
    return workload_class in spec.classes


def build_mapping_plan(
    table: RawTable,
    *,
    workload_class: Optional[str] = None,
    overrides: Optional[dict[str, str]] = None,
) -> MappingPlan:
    """Match source columns to canonical fields and infer transforms.

    ``overrides`` maps canonical field name → raw source column name and wins
    over every automatic match. ``workload_class`` restricts the candidate
    field set; when ``None`` both classes compete and the plan reports which
    class the columns look like.
    """
    overrides = dict(overrides or {})
    norm_to_raw: dict[str, str] = {}
    for raw in table.columns:
        norm = normalize_column_name(raw)
        norm_to_raw.setdefault(norm, raw)

    warnings: list[str] = []
    for canonical, src in list(overrides.items()):
        if src not in table.columns:
            warnings.append(
                f"override {canonical}={src!r} ignored: column not in file"
            )
            overrides.pop(canonical)

    # First pass: score both classes to detect the workload class.
    def _match_for_class(cls: Optional[str]) -> dict[str, tuple[str, str, float, str]]:
        """canonical -> (raw_col, matched_by, confidence, evidence)"""
        taken: set[str] = set()
        out: dict[str, tuple[str, str, float, str]] = {}
        specs = [f for f in FIELDS if _candidate_fields(f, cls)]
        # user overrides first
        for spec in specs:
            if spec.name in overrides:
                raw = overrides[spec.name]
                out[spec.name] = (raw, "user", 1.0, "explicit user mapping")
                taken.add(raw)
        # exact synonym matches, in registry order (earlier synonyms = stronger)
        for spec in specs:
            if spec.name in out:
                continue
            for syn in spec.synonyms:
                raw = norm_to_raw.get(syn)
                if raw is not None and raw not in taken:
                    out[spec.name] = (
                        raw, "exact_synonym", 0.97,
                        f"column {raw!r} matches synonym {syn!r}",
                    )
                    taken.add(raw)
                    break
        # fuzzy pass for still-missing fields
        available = [n for n in norm_to_raw if norm_to_raw[n] not in taken]
        for spec in specs:
            if spec.name in out or not available:
                continue
            hits = difflib.get_close_matches(
                spec.name, available, n=1, cutoff=_FUZZY_CUTOFF
            )
            if not hits:
                for syn in spec.synonyms[:6]:
                    hits = difflib.get_close_matches(
                        syn, available, n=1, cutoff=_FUZZY_CUTOFF
                    )
                    if hits:
                        break
            if hits:
                raw = norm_to_raw[hits[0]]
                ratio = difflib.SequenceMatcher(None, spec.name, hits[0]).ratio()
                out[spec.name] = (
                    raw, "fuzzy", round(min(0.9, ratio), 3),
                    f"column {raw!r} fuzzy-matches {spec.name!r} "
                    f"(ratio {ratio:.2f}) — confirm before trusting",
                )
                taken.add(raw)
                available.remove(hits[0])
        return out

    detected_class = workload_class
    class_conf, class_evidence = 1.0, "workload class supplied by user"
    if detected_class is None:
        scores: dict[str, float] = {}
        evid: dict[str, list[str]] = {SERVING: [], GPU_JOBS: []}
        trial = {cls: _match_for_class(cls) for cls in (SERVING, GPU_JOBS)}
        for cls in (SERVING, GPU_JOBS):
            score = 0.0
            from .fields import required_fields
            for name, (_raw, _by, conf, _ev) in trial[cls].items():
                spec = next(f for f in FIELDS if f.name == name)
                weight = 2.0 if cls in spec.required_for else 1.0
                if spec.name in ("prompt_tokens", "output_tokens") and cls == SERVING:
                    weight = 3.0
                if spec.name in ("gpu_count", "gres") and cls == GPU_JOBS:
                    weight = 3.0
                score += weight * conf
                evid[cls].append(name)
            req = required_fields(cls)
            missing_req = [
                r for r in req
                if r not in trial[cls]
                and not (cls == GPU_JOBS and r == "gpu_count" and "gres" in trial[cls])
            ]
            score -= 2.5 * len(missing_req)
            scores[cls] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, second = ranked[0], ranked[1]
        if best[1] <= 0:
            detected_class = "unknown"
            class_conf = 0.0
            class_evidence = (
                "no workload class matched enough required fields "
                f"(serving score {scores[SERVING]:.1f}, "
                f"gpu_jobs score {scores[GPU_JOBS]:.1f})"
            )
        else:
            detected_class = best[0]
            spread = best[1] - second[1]
            class_conf = max(0.5, min(1.0, 0.5 + spread / 10.0))
            class_evidence = (
                f"matched fields for {best[0]}: "
                f"{', '.join(sorted(evid[best[0]]))} "
                f"(score {best[1]:.1f} vs {second[0]} {second[1]:.1f})"
            )

    effective_class = detected_class if detected_class in (SERVING, GPU_JOBS) else None
    matches = _match_for_class(effective_class)

    mappings: list[ColumnMapping] = []
    used_raw: set[str] = set()
    specs = [f for f in FIELDS if _candidate_fields(f, effective_class)]
    for spec in specs:
        if spec.name in matches:
            raw, by, conf, ev = matches[spec.name]
            samples = _samples_for(table, raw)
            transform, t_evidence = "identity", ""
            if spec.kind == KIND_TIMESTAMP:
                transform, t_evidence = infer_timestamp_unit(samples)
            elif spec.kind == KIND_DURATION:
                transform, t_evidence = infer_duration_unit(raw, samples)
            elif spec.kind == KIND_GRES:
                transform, t_evidence = "gres", "Slurm TRES string parse"
            evidence = ev if not t_evidence else f"{ev}; {t_evidence}"
            if spec.kind == KIND_TIMESTAMP and transform == "unknown":
                warnings.append(
                    f"{spec.name}: could not infer timestamp encoding for "
                    f"column {raw!r} — rows will fail validation until mapped"
                )
            mappings.append(ColumnMapping(
                canonical=spec.name, source=raw, matched_by=by,
                confidence=conf, transform=transform, evidence=evidence,
                sample_values=samples[:5],
            ))
            used_raw.add(raw)
        else:
            mappings.append(ColumnMapping(
                canonical=spec.name, source=None, matched_by="absent",
                confidence=0.0, transform="identity",
                evidence=spec.description, sample_values=[],
            ))

    unmapped = [c for c in table.columns if c not in used_raw]
    return MappingPlan(
        workload_class=detected_class or "unknown",
        class_confidence=class_conf,
        class_evidence=class_evidence,
        mappings=mappings,
        unmapped_source_columns=unmapped,
        warnings=warnings,
    )
