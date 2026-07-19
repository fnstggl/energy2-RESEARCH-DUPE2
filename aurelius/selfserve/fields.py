"""Canonical field registry + value coercion for the self-serve importer.

Two workload classes are supported in the POC, matching the two locked
trace-replay harnesses:

- ``serving``  → ``aurelius/traces/schema.py::NormalizedLLMRequest``
                 replayed by ``aurelius/traces/backtest.py::run_backtest``
- ``gpu_jobs`` → ``aurelius/traces/schema.py::NormalizedGPUJob``
                 replayed by ``aurelius/traces/gpu_scheduling.py::run_backtest``

Every canonical field carries an explicit synonym list (matched on a
normalized column name) so that vendor-flavored exports (Slurm ``sacct``,
K8s job dumps, gateway logs, vLLM access logs, …) map without the user
renaming anything. Every coercion records *evidence* for how a unit was
inferred — silent unit guessing is exactly how a self-serve tool produces a
confidently wrong number, which is worse than no number.

Stdlib only, deterministic. No pandas, matching ``aurelius/traces``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

SERVING = "serving"
GPU_JOBS = "gpu_jobs"
WORKLOAD_CLASSES = (SERVING, GPU_JOBS)

# ---------------------------------------------------------------------------
# Field kinds
# ---------------------------------------------------------------------------

KIND_TIMESTAMP = "timestamp"   # absolute time → epoch seconds (float)
KIND_DURATION = "duration"     # elapsed time → seconds (float)
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_STR = "str"
KIND_STATUS = "status"         # free-form status → is_failure bool
KIND_GRES = "gres"             # Slurm TRES string → gpu_count (+ gpu_type)


@dataclass(frozen=True)
class FieldSpec:
    """One canonical field the importer knows how to map onto."""

    name: str
    kind: str
    required_for: tuple[str, ...]   # workload classes where this is required
    classes: tuple[str, ...]        # workload classes where this is meaningful
    synonyms: tuple[str, ...]       # normalized source-column names
    description: str


def normalize_column_name(raw: str) -> str:
    """Lowercase, strip units in parens, collapse non-alphanumerics to ``_``."""
    s = raw.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)          # "duration (s)" -> "duration "
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# ---------------------------------------------------------------------------
# Registry
#
# Synonyms are matched against normalize_column_name(raw). Sources: Slurm
# sacct fields, Kubernetes/Volcano job dumps, Philly/Alibaba/MIT public
# traces, vLLM / gateway access-log conventions, and the pilot-guide CSV
# (enterprisedocs/pilot-guide.md — job_id, workload_type, submit_time,
# duration). Extending this table IS the product loop: every fingerprint a
# prospect shares adds rows here.
# ---------------------------------------------------------------------------

FIELDS: tuple[FieldSpec, ...] = (
    # --- shared -----------------------------------------------------------
    FieldSpec(
        "status", KIND_STATUS, (), (SERVING, GPU_JOBS),
        ("status", "state", "job_state", "final_status", "http_status",
         "status_code", "result", "exit_code", "exitcode", "phase", "pod_phase",
         "job_status", "error", "outcome"),
        "Terminal status; mapped to a failure flag with a documented value table.",
    ),
    # --- serving ----------------------------------------------------------
    FieldSpec(
        "timestamp", KIND_TIMESTAMP, (SERVING,), (SERVING,),
        ("timestamp", "ts", "time", "arrival_time", "arrival", "request_time",
         "request_timestamp", "created_at", "date", "datetime", "start_time",
         "event_time", "received_at", "@timestamp"),
        "Request arrival time.",
    ),
    FieldSpec(
        "prompt_tokens", KIND_INT, (SERVING,), (SERVING,),
        ("prompt_tokens", "input_tokens", "request_tokens", "context_tokens",
         "contexttokens", "input_tok", "in_tokens", "tokens_in", "prompt_len",
         "prompt_length", "input_length", "n_input_tokens", "prompt_size"),
        "Prompt/context tokens per request.",
    ),
    FieldSpec(
        "output_tokens", KIND_INT, (SERVING,), (SERVING,),
        ("output_tokens", "response_tokens", "generated_tokens",
         "generatedtokens", "completion_tokens", "gen_tok", "out_tokens",
         "tokens_out", "output_len", "output_length", "n_output_tokens",
         "decode_tokens", "generation_tokens"),
        "Generated/output tokens per request.",
    ),
    FieldSpec(
        "model", KIND_STR, (), (SERVING,),
        ("model", "model_name", "model_id", "deployment", "deployment_name",
         "endpoint", "endpoint_id", "engine", "served_model"),
        "Model / deployment the request hit (enables model-mix routing).",
    ),
    FieldSpec(
        "session_id", KIND_STR, (), (SERVING,),
        ("session_id", "session", "conversation_id", "conv_id", "chat_id",
         "user_session", "thread_id", "trace_id_session"),
        "Session/conversation key (enables the cache-affinity lever; "
        "without it NO cache benefit is simulated — never invented).",
    ),
    FieldSpec(
        "request_id", KIND_STR, (), (SERVING,),
        ("request_id", "req_id", "id", "uuid", "trace_id", "correlation_id"),
        "Request id (synthesized from row number when absent).",
    ),
    FieldSpec(
        "elapsed", KIND_DURATION, (), (SERVING,),
        ("elapsed", "elapsed_s", "latency", "latency_ms", "latency_s",
         "duration_ms", "e2e_latency", "response_time", "response_time_ms",
         "total_time", "e2e_s", "request_duration"),
        "Measured end-to-end latency (diagnostic only; never treated as TTFT).",
    ),
    # --- gpu_jobs ---------------------------------------------------------
    FieldSpec(
        "job_id", KIND_STR, (), (GPU_JOBS,),
        ("job_id", "jobid", "job", "id", "name", "job_name", "pod_name",
         "task_id", "run_id", "uuid"),
        "Job id (synthesized from row number when absent).",
    ),
    FieldSpec(
        "submit_time", KIND_TIMESTAMP, (GPU_JOBS,), (GPU_JOBS,),
        ("submit_time", "submit", "submitted", "submitted_time", "submission_time",
         "queued_at", "creation_time", "created_at", "create_time", "queue_time",
         "enqueue_time", "arrival_time", "timestamp"),
        "Job submission time.",
    ),
    FieldSpec(
        "start_time", KIND_TIMESTAMP, (), (GPU_JOBS,),
        ("start_time", "start", "started", "started_time", "scheduled_time",
         "run_start", "begin_time", "dispatch_time", "exec_start"),
        "Actual start time (unlocks measured queue-wait calibration).",
    ),
    FieldSpec(
        "end_time", KIND_TIMESTAMP, (), (GPU_JOBS,),
        ("end_time", "end", "ended", "finish_time", "finished_time", "completion_time",
         "completed_at", "deletion_time", "stop_time", "run_end", "exec_end"),
        "End time (with start_time, substitutes for an explicit duration).",
    ),
    FieldSpec(
        "duration", KIND_DURATION, (), (GPU_JOBS,),
        ("duration", "duration_s", "duration_sec", "duration_seconds",
         "duration_hours", "duration_hrs", "runtime", "runtime_hours",
         "runtime_s", "run_time", "elapsed", "elapsed_time", "walltime",
         "wall_time", "execution_time", "job_duration"),
        "Job runtime (or derived from start/end when both exist).",
    ),
    FieldSpec(
        "gpu_count", KIND_INT, (GPU_JOBS,), (GPU_JOBS,),
        ("gpu_count", "gpus", "num_gpu", "num_gpus", "n_gpus", "gpu_num",
         "gpu_request", "gpus_requested", "requested_gpus", "alloc_gpus",
         "gpu", "ngpus", "gpu_cnt", "totalgpus"),
        "Whole GPUs requested by the job.",
    ),
    FieldSpec(
        "gres", KIND_GRES, (), (GPU_JOBS,),
        ("alloctres", "reqtres", "tres", "gres", "alloc_tres", "req_tres",
         "tres_alloc", "tres_req", "resources"),
        "Slurm TRES/GRES string; ``gres/gpu=N`` (and gpu type) is parsed out.",
    ),
    FieldSpec(
        "gpu_type", KIND_STR, (), (GPU_JOBS,),
        ("gpu_type", "gpu_model", "gpu_spec", "accelerator", "accelerator_type",
         "gpu_sku", "device_type", "instance_type"),
        "GPU model (enables price-aware routing across heterogeneous GPUs).",
    ),
    FieldSpec(
        "user_or_group", KIND_STR, (), (GPU_JOBS,),
        ("user", "username", "user_id", "account", "group", "team",
         "project", "vc", "tenant", "tenant_id", "namespace", "queue"),
        "Owner (tenant fairness diagnostics; hashed in the fingerprint).",
    ),
    FieldSpec(
        "workload_type", KIND_STR, (), (GPU_JOBS,),
        ("workload_type", "job_type", "type", "task_type", "qos", "partition",
         "priority_class", "workload"),
        "Workload class label.",
    ),
    FieldSpec(
        "priority", KIND_STR, (), (GPU_JOBS,),
        ("priority", "prio", "preemption_priority", "sched_priority"),
        "Scheduling priority.",
    ),
    FieldSpec(
        "queue_wait", KIND_DURATION, (), (GPU_JOBS,),
        ("queue_wait", "queue_wait_s", "wait_time", "wait_time_s", "queue_time_s",
         "pending_time", "queue_delay", "wait", "queue_duration"),
        "Measured queue wait (else derived as start_time − submit_time).",
    ),
)

FIELDS_BY_NAME = {f.name: f for f in FIELDS}


def required_fields(workload_class: str) -> tuple[str, ...]:
    return tuple(f.name for f in FIELDS if workload_class in f.required_for)


# ---------------------------------------------------------------------------
# Status → failure table (documented, not guessed)
# ---------------------------------------------------------------------------

FAILURE_STATUS_VALUES = frozenset({
    "failed", "fail", "failure", "error", "err", "killed", "kill",
    "node_fail", "nodefail", "timeout", "timed_out", "deadline_exceeded",
    "oom", "oomkilled", "out_of_memory", "cancelled", "canceled",
    "cancelled_by_user", "preempted_failed", "boot_fail", "crashed",
})

SUCCESS_STATUS_VALUES = frozenset({
    "completed", "complete", "completing", "success", "succeeded", "ok",
    "pass", "passed", "finished", "done", "running", "200", "0",
})


def status_to_failure(value: str) -> Optional[bool]:
    """Map a raw status value to is_failure; ``None`` when unrecognized.

    Slurm compound states like ``CANCELLED by 1234`` collapse to their first
    token. Numeric values are read as HTTP-style (>=400 → failure) or
    exit-code-style (nonzero → failure) only when cleanly integral.
    """
    v = value.strip().lower()
    if not v:
        return None
    v = re.sub(r"[^a-z0-9_]+", "_", v).strip("_")
    first = v.split("_by_")[0] if "_by_" in v else v
    if first in FAILURE_STATUS_VALUES or v in FAILURE_STATUS_VALUES:
        return True
    if first in SUCCESS_STATUS_VALUES or v in SUCCESS_STATUS_VALUES:
        return False
    try:
        n = int(v)
    except ValueError:
        return None
    if 100 <= n <= 599:          # HTTP status
        return n >= 400
    return n != 0                # exit code


# ---------------------------------------------------------------------------
# Timestamp / duration coercion with recorded evidence
# ---------------------------------------------------------------------------

_EPOCH_2000_S = 946_684_800.0
_EPOCH_2100_S = 4_102_444_800.0

_DT_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S", "%Y%m%d%H%M%S", "%Y-%m-%d",
)


def _parse_datetime_str(s: str) -> Optional[float]:
    """Parse an absolute datetime string → epoch seconds (UTC assumed if naive)."""
    txt = s.strip()
    if not txt:
        return None
    iso = txt.replace("Z", "+00:00") if txt.endswith("Z") else txt
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        dt = None
    if dt is None:
        for fmt in _DT_FORMATS:
            try:
                dt = datetime.strptime(txt, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def infer_timestamp_unit(samples: list[str]) -> tuple[str, str]:
    """Infer how a timestamp column is encoded from non-null samples.

    Returns ``(transform, evidence)`` where transform is one of
    ``iso8601 | epoch_s | epoch_ms | epoch_us | relative_s | unknown``.
    """
    numeric: list[float] = []
    parsed_str = 0
    considered = 0
    for s in samples:
        txt = str(s).strip()
        if not txt:
            continue
        considered += 1
        try:
            numeric.append(float(txt))
            continue
        except ValueError:
            pass
        if _parse_datetime_str(txt) is not None:
            parsed_str += 1
    if considered == 0:
        return "unknown", "no non-empty samples"
    if parsed_str and parsed_str >= considered * 0.8:
        return "iso8601", f"{parsed_str}/{considered} samples parse as datetime strings"
    if len(numeric) >= considered * 0.8 and numeric:
        mags = sorted(abs(x) for x in numeric)
        med = mags[len(mags) // 2]
        if med >= _EPOCH_2000_S * 1e6:
            return "epoch_us", f"numeric magnitude ~{med:.3g} → microseconds since epoch"
        if med >= _EPOCH_2000_S * 1e3:
            return "epoch_ms", f"numeric magnitude ~{med:.3g} → milliseconds since epoch"
        if _EPOCH_2000_S <= med <= _EPOCH_2100_S:
            return "epoch_s", f"numeric magnitude ~{med:.3g} → seconds since epoch"
        return "relative_s", (
            f"numeric magnitude ~{med:.3g} below epoch range → relative seconds"
        )
    return "unknown", "samples neither consistently numeric nor datetime strings"


def coerce_timestamp(value: str, transform: str) -> Optional[float]:
    """Apply an inferred timestamp transform → seconds (epoch or relative)."""
    txt = str(value).strip()
    if not txt:
        return None
    if transform == "iso8601":
        return _parse_datetime_str(txt)
    try:
        x = float(txt)
    except ValueError:
        # Mixed columns: fall back to string parse so a stray ISO row still lands.
        return _parse_datetime_str(txt)
    if transform == "epoch_us":
        return x / 1e6
    if transform == "epoch_ms":
        return x / 1e3
    return x  # epoch_s | relative_s


_SLURM_DUR = re.compile(
    r"^(?:(?P<days>\d+)-)?(?P<h>\d{1,3}):(?P<m>\d{2})(?::(?P<s>\d{2}(?:\.\d+)?))?$"
)


def infer_duration_unit(column_name: str, samples: list[str]) -> tuple[str, str]:
    """Infer duration encoding: ``seconds | ms | hours | minutes | hms``.

    Column-name hints win (``_ms``, ``_hours``…); Slurm ``D-HH:MM:SS`` strings
    are detected from values; otherwise seconds is the documented default.
    """
    name = normalize_column_name(column_name)
    hms = sum(1 for s in samples if _SLURM_DUR.match(str(s).strip()))
    nonempty = sum(1 for s in samples if str(s).strip())
    if nonempty and hms >= nonempty * 0.8:
        return "hms", f"{hms}/{nonempty} samples look like [D-]HH:MM:SS"
    if re.search(r"(_|^)(ms|msec|millis|milliseconds)($|_)", name):
        return "ms", "column name indicates milliseconds"
    if re.search(r"(_|^)(hours|hrs|hr|h)($|_)", name):
        return "hours", "column name indicates hours"
    if re.search(r"(_|^)(minutes|mins|min)($|_)", name):
        return "minutes", "column name indicates minutes"
    return "seconds", "default unit (no name hint, values not [D-]HH:MM:SS)"


def coerce_duration(value: str, transform: str) -> Optional[float]:
    txt = str(value).strip()
    if not txt:
        return None
    if transform == "hms":
        m = _SLURM_DUR.match(txt)
        if not m:
            return None
        days = int(m.group("days") or 0)
        secs = float(m.group("s") or 0.0)
        return days * 86400 + int(m.group("h")) * 3600 + int(m.group("m")) * 60 + secs
    try:
        x = float(txt)
    except ValueError:
        return None
    if transform == "ms":
        return x / 1e3
    if transform == "hours":
        return x * 3600.0
    if transform == "minutes":
        return x * 60.0
    return x


_GRES_GPU = re.compile(r"gres/gpu(?::(?P<type>[a-z0-9_.-]+))?=(?P<n>\d+)", re.I)


def parse_gres_gpus(value: str) -> tuple[Optional[int], Optional[str]]:
    """Extract (gpu_count, gpu_type) from a Slurm TRES string.

    ``billing=64,cpu=64,gres/gpu=8,mem=512G`` → (8, None);
    ``gres/gpu:h100=4`` → (4, "h100"). Typed entries win over the untyped
    total when both appear.
    """
    txt = str(value).strip()
    if not txt:
        return None, None
    typed_n, typed_t, plain_n = None, None, None
    for m in _GRES_GPU.finditer(txt):
        n = int(m.group("n"))
        if m.group("type"):
            typed_n, typed_t = n, m.group("type")
        else:
            plain_n = n
    if typed_n is not None:
        return typed_n, typed_t
    if plain_n is not None:
        return plain_n, None
    return None, None


def coerce_int(value: str) -> Optional[int]:
    txt = str(value).strip()
    if not txt:
        return None
    try:
        return int(float(txt))
    except ValueError:
        return None


def coerce_str(value: str) -> Optional[str]:
    txt = str(value).strip()
    if not txt or txt.upper() in ("NULL", "NONE", "NA", "N/A", "NAN"):
        return None
    return txt
