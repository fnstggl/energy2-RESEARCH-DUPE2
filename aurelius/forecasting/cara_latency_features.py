"""Feature engineering for the CARA latency forecaster v1.

This module is a pure feature pipeline. It does NOT train models, NOT
call any scheduler, NOT mutate state. The forecaster
(`aurelius/forecasting/cara_latency_forecaster.py`) and the driver
script (`scripts/run_cara_latency_forecaster_v1.py`) consume the
matrices this module builds.

Honesty rules (binding):

- ``LEAKAGE_TARGET_FIELDS`` lists every column that depends on the
  request's completion. None of them are ever used as a feature in the
  ``predicted_only_model`` variant.
- ``actual_output_tokens`` is leakage at prediction time (it is only
  known post-completion). The ``predicted_only_model`` uses
  ``num_predicted_output_tokens`` instead. An ``oracle_shape_model``
  variant is supported but every metric it emits must be labelled
  ``analysis_only``.
- ``completion_timestamp_s`` is leakage and never enters the feature set.
- ``prediction_timestamp_s`` is allowed; only an ``hour_of_day``
  derivation is exposed.
- Bin boundaries are pre-registered constants; they are NEVER fitted
  from the holdout.

Inputs are list-of-dict rows (the gitignored
``data/external/hf/asdwb__cara_latency_prediction/<config>/processed/
analysis_sample.jsonl`` files). Outputs are dense numpy arrays plus
the matching list of feature names + group-key arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Leakage rules + canonical targets
# ---------------------------------------------------------------------------


# Fields that are only observable after the request completes. They MUST
# NOT appear in the predict-time feature set (use as labels only).
LEAKAGE_TARGET_FIELDS = frozenset({
    "actual_e2e_latency_s",
    "actual_ttft_s",
    "actual_tpot_s",
    "actual_output_tokens",
    "completion_timestamp_s",
    # The audit-tier ingestion drops elapsed-style fields; we still list
    # the legacy names defensively so callers cannot reintroduce them.
    "actual_e2e_latency",
    "actual_ttft",
    "actual_tpot",
    "completion_timestamp",
})


# The targets v1 trains on.
TARGETS = ("actual_ttft_s", "actual_e2e_latency_s")


# Predict-time features (allowed). Every entry must be a top-level CARA
# normalised column. ``derive_*`` columns are appended in
# ``build_feature_matrix`` after one-hot / bin encoding.
PREDICT_TIME_NUMERIC_FEATURES = (
    # Request shape known at decision time.
    "num_prompt_tokens",
    "num_predicted_output_tokens",
    # Queue + scheduler state at decision time.
    "num_running",
    "num_waiting",
    "num_active_decode_seqs",
    "running_requests_count",
    "waiting_requests_count",
    "pending_prefill_tokens",
    "pending_decode_tokens",
    "decode_ctx_p50",
    "decode_ctx_p95",
    "decode_ctx_max",
    "token_budget_per_iter",
    "prefill_chunk_size",
    "max_num_seqs",
    "num_preempted",
    # KV cache pressure at decision time.
    "kv_cache_utilization",
    "kv_free_blocks",
    "kv_evictions_per_s",
    # Throughput priors (EMA — derived from history, not the request).
    "ema_decode_tok_per_s",
    "ema_prefill_tok_per_s",
    "ema_decode_iter_ms",
)


PREDICT_TIME_CATEGORICAL_FEATURES = (
    "instance_type",
    # Derived from instance_type, but exposed explicitly so the model
    # can split on them.
    "model_size",
    "gpu_type",
    # Bin-encoded numeric features.
    "prompt_token_bin",
    "predicted_output_token_bin",
    "queue_depth_bin",
    "kv_util_bin",
    # Hour-of-day from prediction_timestamp_s.
    "hour_of_day",
)


# ---------------------------------------------------------------------------
# Instance-type parsing
# ---------------------------------------------------------------------------


def derive_model_size(instance_type: Optional[str]) -> Optional[str]:
    """Return the model-size token from a CARA instance_type.

    Example: ``"qwen2.5-3b_p100"`` -> ``"3b"``. Returns ``None`` when
    the input is malformed.
    """
    if not instance_type:
        return None
    head = str(instance_type).split("_", 1)[0]
    parts = head.split("-")
    if len(parts) < 2:
        return None
    return parts[-1].lower()


def derive_gpu_type(instance_type: Optional[str]) -> Optional[str]:
    """Return the GPU token from a CARA instance_type."""
    if not instance_type:
        return None
    parts = str(instance_type).split("_", 1)
    if len(parts) != 2:
        return None
    return parts[1].lower()


# ---------------------------------------------------------------------------
# Pre-registered bin boundaries (NEVER fitted from data)
# ---------------------------------------------------------------------------


PROMPT_TOKEN_BINS = [(0, 50), (50, 200), (200, 800), (800, 3200),
                     (3200, 1_000_000)]
OUTPUT_TOKEN_BINS = [(0, 64), (64, 256), (256, 1024), (1024, 4096),
                     (4096, 1_000_000)]
QUEUE_DEPTH_BINS = [(0, 1), (1, 5), (5, 20), (20, 100), (100, 1_000_000)]
KV_UTIL_BINS = [(0.0, 0.1), (0.1, 0.4), (0.4, 0.7), (0.7, 0.9), (0.9, 1.01)]


def _bin_label(v, bins) -> str:
    if v is None:
        return "missing"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi in bins:
        if lo <= v < hi:
            return f"[{lo},{hi})"
    return f">={bins[-1][1]}"


def bin_prompt_tokens(n) -> str:
    return _bin_label(n, PROMPT_TOKEN_BINS)


def bin_output_tokens(n) -> str:
    return _bin_label(n, OUTPUT_TOKEN_BINS)


def bin_queue_depth(n) -> str:
    return _bin_label(n, QUEUE_DEPTH_BINS)


def bin_kv_util(v) -> str:
    return _bin_label(v, KV_UTIL_BINS)


def hour_of_day(ts) -> Optional[int]:
    if ts is None:
        return None
    try:
        return int(datetime.fromtimestamp(float(ts), tz=timezone.utc).hour)
    except (TypeError, ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Frozen feature specification — fixed across train + holdout."""

    output_tokens_mode: str  # "predicted_only" | "oracle_shape"
    categorical_levels: dict  # column_name -> sorted list of levels
    numeric_columns: tuple
    categorical_columns: tuple


class LeakageError(ValueError):
    """Raised when a feature flagged as leakage would be emitted."""


def _assert_no_leakage(feature_names: Iterable[str], mode: str) -> None:
    bad = [f for f in feature_names if f in LEAKAGE_TARGET_FIELDS]
    # The oracle_shape mode is allowed to use actual_output_tokens; it
    # must label every result analysis_only.
    if mode == "oracle_shape":
        bad = [f for f in bad if f != "actual_output_tokens"]
    if bad:
        raise LeakageError(
            f"feature pipeline would emit leakage columns {bad} "
            f"(mode='{mode}'); refusing to build feature matrix"
        )


def _enrich_row(row: dict, *, mode: str) -> dict:
    out = dict(row)
    out["model_size"] = derive_model_size(row.get("instance_type"))
    out["gpu_type"] = derive_gpu_type(row.get("instance_type"))
    out["prompt_token_bin"] = bin_prompt_tokens(row.get("num_prompt_tokens"))
    out["predicted_output_token_bin"] = bin_output_tokens(
        row.get("num_predicted_output_tokens")
    )
    out["queue_depth_bin"] = bin_queue_depth(row.get("num_running"))
    out["kv_util_bin"] = bin_kv_util(row.get("kv_cache_utilization"))
    h = hour_of_day(row.get("prediction_timestamp_s"))
    out["hour_of_day"] = ("hour=" + str(h)) if h is not None else "missing"
    if mode == "oracle_shape":
        # The oracle-shape variant additionally exposes actual_output_tokens
        # as a numeric feature (analysis_only metric label is forced
        # downstream by the forecaster).
        pass
    return out


def _coerce_numeric(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fit_categorical_levels(
    rows: Iterable[dict], categorical_columns: tuple, *, mode: str,
) -> dict:
    """Discover the distinct levels of each categorical column across the
    training rows. Holdout rows that carry a new level are encoded as the
    last (sentinel) one-hot column to prevent silent drift."""
    levels: dict = {c: set() for c in categorical_columns}
    for r in rows:
        enriched = _enrich_row(r, mode=mode)
        for c in categorical_columns:
            v = enriched.get(c)
            if v is None:
                v = "missing"
            levels[c].add(str(v))
    return {c: sorted(v) for c, v in levels.items()}


def build_feature_spec(
    rows: list,
    *,
    output_tokens_mode: str = "predicted_only",
) -> FeatureSpec:
    if output_tokens_mode not in ("predicted_only", "oracle_shape"):
        raise ValueError(
            f"output_tokens_mode must be 'predicted_only' or 'oracle_shape', "
            f"got {output_tokens_mode!r}"
        )
    numeric = tuple(PREDICT_TIME_NUMERIC_FEATURES)
    if output_tokens_mode == "oracle_shape":
        numeric = numeric + ("actual_output_tokens",)
    categorical = tuple(PREDICT_TIME_CATEGORICAL_FEATURES)
    _assert_no_leakage(numeric, mode=output_tokens_mode)
    levels = fit_categorical_levels(rows, categorical, mode=output_tokens_mode)
    return FeatureSpec(
        output_tokens_mode=output_tokens_mode,
        categorical_levels=levels,
        numeric_columns=numeric,
        categorical_columns=categorical,
    )


def build_feature_matrix(rows: list, spec: FeatureSpec):
    """Build (X, feature_names, group_keys) from rows under ``spec``.

    ``group_keys`` is a dict of arrays {column_name: np.array} for every
    categorical column we want to use for subgroup metrics + holdout
    splitting (instance_type, gpu_type, model_size, prompt_token_bin,
    queue_depth_bin, kv_util_bin).
    """

    _assert_no_leakage(spec.numeric_columns, mode=spec.output_tokens_mode)

    n = len(rows)
    numeric_cols = spec.numeric_columns
    cat_cols = spec.categorical_columns
    cat_levels = spec.categorical_levels

    feature_names: list[str] = list(numeric_cols)
    cat_index: list[tuple] = []
    for c in cat_cols:
        for lvl in cat_levels[c]:
            feature_names.append(f"{c}={lvl}")
        # Sentinel column for unseen levels.
        feature_names.append(f"{c}=__UNSEEN__")
        cat_index.append((c, cat_levels[c]))

    n_features = len(feature_names)
    X = np.zeros((n, n_features), dtype=np.float64)

    group_arrays: dict[str, list] = {c: [] for c in (
        "instance_type", "gpu_type", "model_size",
        "prompt_token_bin", "queue_depth_bin", "kv_util_bin",
    )}

    for i, r in enumerate(rows):
        enriched = _enrich_row(r, mode=spec.output_tokens_mode)
        # Numeric.
        for j, col in enumerate(numeric_cols):
            X[i, j] = _coerce_numeric(enriched.get(col))
        # Categorical one-hot.
        col_offset = len(numeric_cols)
        for c, levels in cat_index:
            v = enriched.get(c)
            v = "missing" if v is None else str(v)
            try:
                idx = levels.index(v)
                X[i, col_offset + idx] = 1.0
            except ValueError:
                # Unseen level -> sentinel.
                X[i, col_offset + len(levels)] = 1.0
            col_offset += len(levels) + 1
        # Group arrays for subgroup metrics.
        for c in group_arrays.keys():
            group_arrays[c].append(enriched.get(c))

    group_keys = {c: np.array(v, dtype=object) for c, v in group_arrays.items()}
    return X, feature_names, group_keys


def extract_target(rows: list, target: str) -> np.ndarray:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    out = np.fromiter(
        (_coerce_numeric(r.get(target)) for r in rows),
        dtype=np.float64, count=len(rows),
    )
    return out


# ---------------------------------------------------------------------------
# Holdout split utilities (deterministic)
# ---------------------------------------------------------------------------


def random_holdout(n: int, *, holdout_frac: float = 0.2,
                   seed: int = 1773889) -> tuple:
    """Deterministic random split. Returns ``(train_idx, holdout_idx)``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_holdout = int(round(n * holdout_frac))
    return np.sort(perm[n_holdout:]), np.sort(perm[:n_holdout])


def holdout_by_group(group_values: np.ndarray, hold_groups: tuple) -> tuple:
    """Hold out every row whose group value is in ``hold_groups``."""
    hold_mask = np.zeros(len(group_values), dtype=bool)
    for g in hold_groups:
        hold_mask |= (group_values == g)
    train_idx = np.where(~hold_mask)[0]
    holdout_idx = np.where(hold_mask)[0]
    return train_idx, holdout_idx


def time_holdout(timestamps: np.ndarray, *, holdout_frac: float = 0.2) -> tuple:
    """Hold out the last ``holdout_frac`` chronologically."""
    n = len(timestamps)
    order = np.argsort(timestamps, kind="stable")
    n_holdout = int(round(n * holdout_frac))
    holdout_idx = np.sort(order[-n_holdout:])
    train_idx = np.sort(order[:-n_holdout])
    return train_idx, holdout_idx
