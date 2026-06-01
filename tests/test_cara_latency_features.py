"""Tests for the CARA latency forecaster feature pipeline."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.forecasting.cara_latency_features import (  # noqa: E402
    LEAKAGE_TARGET_FIELDS,
    PREDICT_TIME_CATEGORICAL_FEATURES,
    PREDICT_TIME_NUMERIC_FEATURES,
    TARGETS,
    bin_kv_util,
    bin_output_tokens,
    bin_prompt_tokens,
    bin_queue_depth,
    build_feature_matrix,
    build_feature_spec,
    derive_gpu_type,
    derive_model_size,
    extract_target,
    holdout_by_group,
    hour_of_day,
    random_holdout,
    time_holdout,
)


def _row(**overrides):
    base = {
        "request_id": "abc", "instance_id": "host:port",
        "instance_type": "qwen2.5-3b_p100",
        "num_prompt_tokens": 47, "num_predicted_output_tokens": 1024,
        "actual_output_tokens": 250, "actual_e2e_latency_s": 23.7,
        "actual_ttft_s": 0.13, "actual_tpot_s": 0.094,
        "num_running": 11, "num_waiting": 0, "num_active_decode_seqs": 0,
        "decode_ctx_p50": 0.0, "decode_ctx_p95": 0.0, "decode_ctx_max": 0.0,
        "pending_prefill_tokens": 0, "pending_decode_tokens": 0,
        "kv_cache_utilization": 0.0, "kv_free_blocks": 6118,
        "token_budget_per_iter": 0, "prefill_chunk_size": 0,
        "max_num_seqs": 0, "num_preempted": 0,
        "ema_decode_tok_per_s": 10.4, "ema_prefill_tok_per_s": 289.0,
        "ema_decode_iter_ms": 96.0, "kv_evictions_per_s": 0.0,
        "running_requests_count": 11, "waiting_requests_count": 0,
        "prediction_timestamp_s": 1773889091.0,
        "completion_timestamp_s": 1773889115.0,
    }
    base.update(overrides)
    return base


# ---------- 1. Leakage prevention -----------------------------------------


def test_leakage_target_fields_cover_every_realised_field():
    expected_leakage = {
        "actual_e2e_latency_s", "actual_ttft_s", "actual_tpot_s",
        "actual_output_tokens", "completion_timestamp_s",
        "actual_e2e_latency", "actual_ttft", "actual_tpot",
        "completion_timestamp",
    }
    assert expected_leakage == set(LEAKAGE_TARGET_FIELDS)


def test_predict_time_numeric_features_have_no_leakage_overlap():
    leak = set(PREDICT_TIME_NUMERIC_FEATURES) & LEAKAGE_TARGET_FIELDS
    assert leak == set(), (
        f"predict-time numeric features overlap leakage: {leak}"
    )


def test_predict_time_categorical_features_have_no_leakage_overlap():
    leak = set(PREDICT_TIME_CATEGORICAL_FEATURES) & LEAKAGE_TARGET_FIELDS
    assert leak == set()


def test_build_feature_spec_rejects_leakage_in_predicted_only_mode():
    rows = [_row()] * 3
    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    for name in spec.numeric_columns:
        assert name not in LEAKAGE_TARGET_FIELDS, (
            f"predicted_only spec emitted leakage column {name!r}"
        )


def test_oracle_shape_mode_only_admits_actual_output_tokens_leakage():
    rows = [_row()] * 3
    spec = build_feature_spec(rows, output_tokens_mode="oracle_shape")
    assert "actual_output_tokens" in spec.numeric_columns
    # The other leakage fields stay blocked.
    other_leak = (LEAKAGE_TARGET_FIELDS - {"actual_output_tokens"}) \
        & set(spec.numeric_columns)
    assert other_leak == set()


# ---------- 2. Derived columns + bin labels ------------------------------


def test_derive_model_size_and_gpu_type():
    assert derive_model_size("qwen2.5-3b_p100") == "3b"
    assert derive_gpu_type("qwen2.5-3b_p100") == "p100"
    assert derive_model_size("qwen2.5-72b_a100") == "72b"
    assert derive_gpu_type("qwen2.5-14b_v100") == "v100"
    assert derive_model_size(None) is None
    assert derive_gpu_type("") is None


def test_bin_helpers_use_preregistered_boundaries():
    # prompt_token_bin
    assert bin_prompt_tokens(0) == "[0,50)"
    assert bin_prompt_tokens(49) == "[0,50)"
    assert bin_prompt_tokens(50) == "[50,200)"
    assert bin_prompt_tokens(5000) == "[3200,1000000)"
    assert bin_prompt_tokens(None) == "missing"
    # queue_depth_bin
    assert bin_queue_depth(0) == "[0,1)"
    assert bin_queue_depth(11) == "[5,20)"
    # kv_util_bin
    assert bin_kv_util(0.0) == "[0.0,0.1)"
    assert bin_kv_util(0.95) == "[0.9,1.01)"
    # output_token_bin
    assert bin_output_tokens(100) == "[64,256)"


def test_hour_of_day_handles_invalid_input():
    assert hour_of_day(None) is None
    assert hour_of_day("not-a-number") is None
    assert isinstance(hour_of_day(1700000000.0), int)


# ---------- 3. Feature matrix shape + determinism ------------------------


def test_feature_matrix_shape_and_levels():
    rows = [_row(instance_type=it) for it in (
        "qwen2.5-3b_p100", "qwen2.5-7b_a30", "qwen2.5-72b_a100",
    )]
    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X, names, groups = build_feature_matrix(rows, spec)
    assert X.shape[0] == 3
    # One-hot + sentinel column per categorical column.
    n_numeric = len(spec.numeric_columns)
    n_cat = sum(len(spec.categorical_levels[c]) + 1
                for c in spec.categorical_columns)
    assert X.shape[1] == n_numeric + n_cat
    assert len(names) == X.shape[1]
    # Group arrays present.
    for g in ("instance_type", "gpu_type", "model_size",
              "prompt_token_bin", "queue_depth_bin", "kv_util_bin"):
        assert g in groups
        assert groups[g].shape[0] == 3


def test_feature_matrix_is_deterministic():
    rows = [_row()] * 5
    spec1 = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X1, names1, _ = build_feature_matrix(rows, spec1)
    spec2 = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X2, names2, _ = build_feature_matrix(rows, spec2)
    assert names1 == names2
    np.testing.assert_array_equal(X1, X2)


def test_feature_matrix_unseen_categorical_routes_to_sentinel():
    train_rows = [_row(instance_type="qwen2.5-3b_p100")] * 3
    spec = build_feature_spec(train_rows, output_tokens_mode="predicted_only")
    holdout_rows = [_row(instance_type="brand_new_instance")]
    X, names, _ = build_feature_matrix(holdout_rows, spec)
    sentinel_idx = names.index("instance_type=__UNSEEN__")
    assert X[0, sentinel_idx] == 1.0


# ---------- 4. Targets + leakage in extract_target ----------------------


def test_extract_target_returns_numpy_array():
    rows = [_row(actual_e2e_latency_s=v) for v in (1.0, 2.0, 3.0)]
    y = extract_target(rows, "actual_e2e_latency_s")
    assert y.shape == (3,)
    np.testing.assert_allclose(y, [1.0, 2.0, 3.0])


def test_extract_target_rejects_invalid_target_name():
    rows = [_row()]
    with pytest.raises(ValueError):
        extract_target(rows, "not_a_target")


def test_targets_only_contains_realised_latency_fields():
    assert set(TARGETS) == {"actual_ttft_s", "actual_e2e_latency_s"}
    for t in TARGETS:
        assert t in LEAKAGE_TARGET_FIELDS


# ---------- 5. Holdout split determinism ---------------------------------


def test_random_holdout_is_deterministic_and_disjoint():
    a_train, a_holdout = random_holdout(100, seed=42)
    b_train, b_holdout = random_holdout(100, seed=42)
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_holdout, b_holdout)
    # Different seed -> different split.
    c_train, c_holdout = random_holdout(100, seed=43)
    assert not np.array_equal(a_holdout, c_holdout)
    # Disjoint.
    assert set(a_train.tolist()) & set(a_holdout.tolist()) == set()


def test_holdout_by_group_isolates_named_groups():
    g = np.array(["a", "a", "b", "c", "b", "a"], dtype=object)
    train, holdout = holdout_by_group(g, hold_groups=("b",))
    assert set(g[holdout].tolist()) == {"b"}
    assert "b" not in g[train]


def test_time_holdout_uses_chronological_tail():
    ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    train, holdout = time_holdout(ts, holdout_frac=0.3)
    assert ts[holdout].min() > ts[train].max()
