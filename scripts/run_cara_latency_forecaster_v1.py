#!/usr/bin/env python3
"""Train + evaluate the CARA latency forecaster v1.

Pipeline:

1. Load CARA `train_flat` analysis_sample.jsonl (gitignored locally;
   regenerable via `scripts/audit_cara_swissai_telemetry.py --target-set
   analysis_tier`).
2. Run a schema audit: every column gets a role + field_quality label.
   Fail-closed if any target is missing.
3. Build the feature pipeline (predict-time only, leakage-checked).
4. Train baselines: global p95, per-instance_type p95, per-(model_size,
   gpu_type) p95, queue-depth-bin p95, simple-rule placement score.
5. Train HistGradientBoosting quantile forecasters (p50, p95, p99) for
   TTFT and E2E. Add RandomForestRegressor as a robustness candidate.
6. Evaluate on 3 holdout strategies: random, by_instance_type,
   time_holdout. Compute per-subgroup metrics + tail-underprediction
   safety metrics.
7. Apply up to 3 safe improvements: conservative-multiplier
   calibration, fallback-to-baseline, holdout-by-instance-type
   re-training.
8. Write artefacts:
   - data/external/forecasting/cara_latency_forecaster_v1/schema_audit.json
   - data/external/forecasting/cara_latency_forecaster_v1/model_comparison.json
   - data/external/forecasting/cara_latency_forecaster_v1/feature_importance.json

No scheduler / robust energy engine / controller is touched. No
production claim is made.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from aurelius.forecasting.cara_latency_features import (  # noqa: E402
    LEAKAGE_TARGET_FIELDS,
    PREDICT_TIME_NUMERIC_FEATURES,
    TARGETS,
    build_feature_matrix,
    build_feature_spec,
    extract_target,
    holdout_by_group,
    random_holdout,
    time_holdout,
)
from aurelius.forecasting.cara_latency_forecaster import (  # noqa: E402
    ConservativeMultiplierCalibration,
    GlobalConstantP95Baseline,
    GroupConstantQuantileBaseline,
    HistGradientBoostingQuantileForecaster,
    RandomForestMedianForecaster,
    SimpleRulePlacementScoreBaseline,
    classify_gate_status,
    incremental_alpha_pct,
    quantile_metrics,
    subgroup_metrics,
)

logger = logging.getLogger(__name__)


CARA_TRAIN_FLAT = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_flat" / "processed" / "analysis_sample.jsonl"
)

OUT_DIR = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_latency_forecaster_v1"
)


HOLDOUT_BY_INSTANCE_TYPE = ("qwen2.5-7b_a30",)  # held out for OOD test


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# Schema audit
# ---------------------------------------------------------------------------


def _audit_schema(rows: list[dict]) -> dict:
    """Build the schema_audit.json artefact described in PHASE 0."""
    n = len(rows)
    counts: dict = {}
    dtypes: dict = {}
    examples: dict = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            counts[k] = counts.get(k, 0) + (1 if v is not None else 0)
            if k not in dtypes and v is not None:
                dtypes[k] = type(v).__name__
                examples[k] = v if not isinstance(v, str) else v[:80]

    audit: list[dict] = []
    role_map: dict = {}

    for col in sorted(counts.keys()):
        non_null = counts[col]
        null = n - non_null
        role = "ignored"
        field_quality = "real"
        if col in TARGETS:
            role = "target"
        elif col in LEAKAGE_TARGET_FIELDS:
            role = "ignored"
            field_quality = "real"  # the field IS real, just leakage
        elif col in PREDICT_TIME_NUMERIC_FEATURES:
            role = "feature"
        elif col == "instance_type":
            role = "feature"
        elif col in ("instance_id", "request_id"):
            role = "group"
        elif col == "prediction_timestamp_s":
            role = "feature"  # via hour_of_day derivation
        elif col in ("probe_latency_ms", "prediction_latency_ms"):
            # These are out-of-band overheads (not request latency); kept
            # as ``ignored`` rather than features.
            role = "ignored"
        else:
            role = "ignored"

        notes = ""
        if col in LEAKAGE_TARGET_FIELDS:
            notes = "Leakage at decision time; never used as feature."
        elif col == "actual_output_tokens":
            notes = ("Leakage at decision time. Used only by the "
                     "oracle_shape variant, which is analysis_only.")

        role_map[col] = role
        audit.append({
            "column": col,
            "source_config": "train_flat",
            "dtype": dtypes.get(col, "null"),
            "non_null_count": non_null,
            "null_count": null,
            "presence_rate": round(non_null / max(1, n), 6),
            "example_value": examples.get(col, None),
            "role": role,
            "field_quality": field_quality,
            "notes": notes,
        })

    # Required: every target must be present + non-null on every row.
    target_rows_missing: dict = {}
    for t in TARGETS:
        target_rows_missing[t] = n - counts.get(t, 0)
        if counts.get(t, 0) == 0:
            raise SystemExit(
                f"FAIL-CLOSED: target column '{t}' missing from CARA "
                f"analysis sample; refusing to train"
            )

    return {
        "doc_version": "cara_latency_forecaster_v1_schema_audit",
        "source_path": str(CARA_TRAIN_FLAT.relative_to(REPO_ROOT)),
        "row_count": n,
        "audited_at_s": time.time(),
        "columns": audit,
        "role_counts": {
            role: sum(1 for c in audit if c["role"] == role)
            for role in ("feature", "target", "group", "ignored", "missing")
        },
        "target_rows_missing": target_rows_missing,
        "leakage_fields_blocked": sorted(LEAKAGE_TARGET_FIELDS),
    }


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------


def _fit_group_baselines_per_quantile(X_train, y_train, *, group_train):
    """Fit one GroupConstantQuantileBaseline per (group, quantile)."""
    ms_gpu_train = np.array(
        [f"{m}|{g}" for m, g in zip(group_train["model_size"],
                                    group_train["gpu_type"])],
        dtype=object,
    )
    baselines: dict = {}
    for q in (50, 95, 99):
        baselines[("global_constant", q)] = GlobalConstantP95Baseline(
            quantile=q).fit(X_train, y_train)
        baselines[("per_instance_type", q)] = (
            GroupConstantQuantileBaseline(quantile=q).fit(
                X_train, y_train,
                group_keys_train=group_train["instance_type"],
            )
        )
        baselines[("per_model_gpu", q)] = (
            GroupConstantQuantileBaseline(quantile=q).fit(
                X_train, y_train, group_keys_train=ms_gpu_train,
            )
        )
        baselines[("queue_depth_bin", q)] = (
            GroupConstantQuantileBaseline(quantile=q).fit(
                X_train, y_train,
                group_keys_train=group_train["queue_depth_bin"],
            )
        )

    # Simple-rule placement-score baseline (only at p95 — it is a
    # routing scorer, not a quantile estimator).
    qd_train = np.array(
        [_qd_bin_to_midpoint(g) for g in group_train["queue_depth_bin"]],
        dtype=np.float64,
    )
    baselines[("simple_rule_placement_score", 95)] = (
        SimpleRulePlacementScoreBaseline(quantile=95.0).fit(
            X_train, y_train,
            instance_types_train=group_train["instance_type"],
            queue_depths_train=qd_train,
        )
    )
    return baselines, ms_gpu_train


def _baseline_predict(baseline_key, baseline, X_holdout, *, group_holdout):
    name, _q = baseline_key
    if name == "global_constant":
        return baseline.predict(X_holdout)
    if name == "per_instance_type":
        return baseline.predict(
            X_holdout, group_keys_predict=group_holdout["instance_type"],
        )
    if name == "per_model_gpu":
        ms_gpu_holdout = np.array(
            [f"{m}|{g}" for m, g in zip(
                group_holdout["model_size"], group_holdout["gpu_type"])],
            dtype=object,
        )
        return baseline.predict(X_holdout, group_keys_predict=ms_gpu_holdout)
    if name == "queue_depth_bin":
        return baseline.predict(
            X_holdout, group_keys_predict=group_holdout["queue_depth_bin"],
        )
    if name == "simple_rule_placement_score":
        qd_holdout = np.array(
            [_qd_bin_to_midpoint(g) for g in group_holdout["queue_depth_bin"]],
            dtype=np.float64,
        )
        return baseline.predict(
            X_holdout,
            instance_types_predict=group_holdout["instance_type"],
            queue_depths_predict=qd_holdout,
        )
    raise ValueError(f"unknown baseline {name}")


def _evaluate_baselines_per_quantile(
    baselines, X_holdout, y_holdout, *, group_holdout,
) -> dict:
    out: dict = {}
    for (name, q), b in baselines.items():
        pred = _baseline_predict((name, q), b, X_holdout, group_holdout=group_holdout)
        out[f"{name}_p{q}"] = quantile_metrics(y_holdout, pred,
                                               quantile=q / 100.0)
    return out


def _qd_bin_to_midpoint(label) -> float:
    # Map "[0,1)" -> 0.5, "[1,5)" -> 3.0 etc. for the simple-rule baseline.
    mapping = {
        "[0,1)": 0.5, "[1,5)": 3.0, "[5,20)": 12.0,
        "[20,100)": 60.0, "[100,1000000)": 500.0,
    }
    return mapping.get(str(label), 0.0)


def _train_ml(X_train, y_train, *, quantile):
    m = HistGradientBoostingQuantileForecaster(quantile=quantile / 100.0)
    m.fit(X_train, y_train)
    return m


def _ml_eval_per_quantile(
    X_train, y_train, X_holdout, y_holdout,
) -> tuple[dict, dict]:
    """Train HGB quantile p50/p95/p99 + RF median + a safety variant.

    Returns ``(metrics_per_quantile, models)`` where ``metrics_per_quantile``
    is keyed by ``f"hgb_quantile_p{q}"`` so callers can compare apples-to-
    apples against the matching per-(group) p{q} baseline.
    """
    models: dict = {}
    metrics: dict = {}

    for q in (50, 95, 99):
        m = _train_ml(X_train, y_train, quantile=q)
        models[f"hgb_quantile_p{q}"] = m
        metrics[f"hgb_quantile_p{q}"] = quantile_metrics(
            y_holdout, m.predict(X_holdout), quantile=q / 100.0,
        )

    rf = RandomForestMedianForecaster().fit(X_train, y_train)
    models["random_forest_median"] = rf
    # RF mean predictor: compare at q=0.5 (closest analogue).
    metrics["random_forest_median"] = quantile_metrics(
        y_holdout, rf.predict(X_holdout), quantile=0.5,
    )

    # Safe-variant: conservative-multiplier calibration on the p95 model.
    p95_base = models["hgb_quantile_p95"]
    cal = ConservativeMultiplierCalibration(multiplier=1.1, base=p95_base)
    metrics["hgb_p95_x1_10_safety"] = quantile_metrics(
        y_holdout, cal.predict(X_holdout), quantile=0.95,
    )
    models["hgb_p95_x1_10_safety"] = cal

    return metrics, models


def _build_subgroup_metrics(
    y_holdout, y_pred, group_holdout,
) -> dict:
    return {
        subgroup: subgroup_metrics(y_holdout, y_pred, group_holdout[subgroup])
        for subgroup in (
            "instance_type", "gpu_type", "model_size",
            "prompt_token_bin", "queue_depth_bin", "kv_util_bin",
        )
    }


def _feature_importance(model, feature_names) -> Optional[dict]:
    underlying = getattr(model, "_model", None)
    if underlying is None:
        return None
    if hasattr(underlying, "feature_importances_"):
        importances = underlying.feature_importances_
        order = np.argsort(importances)[::-1]
        return {
            "method": type(underlying).__name__ + ".feature_importances_",
            "top": [
                {"feature": feature_names[i],
                 "importance": float(importances[i])}
                for i in order[:40]
            ],
        }
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


from typing import Optional  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=None,
                   help="Cap on rows loaded (for fast smoke runs).")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not CARA_TRAIN_FLAT.exists():
        print(f"[forecaster] FAIL: CARA train_flat analysis sample not "
              f"available at {CARA_TRAIN_FLAT}. Re-run "
              "`scripts/audit_cara_swissai_telemetry.py --target-set "
              "analysis_tier` to regenerate.", file=sys.stderr)
        return 2

    print(f"[forecaster] loading {CARA_TRAIN_FLAT}")
    rows = _load_jsonl(CARA_TRAIN_FLAT, limit=args.limit_rows)
    print(f"[forecaster] loaded {len(rows)} rows")

    # Phase 0: schema audit (fail-closed on missing targets).
    audit = _audit_schema(rows)
    schema_path = out_dir / "schema_audit.json"
    schema_path.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(f"[forecaster] schema audit -> {schema_path} "
          f"(features={audit['role_counts']['feature']}, "
          f"targets={audit['role_counts']['target']})")

    # Phase 2: feature pipeline.
    spec = build_feature_spec(rows, output_tokens_mode="predicted_only")
    X, feature_names, group_keys = build_feature_matrix(rows, spec)
    print(f"[forecaster] feature matrix: {X.shape}; "
          f"categorical levels per column: "
          f"{ {c: len(spec.categorical_levels[c]) for c in spec.categorical_columns} }")

    # Holdouts.
    holdouts = {}
    rng_train, rng_holdout = random_holdout(X.shape[0])
    holdouts["random_holdout"] = (rng_train, rng_holdout)

    by_it_train, by_it_holdout = holdout_by_group(
        group_keys["instance_type"], hold_groups=HOLDOUT_BY_INSTANCE_TYPE,
    )
    if by_it_holdout.size > 0:
        holdouts["holdout_by_instance_type"] = (by_it_train, by_it_holdout)
    else:
        print(f"[forecaster] holdout_by_instance_type group "
              f"{HOLDOUT_BY_INSTANCE_TYPE} not present; skipped")

    # time_holdout uses prediction_timestamp_s if present.
    ts = np.array([r.get("prediction_timestamp_s") for r in rows],
                  dtype=np.float64)
    if np.isfinite(ts).sum() >= 0.95 * len(rows):
        th_train, th_holdout = time_holdout(ts)
        holdouts["time_holdout"] = (th_train, th_holdout)
    else:
        print("[forecaster] time_holdout skipped — sparse "
              "prediction_timestamp_s")

    # Per-target training + evaluation.
    model_comparison: dict = {
        "doc_version": "cara_latency_forecaster_v1_model_comparison",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "trains_ml_models": True,
        "shadow_only": True,
        "row_count": X.shape[0],
        "holdouts_used": sorted(holdouts.keys()),
        "feature_count": X.shape[1],
        "feature_names_head": feature_names[:30],
        "leakage_blocked": sorted(LEAKAGE_TARGET_FIELDS),
        "per_target": {},
        "incremental_alpha_summary": {},
        "gate_classifications": {},
    }
    feature_importance: dict = {"per_target": {}}

    for target in TARGETS:
        y = extract_target(rows, target)
        print(f"\n[forecaster] === target={target} ===")
        per_target = {"per_holdout": {}}
        per_quantile_alpha: dict = {q: {} for q in (50, 95, 99)}
        gate_per_holdout: dict = {}

        for holdout_name, (train_idx, holdout_idx) in holdouts.items():
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_ho, y_ho = X[holdout_idx], y[holdout_idx]
            g_tr = {k: v[train_idx] for k, v in group_keys.items()}
            g_ho = {k: v[holdout_idx] for k, v in group_keys.items()}

            baselines_fitted, _ = _fit_group_baselines_per_quantile(
                X_tr, y_tr, group_train=g_tr,
            )
            base_metrics = _evaluate_baselines_per_quantile(
                baselines_fitted, X_ho, y_ho, group_holdout=g_ho,
            )
            ml_metrics, ml_models = _ml_eval_per_quantile(
                X_tr, y_tr, X_ho, y_ho,
            )

            # Apples-to-apples per-quantile comparison: HGB p{q} vs
            # per_instance_type p{q} on pinball loss. Pinball is the right
            # metric for a quantile predictor.
            per_quantile_gate: dict = {}
            for q in (50, 95, 99):
                ml_key = f"hgb_quantile_p{q}"
                baseline_key = f"per_instance_type_p{q}"
                ml = ml_metrics[ml_key]
                base = base_metrics[baseline_key]
                alpha = incremental_alpha_pct(
                    base["pinball_loss"], ml["pinball_loss"],
                    lower_is_better=True,
                )
                base_tail = base["severe_underprediction_rate"]
                ml_tail = ml["severe_underprediction_rate"]
                # Allow a small tail-underprediction slack at q=50 because
                # baselines at q=50 are not designed for safety; at q=95/99
                # we require ml_tail <= base_tail.
                slack = 0.005 if q == 50 else 0.0
                safety_reg = 1 if ml_tail > base_tail + slack else 0
                gate = classify_gate_status(
                    alpha, tail_underpred_rate=ml_tail,
                    baseline_tail_underpred_rate=base_tail + slack,
                    safety_regression_count=safety_reg,
                )
                per_quantile_gate[q] = {
                    "alpha_pct_pinball_vs_per_instance_type": round(alpha, 4),
                    "ml_pinball_loss": ml["pinball_loss"],
                    "baseline_pinball_loss": base["pinball_loss"],
                    "ml_calibration_coverage": ml["calibration_coverage"],
                    "baseline_calibration_coverage": base["calibration_coverage"],
                    "ml_severe_underprediction_rate": ml_tail,
                    "baseline_severe_underprediction_rate": base_tail,
                    "safety_regression": safety_reg,
                    "gate_classification": gate,
                }
                per_quantile_alpha[q][holdout_name] = round(alpha, 4)

            gate_per_holdout[holdout_name] = per_quantile_gate

            # Subgroup metrics for the p95 model (used by the routing
            # backtest).
            p95_model = ml_models["hgb_quantile_p95"]
            p95_pred = p95_model.predict(X_ho)
            sub = _build_subgroup_metrics(y_ho, p95_pred, g_ho)

            per_target["per_holdout"][holdout_name] = {
                "n_train": int(X_tr.shape[0]),
                "n_holdout": int(X_ho.shape[0]),
                "baselines": base_metrics,
                "ml_models": ml_metrics,
                "subgroup_metrics_for_hgb_p95": sub,
                "per_quantile_gate": per_quantile_gate,
            }

            if holdout_name == "random_holdout":
                fi = _feature_importance(p95_model, feature_names)
                if fi:
                    feature_importance["per_target"][target] = fi

        # Consolidated per-quantile gate: candidate_for_shadow_integration
        # requires the same quantile to clear in >=2 holdouts.
        per_q_consolidated: dict = {}
        for q in (50, 95, 99):
            gates_observed = [
                gate_per_holdout[h][q]["gate_classification"]
                for h in gate_per_holdout
            ]
            cs = (gates_observed.count("candidate_for_shadow_integration")
                  + gates_observed.count("strong_candidate"))
            if cs >= 2:
                per_q_consolidated[q] = "candidate_for_shadow_integration"
            elif "strong_candidate" in gates_observed:
                per_q_consolidated[q] = "promising_needs_validation"
            elif gates_observed.count("promising_needs_validation") >= 2:
                per_q_consolidated[q] = "promising_needs_validation"
            else:
                per_q_consolidated[q] = "diagnostic_only"

        per_target["consolidated_gate_classification_per_quantile"] = (
            per_q_consolidated
        )
        model_comparison["per_target"][target] = per_target
        model_comparison["incremental_alpha_summary"][target] = per_quantile_alpha
        model_comparison["gate_classifications"][target] = per_q_consolidated

    # Write artefacts.
    cmp_path = out_dir / "model_comparison.json"
    cmp_path.write_text(json.dumps(model_comparison, indent=2, sort_keys=True))
    fi_path = out_dir / "feature_importance.json"
    fi_path.write_text(json.dumps(feature_importance, indent=2, sort_keys=True))
    print(f"\n[forecaster] model comparison -> {cmp_path}")
    print(f"[forecaster] feature importance -> {fi_path}")

    print("\n[forecaster] per-quantile gate classifications:")
    for target, per_q in model_comparison["gate_classifications"].items():
        print(f"  {target}:")
        for q, gate in per_q.items():
            print(f"    p{q:>2}  gate={gate}")
            for holdout, alpha in model_comparison[
                    "incremental_alpha_summary"][target][q].items():
                print(f"      {holdout:30s}  alpha%={alpha:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
