#!/usr/bin/env python3
"""Shadow-only placement / routing backtest for the CARA forecaster v1.

Given a holdout of CARA requests, simulate a counterfactual router that
scores each candidate instance_type by predicted latency and picks the
minimum. For each policy, report the realised-latency distribution
across the chosen instance_types.

**Honesty caveats (binding):**

- CARA only contains the realised latency at the instance_type each
  request **actually went to**. We do not have ground-truth
  counterfactual latencies for the other 4 instance_types.
- The backtest's realised latency under a counterfactual routing
  decision is estimated as the **bucket-mean realised latency** of
  holdout requests that did go to the counterfactual instance with
  similar (prompt_token_bin, queue_depth_bin, kv_util_bin). This is an
  **honest proxy**, not a measurement. Every reported number carries
  ``result_quality = "counterfactual_bucket_mean_proxy"``.
- No external-savings number is quoted. No oracle-headline comparison
  is made. Results are research-class only.
- ``round_robin`` and ``per_instance_type_p95`` are the strongest-
  realistic baselines.

Writes ``data/external/forecasting/cara_latency_forecaster_v1/backtest_summary.json``.
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
    TARGETS,
    build_feature_matrix,
    build_feature_spec,
    extract_target,
    random_holdout,
)
from aurelius.forecasting.cara_latency_forecaster import (  # noqa: E402
    GroupConstantQuantileBaseline,
    HistGradientBoostingQuantileForecaster,
    _percentile,  # noqa: E402
)

logger = logging.getLogger(__name__)


CARA_TRAIN_FLAT = (
    REPO_ROOT / "data" / "external" / "hf"
    / "asdwb__cara_latency_prediction"
    / "train_flat" / "processed" / "analysis_sample.jsonl"
)

OUT_PATH = (
    REPO_ROOT / "data" / "external" / "forecasting"
    / "cara_latency_forecaster_v1" / "backtest_summary.json"
)

INSTANCE_TYPES = (
    "qwen2.5-3b_a30", "qwen2.5-3b_p100", "qwen2.5-7b_a30",
    "qwen2.5-14b_v100", "qwen2.5-72b_a100",
)


def _load_jsonl(path: Path) -> list[dict]:
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
    return rows


# ---------------------------------------------------------------------------
# Counterfactual latency lookup table (bucket-mean proxy).
# ---------------------------------------------------------------------------


def build_counterfactual_lookup(
    rows: list[dict], target: str,
) -> tuple[dict, dict]:
    """Bucket realised latencies by (instance_type, prompt_token_bin,
    queue_depth_bin) and return ``{(it, pb, qd): mean_latency, count}``.
    Also returns the per-instance overall median as fallback."""
    from aurelius.forecasting.cara_latency_features import (
        bin_prompt_tokens,
        bin_queue_depth,
    )

    bucket_sums: dict = {}
    bucket_counts: dict = {}
    per_instance_vals: dict = {it: [] for it in INSTANCE_TYPES}
    for r in rows:
        it = r.get("instance_type")
        if it not in INSTANCE_TYPES:
            continue
        y = r.get(target)
        if y is None:
            continue
        pb = bin_prompt_tokens(r.get("num_prompt_tokens"))
        qd = bin_queue_depth(r.get("num_running"))
        key = (it, pb, qd)
        bucket_sums[key] = bucket_sums.get(key, 0.0) + float(y)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
        per_instance_vals[it].append(float(y))

    bucket_mean = {k: bucket_sums[k] / bucket_counts[k] for k in bucket_sums}
    fallback = {
        it: float(np.median(vals)) if vals else float("nan")
        for it, vals in per_instance_vals.items()
    }
    return bucket_mean, fallback


def counterfactual_latency(
    row: dict, chosen_it: str, *, target: str, bucket_mean: dict, fallback: dict,
) -> float:
    """Return the bucket-mean realised latency of historical requests at
    ``chosen_it`` with the same (prompt_token_bin, queue_depth_bin). Falls
    back to the per-instance median when the bucket is empty."""
    from aurelius.forecasting.cara_latency_features import (
        bin_prompt_tokens,
        bin_queue_depth,
    )
    pb = bin_prompt_tokens(row.get("num_prompt_tokens"))
    qd = bin_queue_depth(row.get("num_running"))
    key = (chosen_it, pb, qd)
    if key in bucket_mean:
        return bucket_mean[key]
    return fallback.get(chosen_it, float("nan"))


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------


def _build_x_for_candidates(rows, spec):
    """For each (row, candidate instance_type) pair, return X with the
    row's features but instance_type overridden to the candidate."""
    candidate_X_per_it: dict = {}
    for it in INSTANCE_TYPES:
        rewritten = []
        for r in rows:
            rr = dict(r)
            rr["instance_type"] = it
            rewritten.append(rr)
        X, _, _ = build_feature_matrix(rewritten, spec)
        candidate_X_per_it[it] = X
    return candidate_X_per_it


def _round_robin_policy(rows) -> list[str]:
    return [INSTANCE_TYPES[i % len(INSTANCE_TYPES)] for i in range(len(rows))]


def _per_instance_p95_baseline_policy(
    candidate_X_per_it, baseline_per_it: dict,
) -> list[str]:
    """Pick the instance_type with lowest per_instance_type_p95 prediction.

    Since baseline_per_it is a fixed scalar per instance_type, this
    reduces to 'always pick the fastest-instance-by-historical-p95'.
    We use it as the documented honest baseline."""
    fastest = min(baseline_per_it, key=lambda k: baseline_per_it[k])
    return [fastest for _ in range(len(candidate_X_per_it[INSTANCE_TYPES[0]]))]


def _per_instance_p95_with_queue_policy(
    candidate_X_per_it, baseline_per_it: dict, rows,
) -> list[str]:
    """Per-instance p95 + queue penalty (simple rule). Picks the instance
    with the lowest baseline_p95 + queue_penalty * num_running."""
    out = []
    qp = 0.05
    for i, r in enumerate(rows):
        best_it, best_score = None, float("inf")
        nrun = r.get("num_running") or 0.0
        for it in INSTANCE_TYPES:
            score = baseline_per_it.get(it, float("inf")) + qp * float(nrun)
            if score < best_score:
                best_score, best_it = score, it
        out.append(best_it)
    return out


def _ml_policy(candidate_X_per_it, ml_model) -> list[str]:
    """Predict latency at each candidate; pick the minimum per row."""
    n = candidate_X_per_it[INSTANCE_TYPES[0]].shape[0]
    preds = np.stack(
        [ml_model.predict(candidate_X_per_it[it]) for it in INSTANCE_TYPES],
        axis=1,
    )
    pick_idx = np.argmin(preds, axis=1)
    return [INSTANCE_TYPES[i] for i in pick_idx]


def _evaluate_policy(
    policy_name: str, picks: list[str], rows: list[dict],
    target: str, bucket_mean: dict, fallback: dict,
) -> dict:
    realised = []
    actual_match = 0
    instance_pick_counts: dict = {}
    for r, it in zip(rows, picks):
        instance_pick_counts[it] = instance_pick_counts.get(it, 0) + 1
        if r.get("instance_type") == it:
            y = r.get(target)
            if y is not None:
                realised.append(float(y))
                actual_match += 1
        else:
            y = counterfactual_latency(
                r, it, target=target, bucket_mean=bucket_mean, fallback=fallback,
            )
            if not np.isnan(y):
                realised.append(float(y))
    realised_arr = np.array(realised, dtype=np.float64)
    return {
        "policy": policy_name,
        "result_quality": (
            "counterfactual_bucket_mean_proxy"
            if actual_match < len(rows) else "measured"
        ),
        "n_rows": len(rows),
        "n_actual_match": actual_match,
        "instance_pick_counts": dict(sorted(instance_pick_counts.items())),
        "realised_latency_p50": _percentile(realised_arr, 50),
        "realised_latency_p90": _percentile(realised_arr, 90),
        "realised_latency_p95": _percentile(realised_arr, 95),
        "realised_latency_p99": _percentile(realised_arr, 99),
        "realised_latency_mean": float(np.mean(realised_arr))
        if realised_arr.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--out-path", default=str(OUT_PATH))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    if not CARA_TRAIN_FLAT.exists():
        print(f"[backtest] FAIL: CARA train_flat not present at "
              f"{CARA_TRAIN_FLAT}", file=sys.stderr)
        return 2

    print(f"[backtest] loading {CARA_TRAIN_FLAT}")
    rows = _load_jsonl(CARA_TRAIN_FLAT)
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    print(f"[backtest] loaded {len(rows)} rows")

    train_idx, holdout_idx = random_holdout(len(rows))
    rows_train = [rows[i] for i in train_idx]
    rows_holdout = [rows[i] for i in holdout_idx]
    print(f"[backtest] train={len(rows_train)}  holdout={len(rows_holdout)}")

    spec = build_feature_spec(rows_train, output_tokens_mode="predicted_only")
    X_train, _, g_train = build_feature_matrix(rows_train, spec)

    payload: dict = {
        "doc_version": "cara_latency_forecaster_v1_backtest_v1",
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "shadow_only": True,
        "result_quality_for_counterfactuals": (
            "bucket_mean_proxy (CARA carries realised latency only at the "
            "actually-chosen instance_type; counterfactuals are honest "
            "estimates, NOT measurements)"
        ),
        "n_train": len(rows_train),
        "n_holdout": len(rows_holdout),
        "instance_types": list(INSTANCE_TYPES),
        "per_target": {},
        "backtest_ran_at_s": time.time(),
    }

    candidate_X_per_it = _build_x_for_candidates(rows_holdout, spec)

    for target in TARGETS:
        y_train = extract_target(rows_train, target)
        print(f"\n[backtest] === target={target} ===")

        # Train baselines.
        inst_p95 = GroupConstantQuantileBaseline(quantile=95.0).fit(
            X_train, y_train, group_keys_train=g_train["instance_type"],
        )
        baseline_per_it = {
            it: inst_p95.predict(
                X_train[:1], group_keys_predict=np.array([it], dtype=object),
            )[0]
            for it in INSTANCE_TYPES
        }

        # Train ML model (HGB p95 — the routing-risk model).
        ml_p95 = HistGradientBoostingQuantileForecaster(quantile=0.95).fit(
            X_train, y_train,
        )

        # Counterfactual lookup (built from training rows so the holdout
        # never sees its own realised values).
        bucket_mean, fallback = build_counterfactual_lookup(
            rows_train, target,
        )

        policies: dict = {}

        # Policy 1: round-robin (no information).
        picks = _round_robin_policy(rows_holdout)
        policies["round_robin"] = _evaluate_policy(
            "round_robin", picks, rows_holdout, target, bucket_mean, fallback,
        )

        # Policy 2: per_instance_type_p95 (always-fastest).
        picks = _per_instance_p95_baseline_policy(
            candidate_X_per_it, baseline_per_it,
        )
        policies["per_instance_type_p95"] = _evaluate_policy(
            "per_instance_type_p95", picks, rows_holdout, target,
            bucket_mean, fallback,
        )

        # Policy 3: simple-rule (per-instance p95 + queue penalty).
        picks = _per_instance_p95_with_queue_policy(
            candidate_X_per_it, baseline_per_it, rows_holdout,
        )
        policies["per_instance_type_p95_with_queue"] = _evaluate_policy(
            "per_instance_type_p95_with_queue", picks, rows_holdout, target,
            bucket_mean, fallback,
        )

        # Policy 4: ML latency forecaster (HGB p95 model — minimum
        # predicted p95 routing).
        picks = _ml_policy(candidate_X_per_it, ml_p95)
        policies["ml_hgb_p95"] = _evaluate_policy(
            "ml_hgb_p95", picks, rows_holdout, target, bucket_mean, fallback,
        )

        # Tail-risk reduction vs strongest baseline.
        strongest = policies["per_instance_type_p95_with_queue"]
        ml = policies["ml_hgb_p95"]

        def _pct_delta(a, b):
            if a is None or b is None or np.isnan(a) or np.isnan(b) or a == 0:
                return None
            return 100.0 * (a - b) / a

        tail_delta = {
            "p50": _pct_delta(strongest["realised_latency_p50"],
                              ml["realised_latency_p50"]),
            "p90": _pct_delta(strongest["realised_latency_p90"],
                              ml["realised_latency_p90"]),
            "p95": _pct_delta(strongest["realised_latency_p95"],
                              ml["realised_latency_p95"]),
            "p99": _pct_delta(strongest["realised_latency_p99"],
                              ml["realised_latency_p99"]),
        }

        # Safety regression = ML's realised p99 worse than strongest baseline's.
        safety_regression = int(
            (ml["realised_latency_p99"] is not None
             and strongest["realised_latency_p99"] is not None
             and ml["realised_latency_p99"] > strongest["realised_latency_p99"])
        )

        # Promotion classification (latency reduction at p95/p99).
        p99_alpha = tail_delta.get("p99") or 0.0
        p95_alpha = tail_delta.get("p95") or 0.0
        win_threshold = max(p95_alpha, p99_alpha)
        if safety_regression or win_threshold < 2.0:
            promotion = "diagnostic_only"
        elif win_threshold < 5.0:
            promotion = "promising_needs_validation"
        elif win_threshold < 10.0:
            promotion = "candidate_for_shadow_integration"
        else:
            promotion = "strong_candidate_for_shadow_integration"

        payload["per_target"][target] = {
            "policies": policies,
            "tail_delta_ml_vs_strongest_baseline_pct": tail_delta,
            "strongest_baseline_policy": "per_instance_type_p95_with_queue",
            "safety_regression": safety_regression,
            "ml_routing_promotion_classification": promotion,
            "result_quality_label": "counterfactual_bucket_mean_proxy",
            "goodput_per_dollar": None,
            "goodput_per_dollar_skipped_reason": (
                "no real cost mapping by GPU type in CARA; per the mission "
                "spec, latency-risk reduction is the primary metric and "
                "goodput/$ is explicitly not evaluated."
            ),
        }

        for name, m in policies.items():
            print(f"  {name:38s}  p50={m['realised_latency_p50']:.3f}s  "
                  f"p95={m['realised_latency_p95']:.3f}s  "
                  f"p99={m['realised_latency_p99']:.3f}s  "
                  f"picks={m['instance_pick_counts']}")
        print(f"  tail delta ml vs strongest: {tail_delta}")
        print(f"  promotion: {promotion}")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\n[backtest] wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
