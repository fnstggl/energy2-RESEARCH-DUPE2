"""Tests for the expanded canonical-environment ValidationSuite (Phase 1).

Covers the distribution metrics, the PASS/WARN/FAIL/SKIPPED check builders, the
honesty cap + SKIPPED handling in run_validation, the full breadth of source
checks (Azure held-out, v2026 fleet vs FULL_TRACE_EXACT artifacts, Mooncake KV
train/holdout, electricity), and the FleetPlane full-trace anchoring. Fixture-only
(no network); the full-trace path runs whenever the committed v2026 artifacts are
present (they are, in-repo).
"""

from __future__ import annotations

import os

from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane, full_trace_marginals
from aurelius.environment.schemas import HEURISTIC, TRACE_DERIVED, CalibratedParam
from aurelius.environment.validation_suite import (
    FAIL,
    MATCHES_HELDOUT,
    NOT_PRODUCTION_REALISTIC_YET,
    PASS,
    SKIPPED,
    category_mix_l1,
    check_category_mix,
    check_samples,
    check_summary,
    hist_l1_counts,
    ks_statistic,
    rel_err,
    run_validation,
    skipped_check,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROCESSED = os.path.join(_REPO, "data", "external", "alibaba_gpu_v2026", "processed")
_MOONCAKE = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_sample.csv")
_HAS_ARTIFACTS = os.path.exists(os.path.join(_PROCESSED, "pod_hourly_calibration.json"))


# --- metrics ---------------------------------------------------------------

def test_metrics_basic():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert category_mix_l1({"a": 8, "b": 2}, {"a": 8, "b": 2}) == 0.0
    assert abs(category_mix_l1({"a": 10}, {"b": 10}) - 1.0) < 1e-9      # disjoint → TV 1
    assert hist_l1_counts([1, 1, 1], [1, 1, 1]) == 0.0
    assert rel_err(11.0, 10.0) == 0.1 and rel_err(0.0, 0.0) == 0.0


# --- check builders + verdicts ---------------------------------------------

def test_check_builders_verdicts():
    near = check_samples("x", [1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5],
                         source="s", ref_tier="t")
    assert near.verdict == PASS and near.mode == "samples"
    far = check_summary("y", {"mean": 100.0}, {"mean": 10.0}, source="s", ref_tier="t")
    assert far.verdict == FAIL and far.metric_name == "max_rel_err"
    mix = check_category_mix("z", {"a": 1, "b": 1}, {"a": 1, "b": 1}, source="s", ref_tier="t")
    assert mix.verdict == PASS
    sk = skipped_check("w", source="s", required_artifact="thing", command="run X", reason="absent")
    assert sk.verdict == SKIPPED and "run X" in sk.detail and sk.ref_tier == "UNAVAILABLE"


def test_run_validation_honesty_cap_and_skip():
    safe = [CalibratedParam("p", 1, "d", "c", "m", "s", "v", TRACE_DERIVED)]
    unsafe = [CalibratedParam("p", 1, "d", "c", "m", "s", "v", HEURISTIC)]
    passing = check_summary("k", {"mean": 1.0}, {"mean": 1.0}, source="s", ref_tier="t")
    skipped = skipped_check("k2", source="s", required_artifact="a", command="c")
    # all evaluated pass + headline-safe → MATCHES_HELDOUT; SKIPPED counted, not a pass
    rep = run_validation([passing, skipped], safe)
    assert rep.overall_verdict == MATCHES_HELDOUT
    assert rep.counts == {"pass": 1, "warn": 0, "fail": 0, "skipped": 1, "total": 2}
    # a HEURISTIC param caps the verdict no matter the checks
    assert run_validation([passing], unsafe).overall_verdict == NOT_PRODUCTION_REALISTIC_YET
    # all-skipped → not enough evidence → not MATCHES_HELDOUT
    assert run_validation([skipped], safe).overall_verdict == NOT_PRODUCTION_REALISTIC_YET


# --- FleetPlane full-trace anchoring ---------------------------------------

def test_fleet_sample_mode_unanchored():
    fp = V2026FleetPlane()                      # no processed_dir → legacy sample behaviour
    assert fp.full_trace is None
    assert fp.full_trace_params() == []
    assert 0.0 < fp.state_at(0).util_target < 1.0


def test_fleet_anchored_consumes_full_trace():
    if not _HAS_ARTIFACTS:
        return
    fp = V2026FleetPlane(processed_dir=_PROCESSED)
    assert fp.full_trace is not None
    params = fp.full_trace_params()
    assert params and all(p.tier == TRACE_DERIVED and p.safe_for_headline for p in params)
    # anchored fleet state reproduces the full-trace util marginal (mean 27.5% → 0.275)
    s = fp.state_at(0)
    assert abs(s.util_target - fp.full_trace["util_mean_pct"] / 100.0) < 1e-9
    assert s.fidelity["util_target"] == "FULL_TRACE_EXACT"
    assert s.fidelity["capacity_envelope"] == "SAMPLE_FIXTURE"   # topology honest


# --- mooncake KV + electricity builders ------------------------------------

def test_mooncake_kv_checks_pass_on_sample():
    from aurelius.environment.validators import mooncake_kv_checks
    checks = mooncake_kv_checks()
    kinds = {c.kind: c.verdict for c in checks}
    assert "kv_exact_prefix_reuse" in kinds and "kv_cold_vs_warm" in kinds
    # train vs holdout reuse generalizes on the sample → the reuse checks pass
    assert kinds["kv_exact_prefix_reuse"] == PASS


def test_electricity_checks_sanity_and_skip():
    from aurelius.environment.validators import electricity_checks
    checks = electricity_checks(V2026FleetPlane())
    kinds = {c.kind: c for c in checks}
    assert kinds["electricity_price_sanity"].verdict == PASS
    assert kinds["electricity_price_sanity"].ref_tier == "SAMPLE_FIXTURE"
    assert kinds["electricity_heldout_iso"].verdict == SKIPPED


# --- full breadth through the environment ----------------------------------

def _env(processed_dir=None):
    from aurelius.environment.canonical import CanonicalMultiPlaneEnvironment
    env = CanonicalMultiPlaneEnvironment(mooncake_path=_MOONCAKE, processed_dir=processed_dir)
    env.calibrate([(float(i) * 0.5, 100 + (i * 7) % 400) for i in range(200)])
    return env


def test_env_validation_breadth_and_tiers():
    env = _env(processed_dir=_PROCESSED if _HAS_ARTIFACTS else None)
    d = env.validate().to_dict()
    kinds = {c["kind"] for c in d["checks"]}
    # breadth: every required plane is represented
    for required in ("azure_token_distribution", "azure_interarrival",
                     "v2026_gpu_utilization", "v2026_priority_mix", "v2026_queue_delay",
                     "kv_exact_prefix_reuse", "kv_cold_vs_warm", "electricity_price_sanity"):
        assert required in kinds, required
    # held-out Azure + Mooncake pass; every SKIPPED names its required artifact/command
    by = {c["kind"]: c for c in d["checks"]}
    assert by["azure_token_distribution"]["verdict"] == PASS
    for c in d["checks"]:
        if c["verdict"] == SKIPPED:
            assert "requires:" in c["detail"] and "enable with:" in c["detail"]
    # honesty cap holds (cost params are HEURISTIC) regardless of check results
    assert d["overall_verdict"] == NOT_PRODUCTION_REALISTIC_YET


def test_env_anchored_v2026_consistency_passes():
    if not _HAS_ARTIFACTS:
        return
    env = _env(processed_dir=_PROCESSED)
    by = {c["kind"]: c for c in env.validate().to_dict()["checks"]}
    # anchored: env reproduces the full-trace marginals (consistency)
    assert by["v2026_gpu_utilization"]["verdict"] == PASS
    assert by["v2026_priority_mix"]["verdict"] == PASS
    assert "CONSISTENCY" in by["v2026_gpu_utilization"]["detail"]
    # full-trace marginals are actually loaded
    assert full_trace_marginals(_PROCESSED) is not None
