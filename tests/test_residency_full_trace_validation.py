"""Validation tests for the Model Residency Decision Engine full-trace audit.

Measurement-only / artifact-validation: these assert the committed full-trace
validation JSON + markdown are present, internally consistent, honest about the
finding (the engine did NOT beat constraint_aware on KPI — it ties the strongest
residency-blind baseline), and free of unhedged production-savings claims. They
do NOT require the (gitignored) raw trace — they validate the committed report,
the way ``test_model_residency_readiness_audit.py`` validates its summary.
"""

from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(REPO_ROOT, "data", "external", "alibaba_genai", "processed",
                    "model_residency_full_trace_validation.json")
MD = os.path.join(REPO_ROOT, "docs", "MODEL_RESIDENCY_FULL_TRACE_VALIDATION.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def _load():
    assert os.path.exists(JSON), "run scripts/run_residency_full_trace_validation.py"
    return json.load(open(JSON))


def test_validation_json_exists_and_structured():
    d = _load()
    for key in ("question", "primary_kpi", "verdict", "per_pool_verdict",
                "per_request_residency_full_trace", "tick_based_ablation_full_trace",
                "n_requests", "n_models"):
        assert key in d, f"missing key: {key}"
    assert d["primary_kpi"] == "sla_safe_goodput_per_infrastructure_dollar"
    assert d["directional_only_not_production_savings"] is True
    assert d["measurement_only"] is True


def test_per_request_policies_present_with_metrics():
    d = _load()
    pol = d["per_request_residency_full_trace"][f"n_gpus_{d['primary_n_gpus']}"]
    for name in ("fifo_round_robin", "sla_aware_least_queue",
                 "sla_aware_naive_prewarm", "affinity_only", "residency_engine"):
        assert name in pol, f"missing policy {name}"
        s = pol[name]
        for m in ("sla_safe_goodput_per_infra_dollar", "model_residency_hit_rate",
                  "adapter_residency_hit_rate", "cold_start_count", "sla_violations",
                  "warm_pool_gpu_hours", "route_to_resident_count", "prewarm_count",
                  "eviction_count"):
            assert m in s, f"{name} missing metric {m}"


def test_engine_improves_residency_but_not_kpi():
    """The honest measured finding: the engine raises the residency hit-rate and
    cuts cold starts vs residency-blind FIFO, but does NOT beat the strongest
    residency-blind baseline on goodput/$ (it ties within ±1%)."""
    d = _load()
    pol = d["per_request_residency_full_trace"][f"n_gpus_{d['primary_n_gpus']}"]
    eng = pol["residency_engine"]
    fifo = pol["fifo_round_robin"]
    # residency works: higher hit-rate, far fewer cold starts than blind FIFO
    assert eng["model_residency_hit_rate"] > fifo["model_residency_hit_rate"]
    assert eng["cold_start_count"] < fifo["cold_start_count"]
    # but no KPI improvement over the existing/blind baseline
    assert d["verdict"]["kpi_improved_over_existing"] is False
    assert d["verdict"]["constraint_aware_already_captures_affinity_value"] is True


def test_engine_ties_best_blind_baseline_at_every_pool():
    d = _load()
    for ng, pv in d["per_pool_verdict"].items():
        assert pv["classification_vs_best_blind"] in ("TIE", "LOSS"), \
            f"{ng}: unexpected {pv['classification_vs_best_blind']}"
        assert pv["engine_beats_best_blind"] is False
        # within the ±1% tie band
        assert abs(pv["engine_vs_best_blind_margin_pct"]) <= 1.0


def test_engine_beats_naive_affinity_on_safety():
    """vs naive affinity_only the engine keeps a comparable hit-rate but far
    fewer SLA violations (it does affinity SAFELY)."""
    d = _load()
    pol = d["per_request_residency_full_trace"][f"n_gpus_{d['primary_n_gpus']}"]
    eng, aff = pol["residency_engine"], pol["affinity_only"]
    assert eng["sla_violations"] < aff["sla_violations"]
    assert eng["sla_safe_goodput_per_infra_dollar"] >= aff["sla_safe_goodput_per_infra_dollar"]


def test_tick_based_reference_present():
    d = _load()
    tb = d["tick_based_ablation_full_trace"]
    ca = tb.get("constraint_aware", {}).get("sla_safe_goodput_per_infra_dollar")
    ca_no = tb.get("constraint_aware_no_affinity", {}).get(
        "sla_safe_goodput_per_infra_dollar")
    # the affinity value current constraint_aware already captures
    assert ca and ca_no and ca > ca_no


def test_markdown_has_required_sections():
    assert os.path.exists(MD)
    text = open(MD, encoding="utf-8").read().lower()
    for section in ("headline answer", "per-request residency routing",
                    "tick-based ablation", "why the engine adds no incremental kpi",
                    "what remains missing", "alpha vs safety"):
        assert section in text, f"report missing section: {section}"
    assert "no." in text  # KPI improved? NO.


def test_no_unhedged_production_savings_claims():
    text = open(MD, encoding="utf-8").read()
    low = " ".join(text.lower().split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in ("not ", "no ", "never ", "n't ",
                                          "without ")), f"unhedged '{phrase}'"
            i = pos + len(phrase)
