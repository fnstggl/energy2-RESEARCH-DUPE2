"""End-to-end tests for the self-serve pipeline, report, and fingerprint.

Covers the three outcomes a prospect can hit — replay runs, replay refused,
file unreadable — plus the two artifact contracts: the report must satisfy
the docs/RESULTS.md §8 claim rules, and the fingerprint must not leak log
values.
"""

from __future__ import annotations

import json

import pytest

from aurelius.selfserve.demo import write_demo_jobs_log, write_demo_serving_log
from aurelius.selfserve.pipeline import run_pipeline
from aurelius.selfserve.report import (
    ClaimRuleError,
    check_claim_rules,
    render_html,
    write_report,
)


@pytest.fixture(scope="module")
def serving_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    log = write_demo_serving_log(d / "gw.csv", hours=0.6)
    return run_pipeline(log)


@pytest.fixture(scope="module")
def jobs_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("jobs")
    log = write_demo_jobs_log(d / "sacct.csv", days=4.0)
    return run_pipeline(log, fleet_gpus=64, gpu_type="h100")


class TestPipeline:
    def test_serving_end_to_end(self, serving_run):
        a = serving_run
        assert a.ok
        assert a.readiness.workload_class == "serving"
        assert a.readiness.tier in ("ready", "degraded")
        assert a.replay_summary is not None
        pol = a.replay_summary["policies"]
        assert "constraint_aware" in pol and "sla_aware" in pol
        assert a.replay_summary["headline_baseline"] == "sla_aware"
        # one corrupt row planted by the generator is ledgered, not fatal
        assert a.readiness.rows_read == a.readiness.rows_usable + 1

    def test_jobs_end_to_end(self, jobs_run):
        a = jobs_run
        assert a.ok
        assert a.readiness.workload_class == "gpu_jobs"
        assert a.replay_summary is not None
        # headline = strongest realistic safe baseline, picked by the locked
        # classifier from the packing/scheduling baseline family
        assert a.replay_summary["headline_baseline"] in (
            "best_fit", "first_fit", "first_fit_decreasing", "greedy_packing",
            "topology_aware", "utilization_aware")
        assert a.fleet_assumption["fleet_gpus"] == 64
        assert a.fleet_assumption["inferred_from_peak_demand"] is False

    def test_fleet_inferred_when_not_given(self, tmp_path):
        log = write_demo_jobs_log(tmp_path / "sacct.csv", days=4.0)
        a = run_pipeline(log)
        assert a.fleet_assumption["inferred_from_peak_demand"] is True
        assert a.fleet_assumption["fleet_gpus"] >= 8

    def test_unreadable_file_is_reported_not_raised(self, tmp_path):
        p = tmp_path / "binary.bin"
        p.write_bytes(b"\x00\x01\x02\x03" * 100)
        a = run_pipeline(p)
        assert not a.ok
        assert a.error
        # even the failure renders to a legible report
        html = render_html(a)
        assert "Nothing was analyzed" in html

    def test_descriptive_stats_present(self, serving_run):
        d = serving_run.descriptive
        assert d["class"] == "serving"
        assert d["requests"] > 1000
        assert d["arrival"]["peak_rpm"] >= d["arrival"]["mean_rpm"]
        assert d["tokens"]["prompt"]["p50"] > 0


class TestReportClaims:
    def test_report_carries_required_wording(self, serving_run, tmp_path):
        out = write_report(serving_run, tmp_path / "r.html")
        text = out.read_text()
        assert "Directional only" in text
        assert "not production savings" in text
        assert "no data left this machine" in text.lower() or \
            "no data was transmitted" in text.lower()

    def test_jobs_report_renders(self, jobs_run, tmp_path):
        out = write_report(jobs_run, tmp_path / "rj.html")
        text = out.read_text()
        assert jobs_run.replay_summary["headline_baseline"] in text
        assert "fleet" in text.lower()

    def test_claim_rules_reject_forbidden_phrases(self):
        with pytest.raises(ClaimRuleError):
            check_claim_rules("this shows production savings of 40%")
        with pytest.raises(ClaimRuleError):
            check_claim_rules("Aurelius is production-proven")
        # negated form is the allowed wording
        check_claim_rules("directional only — not production savings")

    def test_refused_replay_still_produces_report(self, tmp_path):
        p = tmp_path / "tiny.csv"
        p.write_text("timestamp,prompt_tokens,output_tokens\n"
                     + "\n".join(f"{1772000000 + i},5,5" for i in range(10)))
        a = run_pipeline(p)
        assert a.ok
        assert a.readiness.tier == "insufficient"
        assert a.replay_summary is None
        html = render_html(a)
        assert "Replay refused" in html


class TestFingerprintPrivacy:
    def test_no_identifier_values_leak(self, jobs_run):
        fp_text = json.dumps(jobs_run.fingerprint)
        # user names from the demo generator must not appear anywhere
        assert "user0" not in fp_text and "user1" not in fp_text
        # mapping-plan samples are stripped in the shareable artifact
        for m in jobs_run.fingerprint["mapping_plan"]["mappings"]:
            assert m["sample_values"] == []

    def test_fingerprint_has_diagnostics(self, jobs_run):
        fp = jobs_run.fingerprint
        assert fp["artifact"] == "aurelius-replay-schema-fingerprint"
        assert fp["file"]["rows_read"] > 0
        assert "AllocTRES" in fp["columns"]
        assert fp["readiness"]["tier"] in ("ready", "degraded", "insufficient")
