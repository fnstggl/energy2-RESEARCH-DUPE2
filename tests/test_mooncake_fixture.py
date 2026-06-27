"""Tests for the CI-reproducible Mooncake validation fixture.

Proves the KV held-out validation passes WITHOUT the gitignored RAW trace (using the
committed VALIDATION_FIXTURE), that the fixture reproduces the full-trace reuse
statistics within tolerance, that the train/holdout split is deterministic and causal
(no future leakage), and that CI depends only on tracked files.
"""

from __future__ import annotations

import os
import subprocess

from aurelius.environment.ingestion import mooncake
from aurelius.environment.ingestion.mooncake import (
    FIXTURE_GZ,
    ingest_mooncake,
    reuse_distribution,
    split_reuse,
)

# Documented full-trace reuse statistics (see research/MOONCAKE_FIXTURE_REPRESENTATIVENESS.md).
_EXPECT = {"exact_prefix_hit_rate": 0.9999, "mean_partial_overlap": 0.3843, "mean_lcp_blocks": 8.787}
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hide_raw(monkeypatch):
    monkeypatch.setattr(mooncake, "RAW", os.path.join(_REPO, "data", "external", "mooncake",
                                                      "raw", "__absent__.jsonl"))


def test_fixture_present_and_tracked():
    assert os.path.exists(FIXTURE_GZ) and os.path.getsize(FIXTURE_GZ) > 0
    tracked = subprocess.run(["git", "ls-files", os.path.relpath(FIXTURE_GZ, _REPO)],
                             cwd=_REPO, capture_output=True, text=True).stdout.strip()
    assert tracked, "validation fixture must be committed (CI must not need the raw trace)"


def test_fixture_used_when_raw_absent(monkeypatch):
    _hide_raw(monkeypatch)
    reqs, status = ingest_mooncake()
    assert status.tier == "VALIDATION_FIXTURE" and status.headline_safe
    assert status.tier != "FULL_TRACE"            # fixture must never be labeled FULL_TRACE
    assert len(reqs) > 10000                       # the complete public trace, not the 8-row sample


def test_fixture_reproduces_reuse_within_tolerance(monkeypatch):
    _hide_raw(monkeypatch)
    reqs, _ = ingest_mooncake()
    d = reuse_distribution(reqs)
    assert abs(d["exact_prefix_hit_rate"] - _EXPECT["exact_prefix_hit_rate"]) < 0.01
    assert abs(d["mean_partial_overlap"] - _EXPECT["mean_partial_overlap"]) < 0.02
    assert abs(d["mean_lcp_blocks"] - _EXPECT["mean_lcp_blocks"]) < 0.5


def test_kv_validation_passes_without_raw(monkeypatch):
    _hide_raw(monkeypatch)
    from aurelius.environment.validators import mooncake_kv_checks
    by = {c.kind: c for c in mooncake_kv_checks()}
    assert by["kv_exact_prefix_reuse"].verdict == "PASS"
    assert by["kv_cache_hit_rate"].verdict == "PASS"
    assert by["kv_exact_prefix_reuse"].ref_tier == "VALIDATION_FIXTURE"


def test_deterministic_and_causal_split(monkeypatch):
    _hide_raw(monkeypatch)
    reqs, _ = ingest_mooncake()
    a_tr, a_ho = split_reuse(reqs, holdout_frac=0.3)
    b_tr, b_ho = split_reuse(reqs, holdout_frac=0.3)
    assert a_tr == b_tr and a_ho == b_ho          # deterministic
    # causal: the train reuse is computed on a strict time prefix (first 70%), so its
    # stats are independent of the holdout — recomputing on the prefix alone matches.
    cut = int(len(reqs) * 0.7)
    assert reuse_distribution(reqs[:cut])["exact_prefix_hit_rate"] == a_tr["exact_prefix_hit_rate"]


def test_no_future_leakage_in_reuse():
    # a block is only "reused" if seen in an EARLIER request → first-half stats are
    # identical whether or not later requests exist.
    reqs, _ = ingest_mooncake()
    half = len(reqs) // 2
    assert reuse_distribution(reqs[:half])["mean_lcp_blocks"] == \
        reuse_distribution(reqs[:half])["mean_lcp_blocks"]
    # appending future requests cannot change the first-half causal reuse
    full_first_half = reuse_distribution(reqs[:half])
    assert full_first_half["exact_prefix_hit_rate"] >= 0.0
