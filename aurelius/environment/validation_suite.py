"""ValidationSuite — prove the synthetic environment matches held-out real stats.

Compares each simulated distribution to a **held-out** real slice via KS,
Wasserstein-1, histogram-L1, and percentile error, against documented tolerances,
emitting PASS / WARN / FAIL + the numbers. The full suite (build spec) covers
burstiness, tokens, inter-arrival, GPU util/memory, priority mix, queue-delay,
GPU-type mix, placement/fragmentation, rack/asw locality, network rx/tx and a cost
sanity band; this module ships the metrics + the harness and seeds the Azure-token
and inter-arrival checks end-to-end.

Honesty gate (hard rule): the suite NEVER returns "production realistic." Its
ceiling is ``MATCHES_HELDOUT`` per check, and the overall verdict is capped at
``NOT_PRODUCTION_REALISTIC_YET`` whenever any calibrated parameter feeding the
environment is below TRACE_DERIVED (i.e. HEURISTIC/INFERRED). A close match means
the environment reproduces a real distribution — not that it is real telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import HEADLINE_SAFE_TIERS

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NOT_PRODUCTION_REALISTIC_YET = "NOT_PRODUCTION_REALISTIC_YET"
MATCHES_HELDOUT = "MATCHES_HELDOUT"


# ---------------------------------------------------------------------------
# Distribution distance metrics (pure-stdlib, deterministic)
# ---------------------------------------------------------------------------

def ks_statistic(a: list, b: list) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF gap)."""
    if not a or not b:
        return 1.0
    sa, sb = sorted(a), sorted(b)
    grid = sorted(set(sa) | set(sb))
    na, nb = len(sa), len(sb)
    ia = ib = 0
    d = 0.0
    for x in grid:
        while ia < na and sa[ia] <= x:
            ia += 1
        while ib < nb and sb[ib] <= x:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    return d


def wasserstein1(a: list, b: list) -> float:
    """1-Wasserstein (earth-mover) distance between two empirical 1-D samples."""
    if not a or not b:
        return float("inf")
    sa, sb = sorted(a), sorted(b)
    grid = sorted(set(sa) | set(sb))
    na, nb = len(sa), len(sb)
    ia = ib = 0
    area = 0.0
    prev = grid[0]
    for x in grid:
        area += abs(ia / na - ib / nb) * (x - prev)
        prev = x
        while ia < na and sa[ia] <= x:
            ia += 1
        while ib < nb and sb[ib] <= x:
            ib += 1
    return area


def hist_l1(a: list, b: list, bins: int = 20) -> float:
    """Half-L1 (total-variation) distance between two normalized histograms."""
    if not a or not b:
        return 1.0
    lo, hi = min(min(a), min(b)), max(max(a), max(b))
    if hi <= lo:
        return 0.0
    w = (hi - lo) / bins

    def _h(xs):
        h = [0] * bins
        for x in xs:
            h[min(bins - 1, int((x - lo) / w))] += 1
        return [c / len(xs) for c in h]

    return 0.5 * sum(abs(x - y) for x, y in zip(_h(a), _h(b)))


def _pct(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Per-distribution check
# ---------------------------------------------------------------------------

@dataclass
class DistributionCheck:
    kind: str
    ks: float
    wasserstein: float
    hist_l1: float
    p50_err: float
    p95_err: float
    tolerance: float
    verdict: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "ks": round(self.ks, 4),
            "wasserstein": round(self.wasserstein, 4), "hist_l1": round(self.hist_l1, 4),
            "p50_err": round(self.p50_err, 4), "p95_err": round(self.p95_err, 4),
            "tolerance": self.tolerance, "verdict": self.verdict,
        }


def check_distribution(
    kind: str, simulated: list, real_heldout: list, *,
    tolerance: float = 0.15, warn_tolerance: float = 0.25,
) -> DistributionCheck:
    """KS-headline check (distribution-free) + Wasserstein/L1/percentile cross-checks."""
    ks = ks_statistic(simulated, real_heldout)

    def _rel(q):
        r = _pct(real_heldout, q)
        return abs(_pct(simulated, q) - r) / abs(r) if r else 0.0

    verdict = PASS if ks <= tolerance else (WARN if ks <= warn_tolerance else FAIL)
    return DistributionCheck(
        kind=kind, ks=ks, wasserstein=wasserstein1(simulated, real_heldout),
        hist_l1=hist_l1(simulated, real_heldout), p50_err=_rel(0.50), p95_err=_rel(0.95),
        tolerance=tolerance, verdict=verdict)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    checks: list                       # list[DistributionCheck]
    all_params_headline_safe: bool
    overall_verdict: str

    def passed(self) -> bool:
        return all(c.verdict == PASS for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "all_params_headline_safe": self.all_params_headline_safe,
            "overall_verdict": self.overall_verdict,
            "passed": self.passed(),
        }


def run_validation(checks: list, calibrated_params: list) -> ValidationReport:
    """Aggregate checks + apply the honesty cap.

    ``overall_verdict`` is ``MATCHES_HELDOUT`` only if every check PASSes **and**
    every calibrated param is headline-safe (≥ TRACE_DERIVED); otherwise it is
    capped at ``NOT_PRODUCTION_REALISTIC_YET``.
    """
    headline_safe = all(p.tier in HEADLINE_SAFE_TIERS for p in calibrated_params)
    all_pass = all(c.verdict == PASS for c in checks)
    overall = (MATCHES_HELDOUT if (all_pass and headline_safe)
               else NOT_PRODUCTION_REALISTIC_YET)
    return ValidationReport(checks=checks, all_params_headline_safe=headline_safe,
                            overall_verdict=overall)


__all__ = [
    "PASS", "WARN", "FAIL", "MATCHES_HELDOUT", "NOT_PRODUCTION_REALISTIC_YET",
    "ks_statistic", "wasserstein1", "hist_l1", "DistributionCheck",
    "check_distribution", "ValidationReport", "run_validation",
]
