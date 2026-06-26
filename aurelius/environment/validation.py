"""ValidationSuite seed — prove simulated outputs ≈ held-out real trace stats.

PR-1 ships ONE distribution-match primitive (the pattern the full suite, plan
Part E, extends to burstiness / tokens / utilization / memory / priority /
queue-delay / topology / network / cost). It compares a simulated distribution to
a **held-out** real one via a two-sample KS statistic and a histogram-L1, against
a documented tolerance, and returns PASS / WARN / FAIL + the numbers.

Honesty gate: this never returns "production-grade" — only "matches held-out
within tolerance τ." A close match means the *synthetic environment reproduces a
real distribution*, not that it is real telemetry. Pure-stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class DistributionMatch:
    """Result of comparing a simulated distribution to a held-out real one."""

    kind: str
    ks_statistic: float        # max CDF gap (0=identical, 1=disjoint)
    hist_l1: float             # total-variation-like histogram distance (0..1)
    n_sim: int
    n_real: int
    tolerance: float
    verdict: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "ks_statistic": round(self.ks_statistic, 4),
            "hist_l1": round(self.hist_l1, 4), "n_sim": self.n_sim,
            "n_real": self.n_real, "tolerance": self.tolerance, "verdict": self.verdict,
        }


def _ks_statistic(a: list, b: list) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF gap). Deterministic."""
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


def _hist_l1(a: list, b: list, bins: int = 20) -> float:
    """Half-L1 distance between two normalized histograms on a shared range."""
    if not a or not b:
        return 1.0
    lo = min(min(a), min(b))
    hi = max(max(a), max(b))
    if hi <= lo:
        return 0.0
    width = (hi - lo) / bins

    def _h(xs):
        h = [0] * bins
        for x in xs:
            k = min(bins - 1, int((x - lo) / width))
            h[k] += 1
        n = len(xs)
        return [c / n for c in h]

    ha, hb = _h(a), _h(b)
    return 0.5 * sum(abs(x - y) for x, y in zip(ha, hb))


def match_distribution(
    kind: str, simulated: list, real_heldout: list, *,
    tolerance: float = 0.15, warn_tolerance: float = 0.25,
) -> DistributionMatch:
    """Compare a simulated vs held-out real distribution.

    PASS if KS ≤ ``tolerance``; WARN if ≤ ``warn_tolerance``; else FAIL. The KS
    statistic is the headline (distribution-free); ``hist_l1`` is a shape cross-check.
    """
    ks = _ks_statistic(simulated, real_heldout)
    l1 = _hist_l1(simulated, real_heldout)
    if ks <= tolerance:
        verdict = PASS
    elif ks <= warn_tolerance:
        verdict = WARN
    else:
        verdict = FAIL
    return DistributionMatch(
        kind=kind, ks_statistic=ks, hist_l1=l1,
        n_sim=len(simulated), n_real=len(real_heldout),
        tolerance=tolerance, verdict=verdict)


__all__ = ["DistributionMatch", "match_distribution", "PASS", "WARN", "FAIL"]
