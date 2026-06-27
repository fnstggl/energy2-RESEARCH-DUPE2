"""ValidationSuite — prove the canonical environment matches held-out real stats.

Compares each environment distribution to a **held-out** real reference and emits
PASS / WARN / FAIL / SKIPPED + the numbers. Two comparison modes, because the real
references come at two granularities:

  * **sample-based** (Azure arrivals/tokens, Mooncake overlap): raw held-out
    samples are available → KS, Wasserstein-1, histogram-L1, percentile error.
  * **summary-based** (Alibaba v2026 fleet): only the FULL_TRACE_EXACT calibration
    *aggregates + fixed-bin histograms* are available (6.5 B rows are not held in
    memory) → mean/percentile relative error + category-mix total-variation.

Honesty gate (hard rules):
  * The suite NEVER returns "production realistic"; its ceiling is ``MATCHES_HELDOUT``
    per check, and the overall verdict is capped at ``NOT_PRODUCTION_REALISTIC_YET``
    whenever any calibrated parameter feeding the environment is below TRACE_DERIVED.
  * A check whose real reference is unavailable is ``SKIPPED`` with the EXACT
    artifact/path/command required to enable it — never a silent pass.
  * Every check records the data tier of its real reference (FULL_TRACE_EXACT /
    FULL_TRACE_APPROX / SUBSET_TRACE / SAMPLE_FIXTURE / …) so a SAMPLE-based match
    can never be read as a full-trace match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import HEADLINE_SAFE_TIERS

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
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


def hist_l1_counts(a_counts: list, b_counts: list) -> float:
    """Half-L1 (total-variation) over two ALREADY-BINNED histograms (equal bins).

    For comparing the environment's fixed-bin histogram directly to the v2026
    FULL_TRACE_APPROX histogram (same bin layout) without raw samples."""
    if not a_counts or not b_counts or len(a_counts) != len(b_counts):
        return 1.0
    sa, sb = sum(a_counts) or 1, sum(b_counts) or 1
    return 0.5 * sum(abs(x / sa - y / sb) for x, y in zip(a_counts, b_counts))


def category_mix_l1(a_counts: dict, b_counts: dict) -> float:
    """Half-L1 (total-variation) absolute error between two category mixes."""
    if not a_counts or not b_counts:
        return 1.0
    keys = set(a_counts) | set(b_counts)
    sa = sum(a_counts.values()) or 1
    sb = sum(b_counts.values()) or 1
    return 0.5 * sum(abs(a_counts.get(k, 0) / sa - b_counts.get(k, 0) / sb) for k in keys)


def rel_err(sim: float, real: float) -> float:
    """Relative error |sim-real|/|real| (0 when real==0 and sim==0)."""
    if real == 0:
        return 0.0 if sim == 0 else 1.0
    return abs(sim - real) / abs(real)


def _pct(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Check result schema
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """One validation: env distribution vs a held-out real reference.

    ``metrics`` holds whichever distances applied (ks / wasserstein / hist_l1 /
    p50_err / p95_err / p99_err / mean_err / mix_l1). ``ref_tier`` is the data tier
    of the REAL reference (so SAMPLE-based matches are never read as full-trace).
    A SKIPPED check carries ``detail`` = the exact artifact/path/command needed.
    """

    kind: str
    source: str
    ref_tier: str
    mode: str                          # "samples" | "summary" | "category_mix" | "skipped"
    metric: float                      # the headline distance used for the verdict
    metric_name: str
    metrics: dict
    tolerance: float
    warn_tolerance: float
    verdict: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "source": self.source, "ref_tier": self.ref_tier,
            "mode": self.mode, "metric_name": self.metric_name,
            "metric": round(self.metric, 4) if self.metric == self.metric else None,
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "tolerance": self.tolerance, "warn_tolerance": self.warn_tolerance,
            "verdict": self.verdict, "detail": self.detail,
        }


def _verdict(metric: float, tolerance: float, warn_tolerance: float) -> str:
    return PASS if metric <= tolerance else (WARN if metric <= warn_tolerance else FAIL)


def check_samples(
    kind: str, simulated: list, real_heldout: list, *,
    source: str, ref_tier: str, tolerance: float = 0.15, warn_tolerance: float = 0.25,
) -> ValidationCheck:
    """Sample-based check: KS headline + Wasserstein/L1/percentile cross-checks."""
    if not simulated or not real_heldout:
        return skipped_check(
            kind, source=source, required_artifact="held-out samples",
            command="(provide non-empty simulated + real samples)",
            reason="empty sample(s)")
    ks = ks_statistic(simulated, real_heldout)
    metrics = {
        "ks": ks, "wasserstein": wasserstein1(simulated, real_heldout),
        "hist_l1": hist_l1(simulated, real_heldout),
        "p50_err": rel_err(_pct(simulated, 0.50), _pct(real_heldout, 0.50)),
        "p95_err": rel_err(_pct(simulated, 0.95), _pct(real_heldout, 0.95)),
        "p99_err": rel_err(_pct(simulated, 0.99), _pct(real_heldout, 0.99)),
    }
    return ValidationCheck(
        kind=kind, source=source, ref_tier=ref_tier, mode="samples", metric=ks,
        metric_name="ks", metrics=metrics, tolerance=tolerance,
        warn_tolerance=warn_tolerance, verdict=_verdict(ks, tolerance, warn_tolerance))


def check_summary(
    kind: str, sim_summary: dict, real_summary: dict, *,
    source: str, ref_tier: str, keys=("mean", "p50", "p95", "p99"),
    tolerance: float = 0.15, warn_tolerance: float = 0.30,
) -> ValidationCheck:
    """Summary-based check (no raw samples): max relative error over the moments/
    percentiles both summaries share. Used for v2026 (only aggregates are held)."""
    shared = [k for k in keys if k in sim_summary and k in real_summary]
    if not shared:
        return skipped_check(
            kind, source=source, required_artifact="overlapping summary keys",
            command="(simulated and real summaries share no keys)",
            reason="no shared summary keys")
    errs = {f"{k}_err": rel_err(sim_summary[k], real_summary[k]) for k in shared}
    worst = max(errs.values())
    return ValidationCheck(
        kind=kind, source=source, ref_tier=ref_tier, mode="summary", metric=worst,
        metric_name="max_rel_err", metrics=errs, tolerance=tolerance,
        warn_tolerance=warn_tolerance, verdict=_verdict(worst, tolerance, warn_tolerance))


def check_category_mix(
    kind: str, sim_counts: dict, real_counts: dict, *,
    source: str, ref_tier: str, tolerance: float = 0.10, warn_tolerance: float = 0.20,
) -> ValidationCheck:
    """Category-mix check: total-variation (half-L1) between two category mixes."""
    if not sim_counts or not real_counts:
        return skipped_check(
            kind, source=source, required_artifact="both category mixes",
            command="(provide non-empty simulated + real category counts)",
            reason="empty category mix")
    tv = category_mix_l1(sim_counts, real_counts)
    return ValidationCheck(
        kind=kind, source=source, ref_tier=ref_tier, mode="category_mix", metric=tv,
        metric_name="mix_l1", metrics={"mix_l1": tv, "n_categories": len(set(sim_counts) | set(real_counts))},
        tolerance=tolerance, warn_tolerance=warn_tolerance,
        verdict=_verdict(tv, tolerance, warn_tolerance))


def skipped_check(
    kind: str, *, source: str, required_artifact: str, command: str, reason: str = "",
) -> ValidationCheck:
    """An explicitly SKIPPED check — carries the exact artifact/path/command needed
    to enable it (never a silent pass)."""
    detail = (f"requires: {required_artifact}; reason: {reason or 'unavailable'}; "
              f"enable with: {command}")
    return ValidationCheck(
        kind=kind, source=source, ref_tier="UNAVAILABLE", mode="skipped",
        metric=float("nan"), metric_name="n/a", metrics={}, tolerance=0.0,
        warn_tolerance=0.0, verdict=SKIPPED, detail=detail)


# ---------------------------------------------------------------------------
# Back-compat: the original DistributionCheck / check_distribution API
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
    verdict = _verdict(ks, tolerance, warn_tolerance)
    return DistributionCheck(
        kind=kind, ks=ks, wasserstein=wasserstein1(simulated, real_heldout),
        hist_l1=hist_l1(simulated, real_heldout),
        p50_err=rel_err(_pct(simulated, 0.50), _pct(real_heldout, 0.50)),
        p95_err=rel_err(_pct(simulated, 0.95), _pct(real_heldout, 0.95)),
        tolerance=tolerance, verdict=verdict)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    checks: list                       # list[ValidationCheck | DistributionCheck]
    all_params_headline_safe: bool
    overall_verdict: str
    counts: dict = field(default_factory=dict)

    def passed(self) -> bool:
        """True iff every EVALUATED (non-skipped) check passed and ≥1 was evaluated."""
        evaluated = [c for c in self.checks if c.verdict != SKIPPED]
        return bool(evaluated) and all(c.verdict == PASS for c in evaluated)

    def to_dict(self) -> dict:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "all_params_headline_safe": self.all_params_headline_safe,
            "overall_verdict": self.overall_verdict,
            "passed": self.passed(),
            "counts": self.counts,
        }


def run_validation(checks: list, calibrated_params: list) -> ValidationReport:
    """Aggregate checks + apply the honesty cap.

    ``overall_verdict`` is ``MATCHES_HELDOUT`` only if every EVALUATED check PASSes,
    at least one check was evaluated, **and** every calibrated param is
    headline-safe (≥ TRACE_DERIVED); otherwise it is capped at
    ``NOT_PRODUCTION_REALISTIC_YET``. SKIPPED checks are reported, never silently
    counted as passes.
    """
    headline_safe = all(p.tier in HEADLINE_SAFE_TIERS for p in calibrated_params)
    counts = {
        "pass": sum(1 for c in checks if c.verdict == PASS),
        "warn": sum(1 for c in checks if c.verdict == WARN),
        "fail": sum(1 for c in checks if c.verdict == FAIL),
        "skipped": sum(1 for c in checks if c.verdict == SKIPPED),
        "total": len(checks),
    }
    evaluated = [c for c in checks if c.verdict != SKIPPED]
    all_eval_pass = bool(evaluated) and all(c.verdict == PASS for c in evaluated)
    overall = (MATCHES_HELDOUT if (all_eval_pass and headline_safe)
               else NOT_PRODUCTION_REALISTIC_YET)
    return ValidationReport(checks=checks, all_params_headline_safe=headline_safe,
                            overall_verdict=overall, counts=counts)


__all__ = [
    "PASS", "WARN", "FAIL", "SKIPPED", "MATCHES_HELDOUT", "NOT_PRODUCTION_REALISTIC_YET",
    "ks_statistic", "wasserstein1", "hist_l1", "hist_l1_counts", "category_mix_l1",
    "rel_err", "ValidationCheck", "check_samples", "check_summary",
    "check_category_mix", "skipped_check",
    "DistributionCheck", "check_distribution", "ValidationReport", "run_validation",
]
