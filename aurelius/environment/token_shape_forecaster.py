"""Token-shape scenario forecaster (PR token-shape) — close the dominant forecast gap measured by PR #114.

PR #114 leave-one-out attribution found the planner's remaining forecast value concentrated in TOKEN SHAPE:
output_length 62.8% ≫ prompt_length 24.7% > interarrival_cv 12.3% ≫ arrival_rate 0.3%. This module builds a
small, deterministic scenario forecaster focused ONLY on those three — output-length distribution,
prompt-length distribution, and interarrival burstiness — from RECENT trace windows (no future leakage), to
give the MPC planning rollout a better workload than the single synthetic median (PR #112) or the parametric
six-scenario ensemble (PR #113).

It is NOT a generic forecasting system. It computes empirical rolling quantiles (optionally EWMA-weighted
toward the most recent periods) and emits scenarios across four families:
  * output-length quantile scenarios   (p50 / p75 / p90 / p95-tail)
  * prompt-length quantile scenarios   (p50 / p75 / p90 / p95-tail)
  * burstiness scenarios               (smooth / recent-CV / burst / tail-burst)
  * joint token-shape scenarios        (long-prompt+short-output, short-prompt+long-output, long+long,
                                        burst+long-output, burst+long-prompt)

`scenarios()` returns the full rich list (with provenance) for diagnostics; `planner_scenarios()` /
`__call__` return a compact, weighted projection in the EXACT shape `scenario_forecaster.build_scenarios`
produces, so a `TokenShapeForecaster` instance drops straight into the controller's scenario seam
(`ModelPredictiveEconomicController.scenario_builder`). Arrival *rate* is still taken from the existing
forecaster point (attribution showed it is ~0.3% — not this module's job); only token shape + burstiness are
sourced from the recent empirical distribution. No reward shaping; effects flow only through a better planning
workload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# central-heavier-than-tail deterministic weights (documented; not tuned to a benchmark)
_SMOOTH_CV = 0.2                 # a near-Poisson "smooth" floor for the burstiness sweep
_BURST_MULT = 1.6                # recent-CV → "burst" multiplier
_TAIL_BURST_MULT = 2.2           # recent-CV → "tail-burst" multiplier
_CV_CAP = 4.0                    # never emit an absurd CV


def _wquantile(pairs, q):
    """Weighted quantile of (value, weight) pairs at q∈[0,1] (linear, deterministic)."""
    pts = sorted((float(v), float(w)) for v, w in pairs if w > 0)
    if not pts:
        return 0.0
    tot = math.fsum(w for _, w in pts)
    if tot <= 0:
        return pts[len(pts) // 2][0]
    target = q * tot
    cum = 0.0
    for v, w in pts:
        cum += w
        if cum >= target:
            return v
    return pts[-1][0]


@dataclass
class TokenShapeQuantiles:
    """The empirical recent-window summary the scenarios are built from (provenance-carrying)."""
    out_p50: float
    out_p75: float
    out_p90: float
    out_p95: float
    prompt_p50: float
    prompt_p75: float
    prompt_p90: float
    prompt_p95: float
    cv_recent: float
    arrival_rate: float
    n_obs: int
    source_periods: tuple = (0, 0)
    ewma_half_life: float = 0.0

    def provenance(self) -> str:
        kind = f"EWMA(half_life={self.ewma_half_life:g})" if self.ewma_half_life > 0 else "uniform"
        return (f"recent periods {self.source_periods[0]}–{self.source_periods[1]} "
                f"({self.n_obs} reqs, {kind} weighting)")


@dataclass
class TokenShapeForecaster:
    """Fit on recent records (strictly before the decision period); emit token-shape scenarios.

    Records are ``(arrival_s, output_tokens, prompt_tokens)`` tuples — the same shape the controller and the
    diagnostic harness use. ``window_periods`` records carry an optional period index so EWMA can weight by
    recency; if absent, all records weigh equally.
    """
    q: TokenShapeQuantiles
    period_seconds: float = 60.0

    # ---- construction ----------------------------------------------------
    @classmethod
    def fit(cls, records_by_period: dict, fit_periods, *, ewma_half_life: float = 0.0,
            period_seconds: float = 60.0) -> TokenShapeForecaster:
        """Empirical quantiles over ``fit_periods`` (must all be < the decision period — no leakage).

        ``ewma_half_life`` (in periods) optionally weights more-recent periods higher; 0 = uniform.
        """
        fit_periods = [p for p in fit_periods if records_by_period.get(p)]
        if not fit_periods:
            return cls(TokenShapeQuantiles(*([64] * 4 + [512] * 4), cv_recent=1.0, arrival_rate=0.0,
                                           n_obs=0, source_periods=(0, 0), ewma_half_life=ewma_half_life),
                       period_seconds=period_seconds)
        newest = max(fit_periods)
        out_pairs, prompt_pairs = [], []
        gaps_all, n_obs, n_req_total = [], 0, 0
        for p in fit_periods:
            recs = sorted(records_by_period.get(p, []), key=lambda r: r[0])
            if not recs:
                continue
            w = 1.0
            if ewma_half_life > 0:
                w = 0.5 ** ((newest - p) / ewma_half_life)
            for r in recs:
                out = int(r[1])
                prompt = int(r[2]) if len(r) > 2 else out
                out_pairs.append((out, w))
                prompt_pairs.append((prompt, w))
                n_obs += 1
            gaps_all.extend(recs[i + 1][0] - recs[i][0] for i in range(len(recs) - 1))
            n_req_total += len(recs)
        cv = 1.0
        if len(gaps_all) >= 2:
            m = sum(gaps_all) / len(gaps_all)
            if m > 0:
                var = sum((g - m) ** 2 for g in gaps_all) / len(gaps_all)
                cv = math.sqrt(var) / m
        arrival = n_req_total / (len(fit_periods) * max(period_seconds, 1e-9))
        q = TokenShapeQuantiles(
            out_p50=_wquantile(out_pairs, 0.50), out_p75=_wquantile(out_pairs, 0.75),
            out_p90=_wquantile(out_pairs, 0.90), out_p95=_wquantile(out_pairs, 0.95),
            prompt_p50=_wquantile(prompt_pairs, 0.50), prompt_p75=_wquantile(prompt_pairs, 0.75),
            prompt_p90=_wquantile(prompt_pairs, 0.90), prompt_p95=_wquantile(prompt_pairs, 0.95),
            cv_recent=min(_CV_CAP, cv), arrival_rate=arrival, n_obs=n_obs,
            source_periods=(int(min(fit_periods)), int(newest)), ewma_half_life=ewma_half_life)
        return cls(q, period_seconds=period_seconds)

    # ---- SLA-pressure proxy ---------------------------------------------
    def _sla_pressure(self, arrival_mult, out_level, prompt_level, cv) -> float:
        """Deterministic 0..1 proxy: more decode+prefill work per unit time, burstier ⇒ more SLA pressure.

        Normalised against the central (p50/p50/recent-CV) operating point. A *proxy*, not a measured SLA.
        """
        q = self.q
        base = max(1.0, q.out_p50 + 0.15 * q.prompt_p50)
        work = out_level + 0.15 * prompt_level
        load = arrival_mult * (work / base) * (1.0 + 0.5 * max(0.0, cv - q.cv_recent))
        return round(min(1.0, load / 3.0), 3)

    # ---- rich scenario set (diagnostics + provenance) -------------------
    def scenarios(self) -> list:
        """Full token-shape scenario set across all four families, each with provenance."""
        q = self.q
        prov = q.provenance()
        cv_smooth, cv_recent = _SMOOTH_CV, q.cv_recent
        cv_burst, cv_tail = min(_CV_CAP, cv_recent * _BURST_MULT), min(_CV_CAP, cv_recent * _TAIL_BURST_MULT)
        out = dict(p50=q.out_p50, p75=q.out_p75, p90=q.out_p90, p95=q.out_p95)
        pr = dict(p50=q.prompt_p50, p75=q.prompt_p75, p90=q.prompt_p90, p95=q.prompt_p95)

        def mk(family, label, *, am=1.0, ol=None, pl=None, cv=None, w=1.0):
            ol = out["p50"] if ol is None else ol
            pl = pr["p50"] if pl is None else pl
            cv = cv_recent if cv is None else cv
            return {"family": family, "label": label, "weight": round(w, 4), "arrival_mult": am,
                    "output_p50": round(ol, 1), "output_tail": round(max(ol, out["p95"]), 1),
                    "prompt_p50": round(pl, 1), "prompt_tail": round(max(pl, pr["p95"]), 1),
                    "interarrival_cv": round(cv, 4),
                    "sla_pressure": self._sla_pressure(am, ol, pl, cv), "provenance": prov}

        sc = []
        # 1) output-length quantile sweep (prompt at p50)
        for lbl, key, w in [("out_p50", "p50", 1.0), ("out_p75", "p75", 0.7),
                            ("out_p90", "p90", 0.45), ("out_p95_tail", "p95", 0.25)]:
            sc.append(mk("output_quantile", lbl, ol=out[key], w=w))
        # 2) prompt-length quantile sweep (output at p50)
        for lbl, key, w in [("prompt_p50", "p50", 1.0), ("prompt_p75", "p75", 0.7),
                            ("prompt_p90", "p90", 0.45), ("prompt_p95_tail", "p95", 0.25)]:
            sc.append(mk("prompt_quantile", lbl, pl=pr[key], w=w))
        # 3) burstiness sweep
        for lbl, cv, w in [("smooth", cv_smooth, 0.5), ("recent_cv", cv_recent, 1.0),
                           ("burst", cv_burst, 0.5), ("tail_burst", cv_tail, 0.25)]:
            sc.append(mk("burstiness", lbl, cv=cv, w=w))
        # 4) joint token-shape scenarios
        sc.append(mk("joint", "long_prompt_short_output", ol=out["p50"], pl=pr["p90"], w=0.5))
        sc.append(mk("joint", "short_prompt_long_output", ol=out["p90"], pl=pr["p50"], w=0.5))
        sc.append(mk("joint", "long_prompt_long_output", ol=out["p90"], pl=pr["p90"], w=0.4))
        sc.append(mk("joint", "burst_long_output", ol=out["p90"], cv=cv_burst, w=0.4))
        sc.append(mk("joint", "burst_long_prompt", pl=pr["p90"], cv=cv_burst, w=0.4))
        return sc

    # ---- compact planner projection (drop-in for build_scenarios) -------
    def planner_scenarios(self, ar, tm, tp, cv, *, prompt_tokens=None) -> list:
        """Bounded weighted ensemble in the EXACT shape `build_scenarios` returns, sourced from the recent
        empirical token-shape (arrival rate still from the existing forecaster point ``ar``).

        Keys per dict: label, arrival_rate, tm (output mean), tp (output tail), cv, prompt_mult, weight —
        consumed by ``controller._rollout_ensemble``.
        """
        q = self.q
        base_prompt = float(prompt_tokens) if prompt_tokens else max(1.0, q.prompt_p50)
        ar_mean = float(getattr(ar, "mean", 0.0)) or q.arrival_rate
        ar_low = float(getattr(ar, "p10", ar_mean))
        cv_burst = min(_CV_CAP, q.cv_recent * _BURST_MULT)

        def pm(prompt_level):                       # prompt multiplier vs the planning prompt budget
            return round(max(0.1, prompt_level / base_prompt), 4)

        # central-heaviest; mirrors the #113 ensemble footprint (≈7 scenarios) but on EMPIRICAL quantiles
        rows = [
            ("base", ar_mean, q.out_p50, q.out_p95, q.cv_recent, pm(q.prompt_p50), 1.0),
            ("long_output", ar_mean, q.out_p90, q.out_p95, q.cv_recent, pm(q.prompt_p50), 0.7),
            ("long_prompt", ar_mean, q.out_p50, q.out_p95, q.cv_recent, pm(q.prompt_p90), 0.7),
            ("long_long", ar_mean, q.out_p90, q.out_p95, q.cv_recent, pm(q.prompt_p90), 0.45),
            ("burst", ar_mean, q.out_p75, q.out_p95, cv_burst, pm(q.prompt_p50), 0.5),
            ("tight_sla", ar_mean, q.out_p90, q.out_p95, cv_burst, pm(q.prompt_p90), 0.45),
            ("calm", ar_low, q.out_p50, q.out_p95, max(_SMOOTH_CV, q.cv_recent * 0.6), pm(q.prompt_p50), 0.4),
        ]
        return [{"label": lbl, "arrival_rate": round(arr, 4), "tm": round(t_m, 1),
                 "tp": round(max(t_m, t_p), 1), "cv": round(c, 4), "prompt_mult": p_m, "weight": w}
                for (lbl, arr, t_m, t_p, c, p_m, w) in rows]

    # the controller calls scenario_builder(ar, tm, tp, cv, prompt_tokens=_pt)
    def __call__(self, ar, tm, tp, cv, *, prompt_tokens=None) -> list:
        return self.planner_scenarios(ar, tm, tp, cv, prompt_tokens=prompt_tokens)
