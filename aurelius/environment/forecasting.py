"""Unified forecasting layer for the canonical environment.

Forecasts the causal, decision-relevant signals an economic controller needs, with
**uncertainty on every target** (mean + p10/p50/p90/p99, from the model's held-out
residual distribution — calibrated, not invented). Each target runs a **baseline
ladder**

    naive (last / running-median / EWMA / seasonal)
      → linear (Ridge / ElasticNet)
        → boosted trees (LightGBM, else sklearn HistGradientBoosting; quantile-capable)

trained on the **train split only**, evaluated on a **held-out** split, and a learned
model is kept **only if it beats the best naive baseline** by a margin — otherwise the
naive baseline is kept and that is reported. No future leakage (a target at period t+1
is predicted from features at periods ≤ t).

Granularity is a configurable **period** with an auto-detected seasonal cycle (the
Fourier + naive-seasonal features use ``cycle_len = max(cycle_pos)+1``): the 2024 one-week
Azure trace bins to **168 hourly periods** (cycle_len 24, a real diurnal cycle), while the
2023 one-hour trace / sample fall back to per-minute (cycle_len 60). See
``research/AZURE_TRACE_COVERAGE_AUDIT.md``. v2026 fleet + KV signals have no sub-period
series, so they report a constant naive with a RUNNING_STATISTIC tag; job runtime is
ABSENT/SKIPPED. (See ``research/AURELIUS_FORECASTING_CONTROLLER_AUDIT.md``.)
"""

from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

try:
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None

EXOGENOUS = ("arrival_rate", "output_token_mean", "output_token_p95",
             "input_token_mean", "interarrival_cv", "electricity_price")
ANCHORED = ("gpu_utilization", "gpu_memory_pressure", "network_pressure", "kv_reuse")
ABSENT = ("job_runtime",)
QUANTILE_TARGETS = {"output_token_p95": 0.95}
_Q = (0.1, 0.5, 0.9, 0.99)


@dataclass
class PeriodFrame:
    """One decision period of causal, action-independent features."""
    index: int
    cycle_pos: int
    arrival_rate: float
    n_requests: int
    output_token_mean: float
    output_token_p95: float
    input_token_mean: float
    interarrival_cv: float
    electricity_price: float
    gpu_utilization: float = 0.0
    gpu_memory_pressure: float = 0.0
    network_pressure: float = 0.0
    kv_reuse: float = 0.0

    def get(self, key: str) -> float:
        return float(getattr(self, key))


def _pctl(xs: list, q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def build_frames(per_period: dict, *, period_seconds: float = 60.0, cycle_len: int = 60,
                 price_by_cycle: dict | None = None, anchors: dict | None = None) -> list:
    """Build the period frame series from ``{period_index: [(arrival_s, out_tok, in_tok?), ...]}``."""
    price_by_cycle = price_by_cycle or {}
    anchors = anchors or {}
    frames = []
    for p in sorted(per_period):
        recs = sorted(per_period[p], key=lambda r: r[0])
        arr = [r[0] for r in recs]
        out = [float(r[1]) for r in recs]
        inp = [float(r[2]) for r in recs if len(r) > 2]
        gaps = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
        mg = statistics.mean(gaps) if gaps else 0.0
        cv = (statistics.pstdev(gaps) / mg) if (gaps and mg > 0) else 0.0
        frames.append(PeriodFrame(
            index=p, cycle_pos=p % cycle_len, arrival_rate=len(recs) / period_seconds,
            n_requests=len(recs), output_token_mean=(statistics.mean(out) if out else 0.0),
            output_token_p95=_pctl(out, 0.95), input_token_mean=(statistics.mean(inp) if inp else 0.0),
            interarrival_cv=cv, electricity_price=price_by_cycle.get(p % cycle_len, 0.06),
            gpu_utilization=anchors.get("gpu_utilization", 0.0),
            gpu_memory_pressure=anchors.get("gpu_memory_pressure", 0.0),
            network_pressure=anchors.get("network_pressure", 0.0),
            kv_reuse=anchors.get("kv_reuse", 0.0)))
    return frames


# --- metrics ----------------------------------------------------------------

def mae(y, yhat):
    return sum(abs(a - b) for a, b in zip(y, yhat)) / len(y) if y else 0.0


def rmse(y, yhat):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, yhat)) / len(y)) if y else 0.0


def pinball(y, yhat, q):
    return sum(max(q * (a - b), (q - 1) * (a - b)) for a, b in zip(y, yhat)) / len(y) if y else 0.0


def coverage_error(y, lo, hi, nominal=0.8):
    """|empirical coverage of [lo,hi] − nominal| — calibration of the uncertainty band."""
    if not y:
        return 0.0
    cov = sum(1 for a, lo_, hi_ in zip(y, lo, hi) if lo_ <= a <= hi_) / len(y)
    return abs(cov - nominal)


# --- naive baselines (causal) ----------------------------------------------

def _naive_predict(kind, hist, cyc_hist=None, cyc_next=None):
    if not hist:
        return 0.0
    if kind == "last":
        return hist[-1]
    if kind == "median":
        return statistics.median(hist[-16:])
    if kind == "ewma":
        a, v = 0.4, hist[0]
        for x in hist[1:]:
            v = a * x + (1 - a) * v
        return v
    if kind == "seasonal" and cyc_hist is not None and cyc_next is not None:
        same = [hist[i] for i in range(len(hist)) if cyc_hist[i] == cyc_next]
        return statistics.mean(same) if same else hist[-1]
    return hist[-1]


_NAIVE_KINDS = ("last", "median", "ewma", "seasonal")


def _supervised(frames, target, n_lags=3, cycle_len=60):
    vals = [f.get(target) for f in frames]
    cyc = [f.cycle_pos for f in frames]
    rows, ys, idx = [], [], []
    for t in range(n_lags - 1, len(frames) - 1):
        lags = [vals[t - k] for k in range(n_lags)]
        window = vals[max(0, t - 5):t + 1]
        rmean = statistics.mean(window)
        rstd = statistics.pstdev(window) if len(window) > 1 else 0.0
        cp = cyc[t + 1]
        feat = [*lags, rmean, rstd, math.sin(2 * math.pi * cp / cycle_len),
                math.cos(2 * math.pi * cp / cycle_len), frames[t].arrival_rate]
        rows.append(feat)
        ys.append(vals[t + 1])
        idx.append(t + 1)
    return rows, ys, idx


def _fit_linear(Xtr, ytr):
    try:
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0)
        m.fit(_np.asarray(Xtr), _np.asarray(ytr))
        return ("ridge", m)
    except Exception:
        return None


def _fit_boosted(Xtr, ytr, quantile=None):
    try:
        import lightgbm as lgb
        params = dict(n_estimators=120, max_depth=4, learning_rate=0.08, num_leaves=15,
                      min_child_samples=5, random_state=0, verbose=-1, n_jobs=1, deterministic=True,
                      force_col_wise=True)
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        m = lgb.LGBMRegressor(**params)
        m.fit(_np.asarray(Xtr), _np.asarray(ytr))
        return ("lightgbm" + (f"_q{quantile}" if quantile else ""), m)
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = dict(max_depth=4, learning_rate=0.08, max_iter=120, random_state=0)
        if quantile is not None:
            kw.update(loss="quantile", quantile=quantile)
        m = HistGradientBoostingRegressor(**kw)
        m.fit(_np.asarray(Xtr), _np.asarray(ytr))
        return ("histgb" + (f"_q{quantile}" if quantile else ""), m)
    except Exception:
        return None


@dataclass
class TargetForecaster:
    target: str
    model_used: str
    fidelity: str                      # REAL_ML | LINEAR | RUNNING_STATISTIC | ABSENT
    metric_name: str
    holdout_metric: float
    naive_metric: float
    beats_naive: bool
    coverage_err: float = 0.0
    naive_kind: str = "last"
    n_lags: int = 3
    cycle_len: int = 60
    quantile: float | None = None
    residual_q: dict = field(default_factory=dict)   # {q: residual} for calibrated bands
    _model: object = None
    status: str = "OK"

    def to_dict(self):
        return {"target": self.target, "model_used": self.model_used, "fidelity": self.fidelity,
                "metric": self.metric_name, "holdout_metric": round(self.holdout_metric, 5),
                "naive_metric": round(self.naive_metric, 5), "beats_naive": self.beats_naive,
                "coverage_err": round(self.coverage_err, 3), "status": self.status}

    def _point(self, vals, cyc, frames_tail, cyc_next):
        if self.model_used.startswith("naive") or self.model_used == "constant" or self._model is None:
            return _naive_predict(self.naive_kind, vals, cyc, cyc_next)
        if _np is None or len(frames_tail) < self.n_lags:
            return _naive_predict(self.naive_kind, vals, cyc, cyc_next)
        t = len(vals) - 1
        lags = [vals[t - k] for k in range(self.n_lags)]
        window = vals[max(0, t - 5):t + 1]
        rmean = statistics.mean(window)
        rstd = statistics.pstdev(window) if len(window) > 1 else 0.0
        feat = [*lags, rmean, rstd, math.sin(2 * math.pi * cyc_next / self.cycle_len),
                math.cos(2 * math.pi * cyc_next / self.cycle_len), frames_tail[-1].arrival_rate]
        return float(self._model.predict(_np.asarray([feat]))[0])

    def forecast(self, vals, cyc, frames_tail, cyc_next):
        """Return a calibrated ForecastPoint (mean + residual-quantile band)."""
        pt = max(0.0, self._point(vals, cyc, frames_tail, cyc_next))
        rq = self.residual_q or {q: 0.0 for q in _Q}
        return ForecastPoint(
            target=self.target, value=pt, mean=pt,
            p10=max(0.0, pt + rq.get(0.1, 0.0)), p50=max(0.0, pt + rq.get(0.5, 0.0)),
            p90=max(0.0, pt + rq.get(0.9, 0.0)), p99=max(0.0, pt + rq.get(0.99, 0.0)),
            model_used=self.model_used, fidelity=self.fidelity, status=self.status)


def fit_target(frames, target, *, train_frac=0.6, margin=0.02, cycle_len=60):
    q = QUANTILE_TARGETS.get(target)
    metric_name = "pinball" if q is not None else "mae"
    if len(frames) < 8:
        return TargetForecaster(target, "naive:last", "RUNNING_STATISTIC", metric_name,
                                0.0, 0.0, False, cycle_len=cycle_len, status="INSUFFICIENT_DATA")
    X, y, idx = _supervised(frames, target, cycle_len=cycle_len)
    if not y:
        return TargetForecaster(target, "naive:last", "RUNNING_STATISTIC", metric_name,
                                0.0, 0.0, False, cycle_len=cycle_len, status="INSUFFICIENT_DATA")
    cut = max(3, int(len(y) * train_frac))
    Xtr, ytr, Xho, yho = X[:cut], y[:cut], X[cut:], y[cut:]
    if not yho:
        return TargetForecaster(target, "naive:last", "RUNNING_STATISTIC", metric_name,
                                0.0, 0.0, False, cycle_len=cycle_len, status="INSUFFICIENT_HOLDOUT")
    vals_all = [f.get(target) for f in frames]
    cyc_all = [f.cycle_pos for f in frames]

    def _score(pred):
        return pinball(yho, pred, q) if q is not None else mae(yho, pred)

    def _resid_q(pred):
        res = sorted(a - b for a, b in zip(yho, pred))
        return {qq: _pctl(res, qq) for qq in _Q}

    best_kind, best_naive, best_naive_pred = "last", math.inf, None
    for kind in _NAIVE_KINDS:
        preds = [_naive_predict(kind, vals_all[:i], cyc_all[:i], cyc_all[i]) for i in idx[cut:]]
        s = _score(preds)
        if s < best_naive:
            best_naive, best_kind, best_naive_pred = s, kind, preds

    candidates = []
    if q is None:
        lin = _fit_linear(Xtr, ytr)
        if lin:
            candidates.append((lin[0], "LINEAR", lin[1]))
    bo = _fit_boosted(Xtr, ytr, quantile=q)
    if bo:
        candidates.append((bo[0], "REAL_ML", bo[1]))

    chosen = TargetForecaster(target, f"naive:{best_kind}", "RUNNING_STATISTIC", metric_name,
                              best_naive, best_naive, False, naive_kind=best_kind,
                              cycle_len=cycle_len, quantile=q, residual_q=_resid_q(best_naive_pred))
    for name, fidelity, model in candidates:
        try:
            pred = [float(v) for v in model.predict(_np.asarray(Xho))]
        except Exception:
            continue
        s = _score(pred)
        if s < best_naive * (1.0 - margin):
            rq = _resid_q(pred)
            lo = [p + rq[0.1] for p in pred]
            hi = [p + rq[0.9] for p in pred]
            chosen = TargetForecaster(target, name, fidelity, metric_name, s, best_naive, True,
                                      coverage_err=coverage_error(yho, lo, hi), naive_kind=best_kind,
                                      cycle_len=cycle_len, quantile=q, residual_q=rq, _model=model)
            break
    if not chosen.beats_naive:                      # calibrate the naive band too
        lo = [p + chosen.residual_q[0.1] for p in best_naive_pred]
        hi = [p + chosen.residual_q[0.9] for p in best_naive_pred]
        chosen.coverage_err = coverage_error(yho, lo, hi)
    return chosen


@dataclass
class ForecastPoint:
    target: str
    value: float
    mean: float
    p10: float
    p50: float
    p90: float
    p99: float
    model_used: str
    fidelity: str
    status: str = "OK"

    def to_dict(self):
        return {"target": self.target, "value": round(self.value, 5), "mean": round(self.mean, 5),
                "p10": round(self.p10, 5), "p50": round(self.p50, 5), "p90": round(self.p90, 5),
                "p99": round(self.p99, 5), "model_used": self.model_used,
                "fidelity": self.fidelity, "status": self.status}


@dataclass
class ForecastBundle:
    horizon: int
    points: dict
    meta: dict = field(default_factory=dict)

    def at(self, target, step=0):
        seq = self.points.get(target)
        return seq[step] if seq and step < len(seq) else None

    def to_dict(self):
        return {"horizon": self.horizon,
                "points": {k: [p.to_dict() for p in v] for k, v in self.points.items()},
                "meta": self.meta}


class ForecastingModel:
    def __init__(self):
        self.forecasters = {}
        self.anchors = {}
        self.cycle_len = 60
        self.fitted = False

    def fit(self, frames, *, train_frac=0.6):
        # detect the seasonal cycle length from the frames first (per-minute → 60,
        # hourly → 24), so the Fourier features + naive-seasonal align with the period.
        if frames:
            self.cycle_len = max(1, max(f.cycle_pos for f in frames) + 1)
        for tgt in EXOGENOUS:
            self.forecasters[tgt] = fit_target(frames, tgt, train_frac=train_frac,
                                               cycle_len=self.cycle_len)
        if frames:
            for tgt in ANCHORED:
                self.anchors[tgt] = frames[-1].get(tgt)
        self.fitted = True
        return self

    def report(self):
        return {t: f.to_dict() for t, f in self.forecasters.items()}

    def predict(self, history, horizon=1):
        points = {}
        cyc = [f.cycle_pos for f in history]
        for tgt, fc in self.forecasters.items():
            vals = [f.get(tgt) for f in history]
            tail, seq, cyc_h = list(history), [], list(cyc)
            for step in range(horizon):
                cyc_next = (history[-1].cycle_pos + 1 + step) % self.cycle_len if history else 0
                p = fc.forecast(vals, cyc_h, tail, cyc_next)
                seq.append(p)
                vals = vals + [p.value]
                cyc_h = cyc_h + [cyc_next]
            points[tgt] = seq
        for tgt in ANCHORED:
            v = self.anchors.get(tgt, 0.0)
            points[tgt] = [ForecastPoint(tgt, v, v, v, v, v, v, "constant", "RUNNING_STATISTIC",
                                         "ANCHORED_NO_PER_PERIOD_SERIES") for _ in range(horizon)]
        for tgt in ABSENT:
            points[tgt] = [ForecastPoint(tgt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "none", "ABSENT",
                                         "SKIPPED_NO_SIGNAL") for _ in range(horizon)]
        return ForecastBundle(horizon=horizon, points=points, meta={"n_history": len(history)})


__all__ = [
    "EXOGENOUS", "ANCHORED", "ABSENT", "QUANTILE_TARGETS", "PeriodFrame", "build_frames",
    "mae", "rmse", "pinball", "coverage_error", "TargetForecaster", "fit_target",
    "ForecastPoint", "ForecastBundle", "ForecastingModel",
]
