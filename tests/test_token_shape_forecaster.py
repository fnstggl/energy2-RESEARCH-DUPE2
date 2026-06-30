"""Token-shape forecaster tests (focused — no leakage, deterministic quantiles, burstiness, weights)."""

from __future__ import annotations

from types import SimpleNamespace

from aurelius.environment.scenario_forecaster import build_scenarios
from aurelius.environment.token_shape_forecaster import TokenShapeForecaster, _wquantile


def _records():
    # three recent periods with a clear output/prompt distribution; period 5 is a FUTURE outlier
    return {
        0: [(0.0, 10, 100), (1.0, 20, 200), (2.0, 30, 300)],
        1: [(0.0, 12, 110), (1.5, 22, 220), (3.0, 34, 340)],
        2: [(0.0, 11, 105), (2.0, 21, 210), (4.0, 32, 320)],
        5: [(0.0, 9999, 99999)] * 50,                 # future period — must NEVER be consumed
    }


def _ar(mean=0.05, p10=0.03):
    return SimpleNamespace(mean=mean, p10=p10)


# --- no future leakage -------------------------------------------------------
def test_no_future_leakage():
    recs = _records()
    fit_on_past = TokenShapeForecaster.fit(recs, [0, 1, 2])
    # the exact same fit must result whether or not the future outlier period exists in the dict
    recs_no_future = {p: recs[p] for p in (0, 1, 2)}
    fit_isolated = TokenShapeForecaster.fit(recs_no_future, [0, 1, 2])
    assert fit_on_past.q == fit_isolated.q
    # and the future outlier must not have leaked into the quantiles
    assert fit_on_past.q.out_p95 < 100 and fit_on_past.q.prompt_p95 < 1000
    assert fit_on_past.q.source_periods == (0, 2)


# --- deterministic quantile scenarios ---------------------------------------
def test_quantile_scenarios_deterministic_and_ordered():
    tsf = TokenShapeForecaster.fit(_records(), [0, 1, 2])
    a, b = tsf.scenarios(), tsf.scenarios()
    assert a == b                                              # fully deterministic
    out_q = {s["label"]: s["output_p50"] for s in a if s["family"] == "output_quantile"}
    assert out_q["out_p50"] <= out_q["out_p75"] <= out_q["out_p90"] <= out_q["out_p95_tail"]
    prompt_q = {s["label"]: s["prompt_p50"] for s in a if s["family"] == "prompt_quantile"}
    assert prompt_q["prompt_p50"] <= prompt_q["prompt_p75"] <= prompt_q["prompt_p90"] <= prompt_q["prompt_p95_tail"]


def test_wquantile_weighting_pulls_toward_heavy_mass():
    # equal weights → median 2; heavy weight on the high value pulls the 0.5 quantile up
    assert _wquantile([(1, 1), (2, 1), (3, 1)], 0.5) == 2.0
    assert _wquantile([(1, 1), (3, 100)], 0.5) == 3.0


def test_ewma_weights_recent_periods_higher():
    # a recent period with larger outputs should pull EWMA quantiles above the uniform fit
    recs = {0: [(0.0, 10, 100)] * 5, 1: [(0.0, 10, 100)] * 5, 2: [(0.0, 40, 400)] * 5}
    uniform = TokenShapeForecaster.fit(recs, [0, 1, 2]).q.out_p50
    ewma = TokenShapeForecaster.fit(recs, [0, 1, 2], ewma_half_life=1.0).q.out_p50
    assert ewma >= uniform


# --- burstiness --------------------------------------------------------------
def test_burstiness_scenarios_change_cv():
    tsf = TokenShapeForecaster.fit(_records(), [0, 1, 2])
    cv = {s["label"]: s["interarrival_cv"] for s in tsf.scenarios() if s["family"] == "burstiness"}
    assert cv["smooth"] < cv["recent_cv"] <= cv["burst"] <= cv["tail_burst"]


# --- planner projection (drop-in for build_scenarios) ------------------------
def test_planner_projection_shape_matches_build_scenarios():
    tsf = TokenShapeForecaster.fit(_records(), [0, 1, 2])
    rows = tsf(_ar(), None, None, None, prompt_tokens=200)
    # exact key set the controller's _rollout_ensemble consumes (parity with build_scenarios output)
    ref_keys = set(build_scenarios(SimpleNamespace(mean=0.05, p90=0.07, p10=0.03),
                                   SimpleNamespace(mean=20, p90=30),
                                   SimpleNamespace(value=30, p95=30, p99=40, mean=20),
                                   SimpleNamespace(mean=1.0, p90=1.5))[0])
    assert all(set(r) == ref_keys for r in rows)
    assert tsf(_ar(), None, None, None, prompt_tokens=200) == rows   # __call__ == planner_scenarios


def test_planner_projection_preserves_weights():
    tsf = TokenShapeForecaster.fit(_records(), [0, 1, 2])
    rows = {r["label"]: r for r in tsf(_ar(), None, None, None, prompt_tokens=200)}
    # documented deterministic weights — base heaviest, tails lighter (not tuned to a benchmark)
    assert rows["base"]["weight"] == 1.0
    assert rows["long_long"]["weight"] < rows["long_output"]["weight"] < rows["base"]["weight"]
    # central scenario carries the recent median output; long_output carries p90 (≥ p50)
    assert rows["long_output"]["tm"] >= rows["base"]["tm"]
    # long_prompt stresses prompt only (prompt_mult > base's)
    assert rows["long_prompt"]["prompt_mult"] > rows["base"]["prompt_mult"]
