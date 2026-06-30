# Token-Shape Forecast Gap-Closure Results (Track C)

Does the new recent-empirical **token-shape forecaster** (Track B) close the dominant forecast gap PR #114
attributed to output-length (62.8%) and prompt-length (24.7%)? Measured offline on the bounded Azure window
(`scripts/diagnose_token_shape_gap.py`, artifact `data/external/mpc_controller/token_shape_gap.json`).

**Honest answer: partially, and less than the forecaster already shipped in PR #113. Prompt-length
planner-value drops 36%; output-length is unchanged (it is ~unpredictable beyond the median on this trace).
Reported as-is — not forced.**

## First, a correction to the PR #114 headline

PR #114's "current full MPC = 183 152 (+82.1%)" was measured with the attribution harness's `oracle_var=None`
path, which sets `planning_oracle_records = the exact future` — i.e. it planned with **perfect workload
foresight**. In that artifact `current_mpc_full == oracle_gp_per_dollar == 183 152.2` *exactly*. So +82.1% was
an **oracle-planner** number, not the deployable controller. This diagnostic separates the two cleanly:

| arm (same real eval workload) | SLA-safe gp/$ | gap to oracle | % of current→oracle gap closed |
|--|--|--|--|
| **current** (single synthetic median, deployable, PR #112) | **163 336** | 19 816 | 0% (reference) |
| scenario (PR #113 parametric 6-scenario ensemble) | **182 504** | 648 | **96.7%** |
| tokenshape (this PR, recent empirical quantiles) | **174 933** | 8 219 | **58.5%** |
| oracle (exact future) | 183 152 | 0 | 100% |

The realistic deployable MPC is **163 336**, not 183 152; Track A (`WIDE_VALIDATION_CURRENT_MPC.md`) shows that
deployable controller still beats the strongest baseline by a median **+60%** across regimes.

## Gap closure — the token-shape forecaster helps, but the PR #113 ensemble already helps more

- The token-shape forecaster lifts the deployable current **163 336 → 174 933 (+7.1%)** and closes **58.5%**
  of the gap to oracle. A real improvement over the single-median planning workload.
- **But the existing PR #113 parametric scenario ensemble closes 96.7%** — more than the new forecaster. On
  this window the new empirical forecaster is **not** better than what is already shipped.

## Why — the raw leave-one-out planner-value (unconfounded by renormalisation)

Starting from the oracle, degrade one variable to the model forecast and measure the gp/$ drop. BEFORE
degrades output/prompt to the **global median**; AFTER degrades them to the **token-shape forecaster's
recent-window prediction** (arrival/CV held identical, so the output/prompt shift is clean):

| variable | raw drop BEFORE (global median) | raw drop AFTER (token-shape) | change |
|--|--|--|--|
| **output_length** | 37 550 | **37 550** | **0% — unchanged** |
| **prompt_length** | 14 758 | **9 400** | **−36%** |
| interarrival_cv | 7 339 | 7 339 | (held identical) |
| arrival_rate | 167 | 167 | (held identical) |

- **prompt_length: the forecaster's recent-window quantile beats the global median by 36%** — prompt length
  has temporal structure (median ≈ 828 tokens, drifting) that an 8-period window captures.
- **output_length: zero improvement.** Output median is ≈ 45 tokens and the recent-window quantile ≈ the
  global median, so there is nothing for a recency forecaster to exploit — output length is **~stationary /
  unpredictable beyond the median** on this Azure trace. Since output is the dominant 62.8% lever and the
  forecaster cannot move it, overall gap-closure is capped.

### The normalized %-attribution shift is a renormalisation artifact

| variable | normalized % BEFORE | normalized % AFTER |
|--|--|--|
| output_length | 62.8% | **69.0%** (↑) |
| prompt_length | 24.7% | **17.3%** (↓) |

Output's normalized share *rose* only because the total shrank (prompt's drop fell); its **raw** value is
identical. So "output attribution went up" is not a real worsening — the honest, unconfounded statement is the
raw-drop table above: **prompt −36%, output 0%.**

## Why the PR #113 scenario forecaster still wins here

The scenario ensemble draws its output tail (p90/p99) from the **longer-trained forecast model**, whereas the
token-shape forecaster uses only an 8-period recent window. A longer window + EWMA sensitivity
(`--fit-window 24 --ewma-half-life 6`) was run: the recent quantiles do shift slightly (out_p50 50→48,
prompt_p50 982→968) but the gap-closure is **identical (58.5%)** and the output-length raw drop is **unchanged**
— so output is not a window-length problem. The deeper reason: the leave-one-out degrade collapses output to a
*point* (median or recent-median), and output-length's planner-value lives in its **distribution/tail**, which
collapsing to *any* point loses equally. A recency point-forecast therefore cannot recover it; closing output
needs a per-request distributional predictor (or it is partly irreducible). Prompt, by contrast, has a
genuinely better recent *point* (982 vs the global 828), which is why its drop falls 36%.

## Answers to the success criterion

- **Did output-length attribution decrease?** **No** — raw planner-value unchanged (recent ≈ global median;
  output is ~stationary). The normalized share rose only by renormalisation.
- **Did prompt-length attribution decrease?** **Yes** — raw planner-value down 36%.
- **Did the forecaster close the gap to oracle?** **Partially (58.5%)** — better than single-median (0%) but
  below the already-shipped PR #113 ensemble (96.7%).
- **Did burstiness modelling matter?** Held constant in this attribution by design (cv is 12.3%, not the
  target); the forecaster does emit burstiness scenarios, but they were not the lever measured here.

**Implication for the roadmap:** the dominant lever (output-length, 62.8%) is **not closable by recency-based
quantiles** on this trace and may be partly irreducible; the realistic win is on prompt-length, where the PR
#113 ensemble already captures most of it. Bounded window → magnitudes simulator-inferred; the **direction**
(prompt closable, output not, scenario ensemble ≳ recency forecaster) is the robust finding.
