# Prefill/Decode Controlled Fixtures (PR #107, Phase 8)

The causal bridges, proven in `tests/test_prefill_decode_economics.py` (10) +
`tests/test_cache_locality_physics.py` (PR #106) + `world_validation.py` prefill/decode checks (6). These
prove the *physics*, which the user requires regardless of the held-out headline.

| # | fixture | proves |
|--|--|--|
| 1 | `test_prefix_hit_reduces_prefill_only_decode_unchanged` | KV hit cuts **prefill**; decode (output-token) is **unchanged** — the Cause-A fix |
| 2 | `test_token_conservation_prefill_remaining` | `prefill_remaining = prompt − saved`; decode tokens conserved |
| 3 | `test_ttft_falls_with_prefix_hit` | a prefix hit lowers TTFT (full hit → only the base term) |
| 4 | `test_decode_heavy_stays_decode_bound_despite_reuse` | decode-bound work: KV reuse cuts <5 % of realized GPU-seconds (the honest Azure case) |
| 5 | `test_prefill_heavy_benefits_more_from_reuse` | prefill-heavy work: KV reuse cuts >30 % of realized GPU-seconds |
| 6 | `test_batching_changes_decode_work` | aggressive batching lowers per-token decode work |
| 7 | `test_cost_mode_ordering_and_no_free_cost` | realized ≤ hybrid ≤ provisioned; no mode is free; hybrid keeps a warm-idle floor |
| 8 | `test_realized_work_falls_when_prefill_falls` | KV reuse → lower realized-work cost (in a tight period) |
| 9 | `test_determinism` | deterministic replay |
| 10 | `test_provisioned_mode_reproduces_no_gp_dollar_gain_realized_does` | provisioned reproduces #106; realized monetizes the prefill saving |

Plus `world_validation.py`: prefix-hit-reduces-prefill-only, token conservation, TTFT-falls,
decode-bound-barely-monetizes, cost-mode ordering, no-free-cost (6 PASS). Total suite: **27 PASS / 0 FAIL
/ 3 SKIPPED**.

Mapping to the user's Phase-8 list: (1) TTFT↓ on hit — #3; (2) prefill GPU-s↓ — #1; (3) decode tokens
unchanged — #1; (4) decode-bound stays decode-bound — #4; (5) prefill-heavy benefits more — #5; (6/7)
batching helps/hurts — #6 + the saturation tail; (8) provisioned little benefit — #10; (9) realized
upper bound — #8/#10; (10) hybrid between — #7; (11) prewarm/(12) migration no-free-win — covered by PR
#105/#106 suites (the phase model does not change their conservation, only their payoff channel). The
fixtures are the deliverable: the bridges work causally even though the decode-bound Azure headline is
marginal.
