#!/usr/bin/env python3
"""Controlled MPC-search fixture: coordinate descent misses a coupled optimum beam search finds; exhaustive
confirms; regret reported. Deterministic. Usage: python scripts/compare_mpc_search_strategies.py"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.environment.v2.mpc_search import AdaptiveMPCSearchV2  # noqa: E402


def reward(c):
    # coupled landscape: fp8 helps ONLY together with aggressive spec; either alone is worse than the
    # bf16/off baseline (a memory-leg interaction). Coordinate descent from the baseline gets stuck.
    if c["precision"] == "fp8" and c["spec"] == "aggressive":
        base = 2.0
    elif c["precision"] == "fp8" or c["spec"] == "aggressive":
        base = 0.6
    else:
        base = 1.0
    if c["precision"] == "int4":
        base -= 0.3
    return base - (0.05 if c["clock"] == "high" else 0.0) + (0.01 if c["batch"] == 64 else 0.0)


def main() -> int:
    space = {"precision": ["bf16", "fp8", "int4"], "spec": ["off", "aggressive"],
             "clock": ["base", "low", "high"], "batch": [32, 64, 128]}
    s = AdaptiveMPCSearchV2(space, exhaustive_threshold=1000)
    out = {}
    for strat in ("coordinate_descent", "beam_search", "exhaustive_cartesian"):
        r = s.search(reward, strategy=strat, audit=(strat != "exhaustive_cartesian"))
        out[strat] = {"selected": r.selected, "reward": round(r.selected_reward, 3),
                      "evaluated": r.evaluated_candidate_count, "raw": r.raw_candidate_count,
                      "regret": r.search_regret, "warning": bool(r.warning)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
