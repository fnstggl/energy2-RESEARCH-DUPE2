"""Aurelius Replay — self-serve local trace analyzer (POC).

A read-only, local-first tool that lets a prospect point Aurelius at their own
historical scheduler / serving logs and get (1) a descriptive analysis of their
cluster and (2) a counterfactual replay of the same history through the locked
Aurelius evaluation infrastructure (``aurelius/traces/backtest.py`` for LLM
serving, ``aurelius/traces/gpu_scheduling.py`` for GPU job scheduling), scored
on the canonical KPI from ``docs/RESULTS.md`` §1.

Design rules (see ``docs/SELF_SERVE_REPLAY_POC.md``):

- **Local only.** Nothing is transmitted anywhere. The only shareable artifact
  is an anonymized schema fingerprint the user can send explicitly.
- **Never break, never guess silently.** Unknown schemas produce a mapping /
  readiness report — not a crash, and not a replay on garbage. The replay only
  runs when the validator says the mapped data can support it.
- **Locked physics.** This package maps and validates data; it never
  re-implements or tunes the replay engines, policies, or the economics KPI.
- **Honest wording.** Every rendered report carries the ``docs/RESULTS.md`` §8
  allowed wording ("Directional only — not production savings.") and is
  scanned for the forbidden claim substrings before it is written to disk.
"""

from __future__ import annotations

__version__ = "0.1.0"

TOOL_NAME = "Aurelius Replay"
