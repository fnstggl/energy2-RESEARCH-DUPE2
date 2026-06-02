#!/usr/bin/env python3
"""Audit every HF discovery candidate through the unified RedistributionGate.

This is the first consumer of
``aurelius.ingestion.redistribution_gate.decide_redistribution`` and the
first consumer of ``OperatorPolicyLedger.permits_redistribution`` in a
sample-commit decision path. With the default committed policy file
shipping zero grants:

- Permissive-licensed candidates remain ``permitted`` (same as today).
- ``license = None`` candidates (including the 4 Round-8 license-blocked
  ones) remain ``denied`` (same as today).
- Declared non-permissive licenses (``"other"``, ``"openrail"``, custom
  research licenses) are denied with an explicit
  ``denied_declared_non_permissive_license`` code (this is the only new
  surfacing — those datasets were already not committed, but they were
  not separately reported until this audit).

Outputs
-------

Writes ``data/external/hf_discovery/redistribution_gate_audit.json``:

- ``audited_at_iso`` / ``audited_at_s`` / ``git_sha``
- ``policy_doc_version`` / ``policy_default`` / ``policy_grant_count``
- ``default_scope_evaluated`` (always ``"committed_normalized_sample"``)
- ``per_dataset`` — one row per candidate with
  ``{dataset_id, license_observed, license_status, permitted,
  reason_code, reason_detail, operator_grant_dataset_id,
  recommended_action_at_discovery_time}``
- ``rollup`` — count of decisions by ``(license_status, permitted,
  reason_code)``

Read-only:
    - does NOT mutate the candidate registry
    - does NOT call the HF API
    - does NOT download data
    - does NOT read HF_TOKEN
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
CANDIDATES_PATH = DISC_DIR / "hf_dataset_candidates.json"
POLICY_PATH = DISC_DIR / "operator_redistribution_policy.json"
OUT_PATH = DISC_DIR / "redistribution_gate_audit.json"

DOC_VERSION = "redistribution_gate_audit_v1"
DEFAULT_SCOPE = "committed_normalized_sample"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.ingestion.operator_redistribution_policy import (  # noqa: E402
    POLICY_DOC_VERSION,
    OperatorPolicyLedger,
)
from aurelius.ingestion.redistribution_gate import (  # noqa: E402
    decide_redistribution,
)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _load_candidates() -> list[dict]:
    with open(CANDIDATES_PATH) as fh:
        payload = json.load(fh)
    candidates = payload.get("candidates") or []
    if not isinstance(candidates, list):
        raise ValueError(
            f"hf_dataset_candidates.json::candidates must be a JSON array, "
            f"got {type(candidates)!r}"
        )
    return candidates


def _per_candidate_decision(
    ledger: OperatorPolicyLedger,
    candidate: dict,
    *,
    scope: str,
    now_iso: str,
) -> dict:
    decision = decide_redistribution(
        dataset_id=candidate.get("dataset_id") or "",
        license_str=candidate.get("license"),
        scope=scope,
        ledger=ledger,
        now_iso=now_iso,
    )
    return {
        "dataset_id": decision.license_observed and candidate.get("dataset_id")
        or candidate.get("dataset_id"),
        "license_observed": decision.license_observed,
        "license_status": decision.license_status,
        "permitted": decision.permitted,
        "reason_code": decision.reason_code,
        "reason_detail": decision.reason_detail,
        "operator_grant_dataset_id": decision.operator_grant_dataset_id,
        "scope_evaluated": decision.scope,
        "recommended_action_at_discovery_time": candidate.get(
            "recommended_action"
        ),
    }


def build_audit_payload(
    *,
    ledger: OperatorPolicyLedger,
    candidates: list[dict],
    scope: str = DEFAULT_SCOPE,
    now_iso: str | None = None,
    git_sha: str = "",
) -> dict:
    """Pure function: build the audit payload from in-memory inputs.

    Exposed so tests can call it without writing to disk.
    """

    if now_iso is None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    per_dataset = sorted(
        (
            _per_candidate_decision(
                ledger, c, scope=scope, now_iso=now_iso,
            )
            for c in candidates
        ),
        key=lambda row: row["dataset_id"] or "",
    )

    permitted_count = sum(1 for r in per_dataset if r["permitted"])
    denied_count = len(per_dataset) - permitted_count

    rollup_by_status = Counter(r["license_status"] for r in per_dataset)
    rollup_by_reason = Counter(r["reason_code"] for r in per_dataset)
    rollup_by_decision_status = Counter(
        (r["license_status"], r["permitted"]) for r in per_dataset
    )

    return {
        "doc_version": DOC_VERSION,
        "audited_at_iso": now_iso,
        "audited_at_s": time.time(),
        "git_sha": git_sha,
        "default_scope_evaluated": scope,
        "policy_doc_version": ledger.doc_version,
        "policy_default": ledger.policy_default,
        "policy_grant_count": len(ledger.grants),
        "candidate_count": len(per_dataset),
        "permitted_count": permitted_count,
        "denied_count": denied_count,
        "rollup": {
            "by_license_status": dict(sorted(rollup_by_status.items())),
            "by_reason_code": dict(sorted(rollup_by_reason.items())),
            "by_status_and_permitted": [
                {
                    "license_status": s,
                    "permitted": p,
                    "count": c,
                }
                for (s, p), c in sorted(
                    rollup_by_decision_status.items(),
                    key=lambda kv: (kv[0][0], kv[0][1]),
                )
            ],
        },
        "scope": (
            "Read-only audit: walks every entry in "
            "hf_dataset_candidates.json::candidates through "
            "aurelius.ingestion.redistribution_gate.decide_redistribution "
            "under the committed default operator policy. Does NOT mutate "
            "the candidate registry. Does NOT call the HF API. Does NOT "
            "download data. Does NOT read HF_TOKEN."
        ),
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "ingests_new_hf_data": False,
        "per_dataset": per_dataset,
    }


def main() -> int:
    if not POLICY_PATH.exists():
        print(
            f"[audit] missing policy file at {POLICY_PATH}; nothing to do",
            file=sys.stderr,
        )
        return 1
    if not CANDIDATES_PATH.exists():
        print(
            f"[audit] missing candidates registry at {CANDIDATES_PATH}",
            file=sys.stderr,
        )
        return 1

    ledger = OperatorPolicyLedger.load(POLICY_PATH)
    if ledger.doc_version != POLICY_DOC_VERSION:
        print(
            f"[audit] policy doc_version {ledger.doc_version!r} != "
            f"{POLICY_DOC_VERSION!r}",
            file=sys.stderr,
        )
        return 1

    candidates = _load_candidates()
    payload = build_audit_payload(
        ledger=ledger,
        candidates=candidates,
        scope=DEFAULT_SCOPE,
        now_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_sha=_git_sha(),
    )

    DISC_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(
        f"[audit] wrote {OUT_PATH.relative_to(REPO_ROOT)} — "
        f"{payload['candidate_count']} candidates "
        f"({payload['permitted_count']} permitted, "
        f"{payload['denied_count']} denied) "
        f"under {ledger.policy_default!r} default policy with "
        f"{payload['policy_grant_count']} grants"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
