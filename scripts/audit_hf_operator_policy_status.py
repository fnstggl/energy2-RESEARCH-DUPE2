"""Audit current operator-policy status for license=None HF candidates.

Discovery / data-engine PR. No scheduler change, no controller change, no
robust-energy-engine change, no oracle as headline, no production claim,
no Tier 1 promotion, no new HF data downloaded.

This script reads:

* ``data/external/hf_discovery/hf_dataset_candidates.json`` — the audited
  HF candidate registry produced by Rounds 1-8.
* ``data/external/hf_discovery/operator_redistribution_policy.json`` —
  the operator policy file (default state: zero grants, ``deny_all``).

It produces:

* ``data/external/hf_discovery/operator_policy_status.json`` — a
  deterministic per-dataset status rollup for every candidate whose
  ``recommended_action`` is ``inspect_manually_license_blocked``. For
  each candidate the rollup records:

    - dataset_id
    - recommended_action (inspect_manually_license_blocked)
    - license_observed (expected to be ``None`` for this bucket)
    - default_scope_evaluated ("committed_normalized_sample")
    - default_permitted (False under deny_all)
    - default_reason_code
    - default_reason_detail
    - grant_recorded (False under the committed default policy)
    - what_an_operator_would_need_to_do_to_unblock

The audit is intentionally read-only: it does NOT mutate the candidate
registry, does NOT call the HF API, does NOT read HF_TOKEN, does NOT
download data. It simply documents the current decision the policy
ledger would emit for every license-blocked candidate.

Round-8 surfaced FOUR license-blocked candidates carrying real
operational / economic / infrastructure measurements
(``sasha/co2_models``, ``ohdoking/energy_consumption_by_model_and_gpu``,
``dadadada1/Inference-Performance-Dataset``,
``anon-betterbench/betterbench-inference-logs``). With the default
policy file shipped in this PR (zero grants, ``deny_all``), all four
remain DENIED — same behaviour as before this milestone landed. The
milestone adds the STRUCTURAL ability for an operator to deliberately
unblock one of them, per-dataset, with provenance, expiry, and explicit
scope. It does NOT unblock anything automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
CANDIDATES_PATH = DISC_DIR / "hf_dataset_candidates.json"
POLICY_PATH = DISC_DIR / "operator_redistribution_policy.json"
STATUS_OUT_PATH = DISC_DIR / "operator_policy_status.json"

# Import as a module so tests can reach the same code path. The
# sys.path manipulation is intentional for standalone-script usage.
sys.path.insert(0, str(REPO_ROOT))
from aurelius.ingestion.operator_redistribution_policy import (  # noqa: E402, I001
    OperatorPolicyLedger,
    SUPPORTED_SCOPES,
)

logger = logging.getLogger(__name__)


# The audit is intentionally narrow: it documents the policy decision for
# the closed set of recommended_action values that map to "license=None
# blocks redistribution". The script does NOT enumerate every candidate
# in the registry — only the ones already classified as license-blocked.
LICENSE_BLOCKED_RECOMMENDED_ACTIONS: frozenset[str] = frozenset(
    {
        "inspect_manually_license_blocked",
    }
)

DEFAULT_SCOPE_EVALUATED = "committed_normalized_sample"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover — git not available
        pass
    return ""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _candidate_license_blocked_records(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidates:
        action = c.get("recommended_action")
        if action in LICENSE_BLOCKED_RECOMMENDED_ACTIONS:
            out.append(c)
    out.sort(key=lambda c: c.get("dataset_id", ""))
    return out


def build_status_payload(
    *,
    candidates: list[dict[str, Any]],
    ledger: OperatorPolicyLedger,
    now_iso: str,
    git_sha: str,
) -> dict[str, Any]:
    """Pure-function builder so tests can call it with synthetic input."""

    license_blocked = _candidate_license_blocked_records(candidates)

    per_dataset: list[dict[str, Any]] = []
    permitted_count = 0
    denied_count = 0
    for c in license_blocked:
        dsid = c["dataset_id"]
        decision = ledger.permits_redistribution(
            dsid,
            DEFAULT_SCOPE_EVALUATED,
            now_iso=now_iso,
        )
        grant = ledger.find_grant(dsid)
        per_dataset.append(
            {
                "dataset_id": dsid,
                "recommended_action": c.get("recommended_action"),
                "license_observed": c.get("license"),
                "default_scope_evaluated": DEFAULT_SCOPE_EVALUATED,
                "default_permitted": decision.permitted,
                "default_reason_code": decision.reason_code,
                "default_reason_detail": decision.reason_detail,
                "grant_recorded": grant is not None,
                "what_an_operator_would_need_to_do_to_unblock": (
                    "Add a grant entry to "
                    "data/external/hf_discovery/operator_redistribution_policy.json "
                    "with dataset_id=" + repr(dsid) + ", granted=true, a "
                    "non-empty granted_by, granted_at_iso, allowed_scopes "
                    "including 'committed_normalized_sample' (or another "
                    "supported scope), and explicit notes describing the "
                    "redistribution rationale. The default policy_default "
                    "remains 'deny_all'; this PR does NOT grant any "
                    "dataset by default."
                ),
            }
        )
        if decision.permitted:
            permitted_count += 1
        else:
            denied_count += 1

    return {
        "doc_version": "operator_policy_status_v1",
        "audited_at_iso": now_iso,
        "git_sha": git_sha,
        "scope": (
            "Read-only audit of the operator redistribution policy decision "
            "for every HF candidate whose recommended_action is "
            "inspect_manually_license_blocked. Does NOT mutate the "
            "candidate registry. Does NOT call the HF API. Does NOT "
            "download data. Does NOT read HF_TOKEN."
        ),
        "policy_default": ledger.policy_default,
        "policy_doc_version": ledger.doc_version,
        "policy_grant_count": len(ledger.grants),
        "supported_scopes": sorted(SUPPORTED_SCOPES),
        "default_scope_evaluated": DEFAULT_SCOPE_EVALUATED,
        "license_blocked_candidate_count": len(license_blocked),
        "license_blocked_permitted_under_default_policy": permitted_count,
        "license_blocked_denied_under_default_policy": denied_count,
        "per_dataset": per_dataset,
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "ingests_new_hf_data": False,
        "default_policy_is_unchanged_from_pre_milestone": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--now-iso",
        default=None,
        help="ISO-8601 UTC timestamp override (for test reproducibility).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if not CANDIDATES_PATH.exists():
        logger.error("Missing candidate registry: %s", CANDIDATES_PATH)
        return 2
    if not POLICY_PATH.exists():
        logger.error("Missing operator policy file: %s", POLICY_PATH)
        return 2

    cand = _read_json(CANDIDATES_PATH)
    ledger = OperatorPolicyLedger.load(POLICY_PATH)

    now_iso = args.now_iso or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    payload = build_status_payload(
        candidates=cand.get("candidates", []),
        ledger=ledger,
        now_iso=now_iso,
        git_sha=_git_sha(),
    )

    logger.info(
        "operator-policy audit: %d license-blocked candidates; %d "
        "permitted under default policy; %d denied; %d grants in ledger",
        payload["license_blocked_candidate_count"],
        payload["license_blocked_permitted_under_default_policy"],
        payload["license_blocked_denied_under_default_policy"],
        payload["policy_grant_count"],
    )

    if args.dry_run:
        logger.info("DRY-RUN — would write %s", STATUS_OUT_PATH)
        return 0

    STATUS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote operator-policy audit at %s", STATUS_OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
