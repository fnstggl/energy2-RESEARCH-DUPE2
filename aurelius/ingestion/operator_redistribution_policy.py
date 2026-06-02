"""Operator redistribution policy for license=None HF datasets.

Discovery / data-engine infrastructure. No scheduler change, no controller
change, no robust-energy-engine change, no production claim, no Tier 1
promotion, no HF ingestion, no HF_TOKEN use, no raw data download.

Why this module exists
----------------------

The Round-5/6/7/8 HF discovery audits surfaced public HF datasets that carry
REAL operational / economic / infrastructure measurements but have
``license = None`` on the HF card. Under the conservative redistribution
policy applied by the discovery pipeline, ``license = None`` means the
dataset cannot be ingested into the committed federated corpus (no
committed normalised sample) because there is no declared promise that
redistribution is permitted.

The Round-8 audit explicitly deferred this milestone:

    "Recorded as inspect_manually_license_blocked. Follow-up: contact owner
    to request a permissive license, OR ingest via operator-policy
    permission-flow once that milestone lands."

This module IS that milestone. It does NOT relax the default safety
posture. It defines a structural policy framework where an operator can
deliberately, per-dataset, record explicit redistribution consent with
provenance (operator id, grant timestamp, expiry, allowed scopes, notes).
With NO operator policy entry, every ``license = None`` dataset remains
DENIED — same behaviour as before this module landed.

Design rules
------------

1. **Default DENY**. ``permits_redistribution(dataset_id, scope)`` returns
   ``(False, reason)`` for any dataset with no matching grant.
2. **No silent grants**. A grant must explicitly set ``granted = True``,
   ``granted_by`` non-empty, ``granted_at_iso`` valid, and the requested
   ``scope`` must appear in the entry's ``allowed_scopes`` list.
3. **Expiry honoured**. ``expires_at_iso`` strictly less than the
   evaluation timestamp denies the grant.
4. **Per-dataset**. There is no "deny one, allow all". Every grant is
   keyed by the exact HF ``dataset_id`` (e.g. ``sasha/co2_models``).
5. **Provenance required**. ``granted_by`` and ``granted_at_iso`` are
   required on every grant. Missing fields => DENY.
6. **Scope is closed-set**. The supported scopes are listed in
   :data:`SUPPORTED_SCOPES`. A scope not in the closed set => DENY.
7. **No HF I/O**. This module reads only local JSON. It never calls the
   HF API, never reads ``HF_TOKEN``, never downloads any file.
8. **Backwards compatible**. The default committed policy file has zero
   grants. With zero grants, behaviour of the discovery pipeline is
   exactly as before this module landed.

Supported scopes
----------------

* ``committed_normalized_sample`` — operator consents to a bounded
  normalised sample being committed to the federated corpus under the
  same size + checksum + provenance rules as a permissively-licensed
  dataset.
* ``bounded_ingestion`` — operator consents to a bounded ingestion run
  (download + schema profile + summary + fixture) without necessarily
  committing the normalised sample to git.
* ``schema_only`` — operator consents to inspecting and recording only
  the schema profile + summary, no sample data.

A grant lists which subset of scopes the operator has consented to. A
request for an unlisted scope is DENIED even if the grant is otherwise
valid.

This module is intentionally small and dependency-free. It does NOT alter
the existing ingestion / discovery pipeline. Round-9+ scripts may choose
to consult this policy when handling ``recommended_action ==
inspect_manually_license_blocked`` candidates; until those scripts add an
explicit hook, the default behaviour is unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

POLICY_DOC_VERSION = "operator_redistribution_policy_v1"

SUPPORTED_SCOPES: frozenset[str] = frozenset(
    {
        "committed_normalized_sample",
        "bounded_ingestion",
        "schema_only",
    }
)


@dataclass(frozen=True)
class OperatorGrant:
    """One operator-recorded redistribution grant for one HF dataset.

    Every field is required. Missing or empty required fields => the
    grant is INVALID and the loader will refuse it. ``granted = True``
    is required for the grant to permit anything; ``granted = False``
    entries are recorded as explicit denials and never permit any scope.
    """

    dataset_id: str
    granted: bool
    granted_by: str
    granted_at_iso: str
    allowed_scopes: tuple[str, ...]
    notes: str
    expires_at_iso: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "OperatorGrant":
        return OperatorGrant(
            dataset_id=str(d["dataset_id"]),
            granted=bool(d["granted"]),
            granted_by=str(d.get("granted_by") or ""),
            granted_at_iso=str(d.get("granted_at_iso") or ""),
            allowed_scopes=tuple(d.get("allowed_scopes") or ()),
            notes=str(d.get("notes") or ""),
            expires_at_iso=(
                str(d["expires_at_iso"]) if d.get("expires_at_iso") else None
            ),
        )


@dataclass(frozen=True)
class PolicyDecision:
    """Result of ``permits_redistribution(dataset_id, scope)``."""

    permitted: bool
    reason_code: str
    reason_detail: str
    matched_grant_dataset_id: Optional[str]


# Reason codes — closed set so tests can assert without string-matching the
# detail message.
REASON_NO_GRANT = "no_grant_recorded"
REASON_GRANTED_FALSE = "grant_explicitly_denies"
REASON_MISSING_PROVENANCE = "grant_missing_provenance"
REASON_EXPIRED = "grant_expired"
REASON_SCOPE_NOT_ALLOWED = "requested_scope_not_in_allowed_scopes"
REASON_UNSUPPORTED_SCOPE = "requested_scope_not_in_supported_scopes"
REASON_PERMITTED = "permitted"


@dataclass(frozen=True)
class OperatorPolicyLedger:
    """In-memory view of the operator redistribution policy file.

    Use :meth:`load` to read the canonical JSON file. Use
    :meth:`permits_redistribution` to ask whether one dataset_id may be
    redistributed under one scope.
    """

    doc_version: str
    policy_default: str
    grants: tuple[OperatorGrant, ...]
    loaded_from: Optional[Path] = field(default=None)

    # ----- loaders -----------------------------------------------------

    @staticmethod
    def empty() -> "OperatorPolicyLedger":
        """Return the documented default ledger: deny-all, zero grants."""

        return OperatorPolicyLedger(
            doc_version=POLICY_DOC_VERSION,
            policy_default="deny_all",
            grants=(),
            loaded_from=None,
        )

    @staticmethod
    def load(path: Path) -> "OperatorPolicyLedger":
        """Load a policy file from disk.

        Raises ``FileNotFoundError`` if missing; ``ValueError`` if the
        file is malformed or carries a wrong doc_version / policy_default.
        Invalid grant entries (missing required fields, unsupported
        scopes, malformed timestamps) raise ``ValueError`` — there is no
        silent skip, because a silent skip on a "deny" entry could
        accidentally widen redistribution.
        """

        if not path.exists():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"operator policy file must be a JSON object, got {type(raw)!r}"
            )

        doc_version = raw.get("doc_version")
        if doc_version != POLICY_DOC_VERSION:
            raise ValueError(
                f"unexpected doc_version {doc_version!r} "
                f"(expected {POLICY_DOC_VERSION!r})"
            )

        policy_default = raw.get("policy_default")
        if policy_default != "deny_all":
            raise ValueError(
                f"policy_default must be 'deny_all' "
                f"(got {policy_default!r}); deny_all is the only "
                f"safe default and cannot be widened by this module"
            )

        grants_raw = raw.get("grants", [])
        if not isinstance(grants_raw, list):
            raise ValueError(
                f"grants must be a JSON array, got {type(grants_raw)!r}"
            )

        seen_ids: set[str] = set()
        grants: list[OperatorGrant] = []
        for i, g in enumerate(grants_raw):
            if not isinstance(g, dict):
                raise ValueError(f"grants[{i}] must be a JSON object")
            try:
                grant = OperatorGrant.from_dict(g)
            except KeyError as e:
                raise ValueError(
                    f"grants[{i}] missing required field: {e}"
                ) from e
            _validate_grant_structure(grant, index=i)
            if grant.dataset_id in seen_ids:
                raise ValueError(
                    f"grants[{i}] duplicate dataset_id "
                    f"{grant.dataset_id!r} — duplicates are disallowed"
                )
            seen_ids.add(grant.dataset_id)
            grants.append(grant)

        return OperatorPolicyLedger(
            doc_version=doc_version,
            policy_default=policy_default,
            grants=tuple(grants),
            loaded_from=path,
        )

    # ----- decision API ------------------------------------------------

    def permits_redistribution(
        self,
        dataset_id: str,
        scope: str,
        *,
        now_iso: Optional[str] = None,
    ) -> PolicyDecision:
        """Decide whether ``dataset_id`` may be redistributed under ``scope``.

        The decision is deterministic and depends only on the loaded
        ledger and ``now_iso`` (defaulting to UTC now). The function
        never performs I/O.

        ``now_iso`` is exposed for test reproducibility; production
        callers should leave it unset.
        """

        if scope not in SUPPORTED_SCOPES:
            return PolicyDecision(
                permitted=False,
                reason_code=REASON_UNSUPPORTED_SCOPE,
                reason_detail=(
                    f"scope {scope!r} is not in the supported scopes "
                    f"{sorted(SUPPORTED_SCOPES)!r}"
                ),
                matched_grant_dataset_id=None,
            )

        grant = self._find_grant(dataset_id)
        if grant is None:
            return PolicyDecision(
                permitted=False,
                reason_code=REASON_NO_GRANT,
                reason_detail=(
                    f"no operator grant recorded for {dataset_id!r}; "
                    f"default policy is deny_all"
                ),
                matched_grant_dataset_id=None,
            )

        if not grant.granted:
            return PolicyDecision(
                permitted=False,
                reason_code=REASON_GRANTED_FALSE,
                reason_detail=(
                    f"operator grant for {dataset_id!r} is explicit DENY "
                    f"(granted=false); recorded by "
                    f"{grant.granted_by or '<missing>'}"
                ),
                matched_grant_dataset_id=grant.dataset_id,
            )

        if not grant.granted_by or not grant.granted_at_iso:
            return PolicyDecision(
                permitted=False,
                reason_code=REASON_MISSING_PROVENANCE,
                reason_detail=(
                    f"operator grant for {dataset_id!r} is incomplete: "
                    f"granted_by={grant.granted_by!r}, "
                    f"granted_at_iso={grant.granted_at_iso!r}"
                ),
                matched_grant_dataset_id=grant.dataset_id,
            )

        if grant.expires_at_iso is not None:
            now_str = now_iso or time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            if grant.expires_at_iso < now_str:
                return PolicyDecision(
                    permitted=False,
                    reason_code=REASON_EXPIRED,
                    reason_detail=(
                        f"operator grant for {dataset_id!r} expired at "
                        f"{grant.expires_at_iso!r} (now {now_str!r})"
                    ),
                    matched_grant_dataset_id=grant.dataset_id,
                )

        if scope not in grant.allowed_scopes:
            return PolicyDecision(
                permitted=False,
                reason_code=REASON_SCOPE_NOT_ALLOWED,
                reason_detail=(
                    f"operator grant for {dataset_id!r} does not include "
                    f"scope {scope!r}; allowed_scopes="
                    f"{list(grant.allowed_scopes)!r}"
                ),
                matched_grant_dataset_id=grant.dataset_id,
            )

        return PolicyDecision(
            permitted=True,
            reason_code=REASON_PERMITTED,
            reason_detail=(
                f"operator grant for {dataset_id!r} permits scope "
                f"{scope!r}; granted_by={grant.granted_by!r} "
                f"granted_at_iso={grant.granted_at_iso!r}"
            ),
            matched_grant_dataset_id=grant.dataset_id,
        )

    def find_grant(self, dataset_id: str) -> Optional[OperatorGrant]:
        """Public accessor used by audit tooling. Read-only."""

        return self._find_grant(dataset_id)

    def _find_grant(self, dataset_id: str) -> Optional[OperatorGrant]:
        for g in self.grants:
            if g.dataset_id == dataset_id:
                return g
        return None


def _validate_grant_structure(grant: OperatorGrant, *, index: int) -> None:
    if not grant.dataset_id:
        raise ValueError(f"grants[{index}] dataset_id is empty")
    if grant.granted and not grant.granted_by:
        raise ValueError(
            f"grants[{index}] granted=true requires non-empty granted_by"
        )
    if grant.granted and not grant.granted_at_iso:
        raise ValueError(
            f"grants[{index}] granted=true requires non-empty granted_at_iso"
        )
    for s in grant.allowed_scopes:
        if s not in SUPPORTED_SCOPES:
            raise ValueError(
                f"grants[{index}] allowed_scope {s!r} not in "
                f"SUPPORTED_SCOPES={sorted(SUPPORTED_SCOPES)!r}"
            )
    if grant.expires_at_iso is not None and not grant.expires_at_iso:
        raise ValueError(
            f"grants[{index}] expires_at_iso must be either omitted "
            f"or a non-empty ISO-8601 string"
        )


__all__ = [
    "POLICY_DOC_VERSION",
    "SUPPORTED_SCOPES",
    "OperatorGrant",
    "OperatorPolicyLedger",
    "PolicyDecision",
    "REASON_NO_GRANT",
    "REASON_GRANTED_FALSE",
    "REASON_MISSING_PROVENANCE",
    "REASON_EXPIRED",
    "REASON_SCOPE_NOT_ALLOWED",
    "REASON_UNSUPPORTED_SCOPE",
    "REASON_PERMITTED",
]
