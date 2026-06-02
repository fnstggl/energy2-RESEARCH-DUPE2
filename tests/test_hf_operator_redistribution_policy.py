"""Tests for the operator redistribution policy + license-blocked audit.

Discovery / data-engine PR. The policy module is a structural piece of
safety infrastructure: by default it DENIES redistribution for every
license=None HF dataset, exactly mirroring the behaviour the discovery
pipeline already enforced before this milestone landed. These tests
pin that safety posture so a future grant entry cannot accidentally
widen redistribution without explicit operator action.

The tests cover:

1. The committed default policy file exists, has the right
   doc_version + policy_default, and contains ZERO grants.
2. The policy loader rejects malformed files (wrong doc_version,
   wrong policy_default, malformed grants, duplicate ids, unsupported
   scopes).
3. The decision API denies under every failure mode (no grant,
   granted=false, missing provenance, expired, scope not allowed,
   unsupported scope) and ONLY permits when every check passes.
4. The four Round-8 license-blocked candidates are explicitly DENIED
   under the committed default policy.
5. The audit script produces a deterministic JSON for fixed inputs.
6. No HF_TOKEN appears anywhere in the committed policy / audit /
   module files; no raw data is committed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module_directly(rel_path: str, name: str):
    """Load a module from its file path without going through aurelius.*.

    Lets these tests pass even in stripped-down environments that lack
    aurelius' heavy data-science deps (pandas, sqlalchemy, ...).
    """

    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, rel_path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "policy_module_under_test",
    )


@pytest.fixture(scope="module")
def audit_module():
    # The audit script imports the policy module via
    # ``from aurelius.ingestion.operator_redistribution_policy import ...``,
    # which exercises the lazy ``__init__`` we shipped in this PR.
    # Insert REPO_ROOT so that import resolves the same way it does
    # when the script runs as ``python3 scripts/audit_hf_operator_policy_status.py``.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return _load_module_directly(
        "scripts/audit_hf_operator_policy_status.py",
        "audit_hf_operator_policy_status_under_test",
    )


# ---------------------------------------------------------------------------
# 1. Default committed policy file invariants
# ---------------------------------------------------------------------------


def test_default_policy_file_exists() -> None:
    path = REPO_ROOT / "data" / "external" / "hf_discovery" / (
        "operator_redistribution_policy.json"
    )
    assert path.exists(), "operator_redistribution_policy.json must be committed"


def test_default_policy_file_invariants(policy_module) -> None:
    path = REPO_ROOT / "data" / "external" / "hf_discovery" / (
        "operator_redistribution_policy.json"
    )
    raw = json.loads(path.read_text())
    assert raw["doc_version"] == policy_module.POLICY_DOC_VERSION
    assert raw["policy_default"] == "deny_all", (
        "policy_default must remain 'deny_all'; any change widens "
        "redistribution and requires deliberate review"
    )
    assert raw["grants"] == [], (
        "the committed policy file must ship with ZERO grants — every "
        "license=None dataset stays denied by default"
    )


def test_default_policy_file_is_bounded_in_size() -> None:
    path = REPO_ROOT / "data" / "external" / "hf_discovery" / (
        "operator_redistribution_policy.json"
    )
    size = path.stat().st_size
    # Bounded: the committed policy is purely metadata. 16 KiB is more
    # than enough headroom for the explanatory keys.
    assert size <= 16 * 1024, f"policy file is {size} bytes (>16 KiB)"


def test_default_policy_loads_through_canonical_loader(policy_module) -> None:
    path = REPO_ROOT / "data" / "external" / "hf_discovery" / (
        "operator_redistribution_policy.json"
    )
    ledger = policy_module.OperatorPolicyLedger.load(path)
    assert ledger.policy_default == "deny_all"
    assert ledger.doc_version == policy_module.POLICY_DOC_VERSION
    assert ledger.grants == ()
    assert ledger.loaded_from == path


# ---------------------------------------------------------------------------
# 2. Loader rejects malformed input
# ---------------------------------------------------------------------------


def _write_policy(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(doc))
    return p


def test_loader_rejects_wrong_doc_version(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": "operator_redistribution_policy_v0",
            "policy_default": "deny_all",
            "grants": [],
        },
    )
    with pytest.raises(ValueError, match="doc_version"):
        policy_module.OperatorPolicyLedger.load(p)


def test_loader_rejects_widening_policy_default(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "allow_all",
            "grants": [],
        },
    )
    with pytest.raises(ValueError, match="policy_default"):
        policy_module.OperatorPolicyLedger.load(p)


def test_loader_rejects_non_object(policy_module, tmp_path) -> None:
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(ValueError, match="JSON object"):
        policy_module.OperatorPolicyLedger.load(p)


def test_loader_rejects_grants_not_array(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": {"oops": "not an array"},
        },
    )
    with pytest.raises(ValueError, match="grants must be a JSON array"):
        policy_module.OperatorPolicyLedger.load(p)


def test_loader_rejects_duplicate_dataset_id(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": ["schema_only"],
                    "notes": "",
                },
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": ["schema_only"],
                    "notes": "",
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        policy_module.OperatorPolicyLedger.load(p)


def test_loader_rejects_unsupported_scope_in_grant(
    policy_module, tmp_path
) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": ["totally_unsupported_scope"],
                    "notes": "",
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="SUPPORTED_SCOPES"):
        policy_module.OperatorPolicyLedger.load(p)


def test_loader_rejects_granted_without_granted_by(
    policy_module, tmp_path
) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": ["schema_only"],
                    "notes": "",
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="granted_by"):
        policy_module.OperatorPolicyLedger.load(p)


# ---------------------------------------------------------------------------
# 3. Decision API failure modes
# ---------------------------------------------------------------------------


def test_decision_denies_unknown_dataset(policy_module) -> None:
    ledger = policy_module.OperatorPolicyLedger.empty()
    d = ledger.permits_redistribution(
        "not/in/ledger",
        "committed_normalized_sample",
        now_iso="2026-01-01T00:00:00Z",
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_NO_GRANT
    assert d.matched_grant_dataset_id is None


def test_decision_denies_unsupported_scope(policy_module) -> None:
    ledger = policy_module.OperatorPolicyLedger.empty()
    d = ledger.permits_redistribution(
        "anything",
        "totally_unsupported_scope",
        now_iso="2026-01-01T00:00:00Z",
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_UNSUPPORTED_SCOPE


def test_decision_denies_explicit_false_grant(policy_module, tmp_path) -> None:
    # granted=false with empty granted_by is allowed (it's a record of
    # an explicit deny, not a permitting grant).
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": False,
                    "granted_by": "",
                    "granted_at_iso": "",
                    "allowed_scopes": [],
                    "notes": "operator declined redistribution",
                },
            ],
        },
    )
    ledger = policy_module.OperatorPolicyLedger.load(p)
    d = ledger.permits_redistribution(
        "x/y",
        "committed_normalized_sample",
        now_iso="2026-01-01T00:00:00Z",
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_GRANTED_FALSE
    assert d.matched_grant_dataset_id == "x/y"


def test_decision_denies_expired_grant(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice",
                    "granted_at_iso": "2024-01-01T00:00:00Z",
                    "allowed_scopes": ["committed_normalized_sample"],
                    "notes": "",
                    "expires_at_iso": "2025-01-01T00:00:00Z",
                },
            ],
        },
    )
    ledger = policy_module.OperatorPolicyLedger.load(p)
    d = ledger.permits_redistribution(
        "x/y",
        "committed_normalized_sample",
        now_iso="2026-06-02T00:00:00Z",
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_EXPIRED


def test_decision_denies_scope_not_in_allowed(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": ["schema_only"],
                    "notes": "",
                },
            ],
        },
    )
    ledger = policy_module.OperatorPolicyLedger.load(p)
    d = ledger.permits_redistribution(
        "x/y",
        "committed_normalized_sample",
        now_iso="2026-06-02T00:00:00Z",
    )
    assert d.permitted is False
    assert d.reason_code == policy_module.REASON_SCOPE_NOT_ALLOWED


def test_decision_permits_only_with_complete_valid_grant(
    policy_module, tmp_path
) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice@operator.example",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": [
                        "committed_normalized_sample",
                        "bounded_ingestion",
                    ],
                    "notes": "operator confirmed redistribution",
                    "expires_at_iso": "2030-01-01T00:00:00Z",
                },
            ],
        },
    )
    ledger = policy_module.OperatorPolicyLedger.load(p)
    d = ledger.permits_redistribution(
        "x/y",
        "committed_normalized_sample",
        now_iso="2026-06-02T00:00:00Z",
    )
    assert d.permitted is True
    assert d.reason_code == policy_module.REASON_PERMITTED
    assert d.matched_grant_dataset_id == "x/y"


def test_decision_no_expiry_means_never_expired(policy_module, tmp_path) -> None:
    p = _write_policy(
        tmp_path,
        {
            "doc_version": policy_module.POLICY_DOC_VERSION,
            "policy_default": "deny_all",
            "grants": [
                {
                    "dataset_id": "x/y",
                    "granted": True,
                    "granted_by": "alice",
                    "granted_at_iso": "2026-01-01T00:00:00Z",
                    "allowed_scopes": ["schema_only"],
                    "notes": "",
                },
            ],
        },
    )
    ledger = policy_module.OperatorPolicyLedger.load(p)
    d = ledger.permits_redistribution(
        "x/y", "schema_only", now_iso="2099-01-01T00:00:00Z"
    )
    assert d.permitted is True


# ---------------------------------------------------------------------------
# 4. Round-8 license-blocked candidates explicitly denied under default
# ---------------------------------------------------------------------------


ROUND8_LICENSE_BLOCKED_IDS = frozenset(
    {
        "sasha/co2_models",
        "ohdoking/energy_consumption_by_model_and_gpu",
        "dadadada1/Inference-Performance-Dataset",
        "anon-betterbench/betterbench-inference-logs",
    }
)


def test_round8_license_blocked_denied_under_default_policy(
    policy_module,
) -> None:
    path = REPO_ROOT / "data" / "external" / "hf_discovery" / (
        "operator_redistribution_policy.json"
    )
    ledger = policy_module.OperatorPolicyLedger.load(path)
    for dsid in ROUND8_LICENSE_BLOCKED_IDS:
        for scope in policy_module.SUPPORTED_SCOPES:
            d = ledger.permits_redistribution(
                dsid, scope, now_iso="2026-06-02T00:00:00Z"
            )
            assert d.permitted is False, (
                f"default policy must DENY {dsid} for scope {scope}; got "
                f"permitted=True"
            )
            assert d.reason_code == policy_module.REASON_NO_GRANT


# ---------------------------------------------------------------------------
# 5. Audit script deterministic output
# ---------------------------------------------------------------------------


def test_audit_payload_is_deterministic_for_fixed_inputs(
    audit_module, policy_module
) -> None:
    # Synthetic candidate set: includes one license-blocked candidate
    # plus one non-blocked candidate that must NOT appear in per_dataset.
    candidates = [
        {
            "dataset_id": "x/y",
            "recommended_action": "inspect_manually_license_blocked",
            "license": None,
        },
        {
            "dataset_id": "a/b",
            "recommended_action": "ingest_now_bounded",
            "license": "apache-2.0",
        },
    ]
    ledger = policy_module.OperatorPolicyLedger.empty()
    p1 = audit_module.build_status_payload(
        candidates=candidates,
        ledger=ledger,
        now_iso="2026-06-02T00:00:00Z",
        git_sha="abc1234",
    )
    p2 = audit_module.build_status_payload(
        candidates=candidates,
        ledger=ledger,
        now_iso="2026-06-02T00:00:00Z",
        git_sha="abc1234",
    )
    assert p1 == p2
    assert p1["license_blocked_candidate_count"] == 1
    assert p1["license_blocked_denied_under_default_policy"] == 1
    assert p1["license_blocked_permitted_under_default_policy"] == 0
    assert len(p1["per_dataset"]) == 1
    assert p1["per_dataset"][0]["dataset_id"] == "x/y"


def test_audit_payload_safety_flags_are_set(audit_module, policy_module) -> None:
    ledger = policy_module.OperatorPolicyLedger.empty()
    p = audit_module.build_status_payload(
        candidates=[],
        ledger=ledger,
        now_iso="2026-06-02T00:00:00Z",
        git_sha="abc1234",
    )
    assert p["production_claim"] is False
    assert p["modifies_robust_energy_engine"] is False
    assert p["modifies_controllers_or_defaults"] is False
    assert p["uses_oracle_as_headline"] is False
    assert p["ingests_new_hf_data"] is False
    assert p["default_policy_is_unchanged_from_pre_milestone"] is True


def test_audit_status_file_matches_round8_finding(audit_module) -> None:
    # The committed status JSON (produced by the script) must record
    # the four Round-8 license-blocked candidates and deny them all.
    path = REPO_ROOT / "data" / "external" / "hf_discovery" / (
        "operator_policy_status.json"
    )
    assert path.exists(), (
        "operator_policy_status.json must be committed by this PR"
    )
    payload = json.loads(path.read_text())
    assert payload["doc_version"] == "operator_policy_status_v1"
    assert payload["policy_default"] == "deny_all"
    assert payload["policy_grant_count"] == 0
    assert payload["license_blocked_candidate_count"] == 4
    assert payload["license_blocked_permitted_under_default_policy"] == 0
    assert payload["license_blocked_denied_under_default_policy"] == 4

    ids_in_audit = {row["dataset_id"] for row in payload["per_dataset"]}
    assert ids_in_audit == ROUND8_LICENSE_BLOCKED_IDS, ids_in_audit

    for row in payload["per_dataset"]:
        assert row["default_permitted"] is False
        assert row["default_reason_code"] == "no_grant_recorded"
        assert row["grant_recorded"] is False
        assert row["recommended_action"] == "inspect_manually_license_blocked"


# ---------------------------------------------------------------------------
# 6. No secrets, no raw data
# ---------------------------------------------------------------------------


PR_TOUCHED_FILES = [
    "aurelius/ingestion/operator_redistribution_policy.py",
    "aurelius/ingestion/__init__.py",
    "scripts/audit_hf_operator_policy_status.py",
    "data/external/hf_discovery/operator_redistribution_policy.json",
    "data/external/hf_discovery/operator_policy_status.json",
    "tests/test_hf_operator_redistribution_policy.py",
]


def test_no_hf_token_in_pr_files() -> None:
    # Real HF tokens have the shape ``<prefix>_<base62 chars>`` where
    # ``<prefix>`` is the literal two-character HuggingFace prefix and
    # ``<base62 chars>`` is 20+ alphanumerics. We scan for that pattern
    # rather than the literal substring (the substring legitimately
    # appears in path segments like ``data/external/hf_discovery``).
    # We also build the literal pattern at runtime so this test file's
    # own source doesn't trigger the check.
    import re

    hf_prefix = "h" + "f"  # avoid embedding the literal in source
    token_pattern = re.compile(rf"\b{hf_prefix}_[A-Za-z0-9]{{20,}}\b")
    literal_env_marker = "HF" + "_TOKEN=" + hf_prefix + "_"

    for rel in PR_TOUCHED_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        text = p.read_text()
        assert literal_env_marker not in text, (
            f"{rel} contains literal env-style HF token assignment"
        )
        m = token_pattern.search(text)
        assert m is None, f"{rel} contains likely HF token {m.group(0)!r}"


def test_no_raw_data_committed_under_policy_paths() -> None:
    # The policy work touches data/external/hf_discovery/ — verify that
    # no large raw data files were sneaked in alongside the policy JSON.
    disc_dir = REPO_ROOT / "data" / "external" / "hf_discovery"
    if not disc_dir.exists():
        return
    # Only files directly added by this PR are bounded; we check the
    # specific files we committed.
    for name in [
        "operator_redistribution_policy.json",
        "operator_policy_status.json",
    ]:
        p = disc_dir / name
        if p.exists():
            assert p.stat().st_size <= 32 * 1024, (
                f"{name} is {p.stat().st_size} bytes (>32 KiB); policy "
                f"files should be small structural metadata, not raw data"
            )
