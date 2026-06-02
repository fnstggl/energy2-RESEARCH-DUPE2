"""Tests pinning that ``scripts/ingest_hf_lightcap_runtime_telemetry.py``
is wired through the canonical :func:`decide_redistribution` gate.

Ninth consumer of the gate (after
``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``,
``scripts/ingest_hf_h200_quantization.py``,
``scripts/ingest_hf_llm_energy_consumption.py``,
``scripts/ingest_hf_latency_benchmarks.py``,
``scripts/ingest_hf_optimum_benchmark.py``, and
``scripts/ingest_hf_acmetrace.py``).

The pre-wiring shape carried a hard-coded ``"license": "cc-by-4.0"``
inside ``audit_one`` and wrote it directly into ``summary.json`` with
no gate consultation. The refactor lifts the license tag (and its
provenance + scope) to module-level constants for cc-by-4.0 (every
target shares one tag — Lightcap/agent-runtime-telemetry-small is the
only ingested dataset; the four configs are operations / tool_summary
/ operation_events / audit_records), routes the verdict through the
canonical gate, and writes the gate-derived fields additively onto
every per-config summary.json + the top-level ingest-summary JSON.

This file pins that the script now:

* declares ``LICENSE_TAG`` / ``LICENSE_SOURCE`` / ``GATE_SCOPE`` at
  module level (so a future HF tag change is a one-line edit);
* imports ``decide_redistribution`` from the canonical gate and does
  NOT redeclare the closed permissive allow-list;
* derives ``license_redistribution_status`` from the gate;
* records the gate verdict on every per-config summary.json and on
  every config row of the audit summary;
* refreshes the audit summary to ``v2`` with the top-level
  ``redistribution_gate_*`` triple (scope / policy default / grant
  count);
* keeps the on-disk fixture bytes and the existing committed
  normalised sample bytes byte-for-byte unchanged.

Audit-only — every test reads committed artefacts or runs pure-Python
decision functions. No HF API, no HF_TOKEN read, no data download.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "ingest_hf_lightcap_runtime_telemetry.py"
)
REFRESH_PATH = (
    REPO_ROOT / "scripts" / "refresh_hf_lightcap_gate_metadata.py"
)
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
AUDIT_PATH = (
    DISC_DIR / "lightcap_runtime_telemetry_ingest_summary.json"
)
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "Lightcap/agent-runtime-telemetry-small"
SAFE_NAME = "Lightcap__agent-runtime-telemetry-small"

COMMITTED_CONFIGS = (
    "operations",
    "tool_summary",
    "operation_events",
    "audit_records",
)


def _summary_path(config: str) -> Path:
    return HF_DIR / SAFE_NAME / config / "processed" / "summary.json"


def _normalized_sample_path(config: str) -> Path:
    return HF_DIR / SAFE_NAME / config / "processed" / "normalized_sample.jsonl"


def _fixture_path(config: str) -> Path:
    return FIXTURES_DIR / f"{SAFE_NAME}__{config}_sample.jsonl"


def _load_module_directly(rel_path: str, name: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, rel_path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script_module():
    return _load_module_directly(
        "scripts/ingest_hf_lightcap_runtime_telemetry.py",
        "ingest_hf_lightcap_runtime_telemetry_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_lightcap_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_lightcap_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


# ---------------------------------------------------------------------------
# 1. Module-level license constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_license_constants(script_module):
    """The license tag + provenance + scope live at module level, not
    inline inside ``audit_one`` / the summary writer.

    The four Lightcap targets all share one license (cc-by-4.0), so a
    single ``LICENSE_TAG`` constant suffices (like AcmeTrace; unlike
    the multi-license latency_benchmarks script which carries three
    tag constants).
    """

    assert script_module.DATASET_ID == DATASET_ID
    assert script_module.LICENSE_TAG == "cc-by-4.0"
    assert script_module.LICENSE_SOURCE == (
        "HF card frontmatter license: cc-by-4.0 "
        "(Faruk Alpay / Lightcap — agent-runtime-telemetry-small)"
    )
    assert script_module.GATE_SCOPE == "committed_normalized_sample"


# ---------------------------------------------------------------------------
# 2. Script imports the canonical gate (no duplicated classifier)
# ---------------------------------------------------------------------------


def test_script_imports_decide_redistribution(script_source: str):
    """A future maintainer who silently re-introduces a hard-coded
    classifier inside the script must trip this test.
    """

    assert (
        "from aurelius.ingestion.redistribution_gate import"
        in script_source
    ), "script must import decide_redistribution from the canonical gate"
    assert "decide_redistribution" in script_source
    assert "OperatorPolicyLedger" in script_source, (
        "script must load the operator policy ledger"
    )


def test_script_does_not_redeclare_permissive_set(script_source: str):
    """Confidence rail: no second copy of the closed permissive
    allow-list. The gate is the single source of truth.
    """

    forbidden = [
        '"permissive_apache_2_0":',
        '"permissive_cc_by_4_0":',
        '"permissive_cdla_2":',
        '"permissive_mit":',
        '"permissive_cc_by_sa_4_0":',
    ]
    hits = [f for f in forbidden if f in script_source]
    assert not hits, (
        f"script carries duplicated permissive allow-list: {hits!r}. "
        f"Delete and call classify_license / decide_redistribution."
    )


def test_script_does_not_hardcode_status_code_in_code(script_source: str):
    """The canonical status string ``"permissive_cc_by_4_0"`` must not
    appear inline in the script's executable code — the gate produces
    it. Docstring mentions are allowed.
    """

    tree = ast.parse(script_source)
    offending: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == "permissive_cc_by_4_0"
        ):
            offending.append((node.lineno, node.value))
    # Filter docstring occurrences.
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            body = getattr(node, "body", []) or []
            doc = (
                body[0]
                if body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
                else None
            )
            if doc and ("permissive_cc_by_4_0" in doc.value.value):
                for (ln, v) in list(offending):
                    if (
                        ln == doc.lineno
                        and v == "permissive_cc_by_4_0"
                    ):
                        offending.remove((ln, v))
    assert not offending, (
        f"script hard-codes 'permissive_cc_by_4_0' at lines "
        f"{[ln for (ln, _) in offending]!r}; the gate produces it"
    )


# ---------------------------------------------------------------------------
# 3. evaluate_redistribution — pure function returns the gate verdict
# ---------------------------------------------------------------------------


def test_evaluate_redistribution_returns_gate_decision_type(
    script_module, policy_module,
):
    """``evaluate_redistribution`` returns the gate's
    ``RedistributionGateDecision`` dataclass."""

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert type(decision).__name__ == "RedistributionGateDecision"
    for field in (
        "permitted", "reason_code", "reason_detail",
        "license_status", "license_observed", "scope",
        "operator_grant_dataset_id",
    ):
        assert hasattr(decision, field), (
            f"gate decision missing field {field!r}"
        )


def test_evaluate_redistribution_default_permits_under_empty_ledger(
    script_module, policy_module,
):
    """The default ``license=cc-by-4.0`` permits under any ledger — the
    gate short-circuits the ledger because the license is on the closed
    permissive allow-list.

    This is the NINTH gate consumer; the second one (after AcmeTrace)
    whose default license tag is a CC-BY (not apache / cdla / none /
    mit). The decision proves the cc-by-4.0 path of the gate is
    exercised here.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True
    assert decision.license_status == "permissive_cc_by_4_0"
    assert decision.reason_code == "permitted_declared_permissive_license"
    assert decision.operator_grant_dataset_id is None
    assert decision.scope == "committed_normalized_sample"
    assert decision.license_observed == "cc-by-4.0"


def test_evaluate_redistribution_swap_to_none_denies(
    script_module, policy_module,
):
    """Swap the Lightcap tag to ``None`` under the same empty ledger
    → the gate flips to DENY. Proves the wiring actually consults the
    license tag — it is not hard-coded to permit.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag=None,
    )
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"


def test_evaluate_redistribution_swap_to_restrictive_denies(
    script_module, policy_module, gate_module,
):
    """Swap the Lightcap tag to ``cc-by-nc-4.0`` (declared NON-permissive)
    → the gate denies even though cc-by-4.0 is on the permissive list.
    The closed permissive allow-list is conservative — variant tags
    are NOT auto-promoted.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag="cc-by-nc-4.0",
    )
    assert decision.permitted is False
    assert decision.license_status == "declared_non_permissive"
    assert decision.reason_code == (
        gate_module.REASON_DENIED_DECLARED_NON_PERMISSIVE_LICENSE
    )


def test_evaluate_redistribution_operator_grant_irrelevant_for_permissive(
    script_module, policy_module, gate_module,
):
    """An operator grant cannot REVOKE redistribution for an upstream
    permissive license — the gate short-circuits the ledger for
    permissive tags. Pin this here so the ninth consumer cannot
    accidentally re-introduce a ledger check that overrides the
    permissive verdict.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=DATASET_ID,
        granted=False,  # operator says "do not redistribute"
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes=(
            "operator opt-out has no effect on declared permissive "
            "licenses"
        ),
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is True, (
        "permissive license short-circuits the ledger; an operator "
        "'opt-out' grant must NOT flip the verdict to deny"
    )
    assert decision.reason_code == (
        gate_module.REASON_PERMITTED_DECLARED_PERMISSIVE_LICENSE
    )
    assert decision.operator_grant_dataset_id is None


def test_evaluate_redistribution_uses_keyword_args(script_module):
    """The script exposes ``license_tag`` / ``dataset_id`` / ``ledger``
    / ``scope`` as keyword args so tests can drive the gate path
    without invoking the parquet download / normalisation pipeline.
    """

    sig = inspect.signature(script_module.evaluate_redistribution)
    assert "license_tag" in sig.parameters
    assert "dataset_id" in sig.parameters
    assert "ledger" in sig.parameters
    assert "scope" in sig.parameters


# ---------------------------------------------------------------------------
# 4. Per-config summary.json carries the new gate-derived fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_summary_carries_redistribution_gate_metadata(config: str):
    s = json.loads(_summary_path(config).read_text())
    required = {
        "license_redistribution_status",
        "license_redistribution_source",
        "redistribution_gate_reason_code",
        "redistribution_gate_reason_detail",
        "redistribution_gate_permitted",
        "redistribution_gate_operator_grant_dataset_id",
        "redistribution_gate_scope",
    }
    missing = required - s.keys()
    assert not missing, (
        f"committed summary.json for {config} missing gate-derived "
        f"fields: {sorted(missing)!r}"
    )
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_summary_records_permit_verdict_for_cc_by_4_0(config: str):
    """The HF card declares cc-by-4.0, so the gate permits regardless
    of the ledger contents.
    """

    s = json.loads(_summary_path(config).read_text())
    assert s["license"] == "cc-by-4.0"
    assert s["redistribution_gate_permitted"] is True
    assert s["redistribution_gate_reason_code"] == (
        "permitted_declared_permissive_license"
    )
    assert s["license_redistribution_status"] == "permissive_cc_by_4_0"
    assert s["license_redistribution_source"] == (
        "HF card frontmatter license: cc-by-4.0 "
        "(Faruk Alpay / Lightcap — agent-runtime-telemetry-small)"
    )


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_status_matches_gate_classification(config: str, gate_module):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag into. Pinning this gives zero
    behavioural drift on the already-committed summaries.
    """

    s = json.loads(_summary_path(config).read_text())
    expected = gate_module.classify_license(s["license"])
    assert s["license_redistribution_status"] == expected, (
        f"summary status {s['license_redistribution_status']!r} != "
        f"gate classification {expected!r} of license tag "
        f"{s['license']!r}"
    )


# ---------------------------------------------------------------------------
# 5. Audit summary carries v2 doc_version + gate-derived fields
# ---------------------------------------------------------------------------


def test_audit_summary_doc_version_is_v2():
    """The lightcap audit summary moves to v2 here. v1 had no
    ``doc_version`` key at all (it was implicit v1); v2 adds the
    explicit ``doc_version`` plus the top-level
    ``redistribution_gate_*`` triple and the per-row gate fields.
    """

    a = json.loads(AUDIT_PATH.read_text())
    assert a["doc_version"] == (
        "lightcap_runtime_telemetry_ingest_summary_v2"
    )


def test_audit_summary_top_level_gate_metadata():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["redistribution_gate_scope"] == "committed_normalized_sample"
    assert a["redistribution_gate_policy_default"] == "deny_all"
    assert a["redistribution_gate_policy_grant_count"] == 0


def test_audit_summary_preserves_v1_invariants():
    """v2 must NOT drop any v1 invariant — v1 carried ``dataset_id``,
    ``wrote_at_s``, and a ``configs`` list. They all remain.
    """

    a = json.loads(AUDIT_PATH.read_text())
    assert a["dataset_id"] == DATASET_ID
    assert "wrote_at_s" in a
    assert "configs" in a
    assert isinstance(a["configs"], list)


def test_audit_summary_all_lightcap_rows_have_gate_fields():
    """The ninth consumer extends gate coverage to ALL lightcap rows
    — every config entry must carry the four per-row gate fields.
    """

    a = json.loads(AUDIT_PATH.read_text())
    seen = 0
    expected_configs = set(COMMITTED_CONFIGS)
    for entry in a["configs"]:
        cfg = entry.get("config") or entry.get("config_name")
        if cfg not in expected_configs:
            continue
        seen += 1
        for key in (
            "license_redistribution_status",
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
            "redistribution_gate_operator_grant_dataset_id",
        ):
            assert key in entry, (
                f"audit entry {cfg} missing {key!r}"
            )
        assert entry["license"] == "cc-by-4.0"
        assert entry["redistribution_gate_permitted"] is True
        assert entry["redistribution_gate_reason_code"] == (
            "permitted_declared_permissive_license"
        )
        assert entry["license_redistribution_status"] == (
            "permissive_cc_by_4_0"
        )
        assert entry["redistribution_gate_operator_grant_dataset_id"] is None
    assert seen == 4, (
        f"expected exactly 4 lightcap config rows (operations / "
        f"tool_summary / operation_events / audit_records), got {seen}"
    )


def test_audit_summary_preserves_v1_decision_state_rows():
    """The refresh must NOT drop the v1 ``decision_state`` /
    ``decision_tags`` / ``manifest`` / ``summary_path`` row keys.
    """

    a = json.loads(AUDIT_PATH.read_text())
    for entry in a["configs"]:
        cfg = entry.get("config") or entry.get("config_name")
        if cfg not in set(COMMITTED_CONFIGS):
            continue
        # v1 keys carried forward
        for key in (
            "audit_status", "manifest", "summary_path",
            "decision_state", "decision_tags",
        ):
            assert key in entry, (
                f"audit row {cfg} lost v1 key {key!r} during the "
                f"v2 refresh"
            )


# ---------------------------------------------------------------------------
# 6. Function signatures accept ledger as keyword arg
# ---------------------------------------------------------------------------


def test_audit_one_accepts_ledger_keyword_arg(script_module):
    sig = inspect.signature(script_module.audit_one)
    assert "ledger" in sig.parameters
    p = sig.parameters["ledger"]
    assert p.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert p.default is None


def test_load_ledger_returns_empty_when_policy_path_missing(
    script_module, tmp_path,
):
    """``_load_ledger`` falls back to ``OperatorPolicyLedger.empty()``
    when the policy file is absent — fresh-checkout self-sufficiency.
    """

    nonexistent = tmp_path / "no_such_file.json"
    assert not nonexistent.exists()
    ledger = script_module._load_ledger(nonexistent)
    assert ledger.policy_default == "deny_all"
    assert ledger.grants == ()


# ---------------------------------------------------------------------------
# 7. Safety — no HF_TOKEN literal in the refactored script
# ---------------------------------------------------------------------------


def test_no_hf_token_literal_in_script(script_source: str):
    candidates = re.findall(r"\bhf_[A-Za-z0-9]{20,}\b", script_source)
    suspicious = [
        c for c in candidates
        if any(ch.isupper() for ch in c[3:])
        and any(ch.islower() for ch in c[3:])
    ]
    assert not suspicious, (
        f"script contains an HF-token-shaped literal: {suspicious!r}"
    )
    bad_assignment = re.search(
        r'HF_TOKEN\s*=\s*["\']hf_', script_source,
    )
    assert bad_assignment is None, (
        "HF_TOKEN appears to be assigned a literal hf_ value"
    )


def test_no_hf_token_literal_in_refresh_script():
    src = REFRESH_PATH.read_text()
    candidates = re.findall(r"\bhf_[A-Za-z0-9]{20,}\b", src)
    suspicious = [
        c for c in candidates
        if any(ch.isupper() for ch in c[3:])
        and any(ch.islower() for ch in c[3:])
    ]
    assert not suspicious, (
        f"refresh helper contains an HF-token-shaped literal: "
        f"{suspicious!r}"
    )


# ---------------------------------------------------------------------------
# 8. Fixture bytes are byte-for-byte unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_fixture_sha256_matches_summary(config: str):
    """Wiring the gate must not change the on-disk fixture bytes — the
    sha256 the summary records must match what's on disk.
    """

    s = json.loads(_summary_path(config).read_text())
    p = _fixture_path(config)
    assert p.exists(), f"missing fixture for {config}"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    assert h.hexdigest() == s["sample_sha256"], (
        f"fixture for {config} bytes have drifted from recorded sha256"
    )
    assert p.stat().st_size == s["fixture_sample_bytes"]


# ---------------------------------------------------------------------------
# 9. Normalised-sample bytes are byte-for-byte unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", COMMITTED_CONFIGS)
def test_normalized_sample_sha256_matches_summary(config: str):
    """Wiring the gate must not change the committed normalised-sample
    bytes either — the sha256 the summary records must match what's
    on disk. Lightcap commits a real cc-by-4.0 normalised sample
    (the corpus's first committed analysis-tier tool_runtime_trace
    sample), so this invariant is a stronger guarantee than the
    fixture-only invariant above.
    """

    s = json.loads(_summary_path(config).read_text())
    p = _normalized_sample_path(config)
    assert p.exists(), f"missing normalized_sample.jsonl for {config}"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    assert h.hexdigest() == s["normalized_sample_sha256"], (
        f"normalized_sample.jsonl for {config} bytes have drifted "
        f"from recorded sha256"
    )
    assert p.stat().st_size == s["normalized_sample_bytes"]


# ---------------------------------------------------------------------------
# 10. Refresh helper does not invent additional rows or strip v1 fields
# ---------------------------------------------------------------------------


def test_refresh_helper_does_not_add_extra_configs():
    """The refresh helper must only update rows for the 4 declared
    targets — no silent expansion of the audit summary.
    """

    a = json.loads(AUDIT_PATH.read_text())
    configs = {
        (e.get("config") or e.get("config_name"))
        for e in a["configs"]
    }
    assert configs == set(COMMITTED_CONFIGS), (
        f"unexpected configs in audit summary: {configs}"
    )


def test_targets_table_is_the_four_committed_configs(script_module):
    """The TARGETS table is the canonical list of committed configs;
    pin it so a future drift (adding a fifth target without committing
    a fixture) trips this test.
    """

    configs = {t["config_name"] for t in script_module.TARGETS}
    assert configs == set(COMMITTED_CONFIGS)


# ---------------------------------------------------------------------------
# 11. Confidence rail: ingest script and refresh helper agree
# ---------------------------------------------------------------------------


def test_refresh_helper_and_ingest_agree_on_constants():
    """The refresh helper imports the ingest module — they must agree
    on ``LICENSE_TAG`` / ``LICENSE_SOURCE`` / ``GATE_SCOPE`` /
    ``DATASET_ID`` by construction. Pin that the refresh script's
    text doesn't accidentally duplicate the constants (a future
    refactor must keep importing them).
    """

    src = REFRESH_PATH.read_text()
    # Must use the ingest module's constants, not redeclare them.
    assert "ingest.LICENSE_TAG" in src or "ingest.LICENSE_SOURCE" in src
    assert "ingest.GATE_SCOPE" in src
    assert "ingest.DATASET_ID" in src
    # Must not redeclare the literal status code.
    assert '"permissive_cc_by_4_0"' not in src


def test_refresh_helper_writes_v2_doc_version():
    """The refresh helper must stamp ``doc_version`` =
    ``lightcap_runtime_telemetry_ingest_summary_v2`` on the rewritten
    audit summary. Pin the literal so a future refactor cannot
    silently downgrade to v1.
    """

    src = REFRESH_PATH.read_text()
    assert "lightcap_runtime_telemetry_ingest_summary_v2" in src
