"""Tests pinning that ``scripts/ingest_hf_h200_quantization.py`` is wired
through the canonical :func:`decide_redistribution` gate.

Fourth consumer of the gate (after ``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``, and
``scripts/ingest_hf_agent_llm_traces.py``). The previous shape hard-coded
``committed_normalized_sample_reason_skipped =
"license_unspecified_no_redistribution_promise"`` into the summary writer
and emitted NO canonical ``license_redistribution_status`` /
``redistribution_gate_*`` fields at all. This file pins that the script
now derives the gate-canonical status code, reason code, and permit
decision from the gate, and writes the gate-derived metadata into both
the per-dataset summary.json and the round-6 cross-dataset audit
summary.

The committed normalised sample skip-reason string
(``license_unspecified_no_redistribution_promise``) is preserved
verbatim — the v1 ``test_no_committed_normalized_sample_under_unspecified_license``
in ``tests/test_hf_h200_quantization_ingest.py`` continues to pin it.
The gate fields are additive.

This is audit-only — every test reads committed artefacts or runs
pure-Python decision functions. No HF API, no HF_TOKEN read, no data
download.

Inventory
---------

1. Script declares ``LICENSE_TAG`` + ``LICENSE_SOURCE`` + ``GATE_SCOPE``
   as the single source of truth for the dataset's license metadata
   (mirrors the ``TARGETS`` shape PR #152 introduced for the gap commit
   script and the ``LICENSE_TAG`` constant PR #154 introduced for
   ``ingest_hf_agent_llm_traces.py``).
2. Script imports the gate (no duplicated license classifier).
3. Script does NOT redeclare the permissive allow-list.
4. ``evaluate_redistribution`` (pure function) returns the gate's
   ``RedistributionGateDecision`` for the dataset's tag under any
   ledger the caller supplies.
5. Under the default-empty ledger and the dataset's declared
   ``license = None`` tag the gate DENIES with reason_code
   ``no_grant_recorded`` and status
   ``unspecified_no_committed_sample`` — identical to the v1 behaviour
   (no normalised sample committed).
6. Swapping the license tag to an explicit permissive value (e.g.
   ``"mit"``) under the same default-empty ledger flips the gate to
   PERMIT with reason_code ``permitted_declared_permissive_license``
   and status ``permissive_mit`` (proves the wiring actually consults
   the gate, not a hard-coded ``deny``).
7. An in-memory operator grant for the dataset under the
   ``committed_normalized_sample`` scope flips the ``license = None``
   path to PERMIT with reason_code ``permitted_operator_grant`` (also
   proves the wiring consults the ledger).
8. The committed per-dataset summary.json carries the new
   gate-derived fields:
   ``redistribution_gate_reason_code``,
   ``redistribution_gate_reason_detail``,
   ``redistribution_gate_permitted``,
   ``redistribution_gate_operator_grant_dataset_id``,
   ``redistribution_gate_scope``,
   ``license_redistribution_status``,
   ``license_redistribution_source``.
9. The committed ``license_redistribution_status`` matches what the
   gate classifies the recorded ``license`` tag into. Backwards-compat:
   no behavioural drift on the already-committed artefacts —
   ``committed_normalized_sample_reason_skipped`` still reads
   ``"license_unspecified_no_redistribution_promise"`` and no
   normalised sample is committed.
10. The committed round-6 audit summary
    ``data/external/hf_discovery/round6_broadened_discovery_audit_summary.json``
    has ``doc_version =
    round6_broadened_discovery_audit_summary_v2`` and carries the gate
    metadata both top-level and per-dataset.
11. No HF_TOKEN literal in the refactored script.
12. ``ingest`` and ``_write_round6_audit_summary`` accept the ledger
    as a keyword-only ``Optional`` argument so the round-6
    ``main()`` can load it once and pass it through.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_hf_h200_quantization.py"
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
AUDIT_PATH = DISC_DIR / "round6_broadened_discovery_audit_summary.json"

DATASET_ID = "ssakethch/h200-quantization-benchmarks"
SAFE_DATASET = "ssakethch__h200-quantization-benchmarks"
CONFIG = "throughput"


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
        "scripts/ingest_hf_h200_quantization.py",
        "ingest_hf_h200_quantization_under_test",
    )


@pytest.fixture(scope="module")
def gate_module():
    return _load_module_directly(
        "aurelius/ingestion/redistribution_gate.py",
        "redistribution_gate_for_h200_quantization_wiring_test",
    )


@pytest.fixture(scope="module")
def policy_module():
    return _load_module_directly(
        "aurelius/ingestion/operator_redistribution_policy.py",
        "operator_policy_for_h200_quantization_wiring_test",
    )


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


def _summary_path() -> Path:
    return HF_DIR / SAFE_DATASET / CONFIG / "processed" / "summary.json"


# ---------------------------------------------------------------------------
# 1. License constants are the single source of truth
# ---------------------------------------------------------------------------


def test_script_declares_license_constants(script_module):
    """The license tag + provenance + scope live at module level, not
    inside the summary writer.

    The previous shape had ``LICENSE = None`` and the script-specific
    ``committed_normalized_sample_reason_skipped`` string baked into
    the summary writer with no canonical
    ``license_redistribution_status`` / ``redistribution_gate_*``
    fields at all. Lifting the canonical tag + provenance + scope to
    module-level constants means a future tag change (e.g. the
    dataset owner adding ``license: mit`` to the card YAML) is a
    one-line edit and the gate handles the status derivation.
    """

    assert script_module.LICENSE_TAG is None, (
        "ssakethch/h200-quantization-benchmarks ships with no `license:` "
        "field in the HF card front-matter; LICENSE_TAG must remain None "
        "until the upstream owner adds one"
    )
    assert script_module.LICENSE_SOURCE == (
        "HF card frontmatter has no `license:` field; recorded as unspecified"
    )
    assert script_module.GATE_SCOPE == "committed_normalized_sample"
    # Pre-existing script-level skip-reason string preserved verbatim —
    # downstream tests in ``tests/test_hf_h200_quantization_ingest.py``
    # pin this exact value.
    assert script_module.COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON == (
        "license_unspecified_no_redistribution_promise"
    )


def test_script_license_back_compat_alias(script_module):
    """The back-compat ``LICENSE`` alias still resolves to ``LICENSE_TAG``.

    A small number of out-of-tree audits read the script's
    module-level ``LICENSE`` attribute. The refactor renames the
    canonical constant to ``LICENSE_TAG`` (mirrors agent-llm-traces)
    but keeps ``LICENSE`` as an alias so those audits don't break.
    """

    assert script_module.LICENSE is script_module.LICENSE_TAG
    assert script_module.LICENSE is None


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
    ), (
        "script must import decide_redistribution from the canonical gate"
    )
    assert "decide_redistribution" in script_source, (
        "script must reference decide_redistribution in its code path"
    )
    assert "OperatorPolicyLedger" in script_source, (
        "script must load the operator policy ledger"
    )


def test_script_does_not_redeclare_permissive_set(script_source: str):
    """Confidence rail: no second copy of the closed permissive allow-list."""

    forbidden = [
        "PERMISSIVE_LICENSE_TAGS = {",
        "PERMISSIVE_LICENSES = {",
        '"permissive_apache_2_0":',
        '"permissive_cdla_2":',
        '"permissive_mit":',
    ]
    hits = [f for f in forbidden if f in script_source]
    assert not hits, (
        f"script carries duplicated permissive allow-list: {hits!r}. "
        f"Delete and call classify_license / decide_redistribution."
    )


def test_script_does_not_hardcode_license_redistribution_status(
    script_source: str,
):
    """The canonical status string ``"unspecified_no_committed_sample"``
    must not appear inline in the script's summary writer — the gate
    produces it; the script consumes it.

    Docstring mentions are allowed (the module docstring explicitly
    documents the verdict). We therefore filter out occurrences that
    sit inside Module / FunctionDef / AsyncFunctionDef docstrings.
    """

    tree = ast.parse(script_source)
    offending: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "unspecified_no_committed_sample":
                offending.append((node.lineno, node.value))
    # Remove docstring occurrences.
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            body = getattr(node, "body", []) or []
            docstring_node = (
                body[0]
                if body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
                else None
            )
            if docstring_node and (
                "unspecified_no_committed_sample"
                in docstring_node.value.value
            ):
                for (ln, v) in list(offending):
                    if ln == docstring_node.lineno and v == (
                        "unspecified_no_committed_sample"
                    ):
                        offending.remove((ln, v))
    assert not offending, (
        f"script hard-codes the 'unspecified_no_committed_sample' status "
        f"string in code at lines {[ln for (ln, _) in offending]!r}; let "
        f"the gate produce it via decide_redistribution()"
    )


# ---------------------------------------------------------------------------
# 3. evaluate_redistribution — pure function returns gate verdict
# ---------------------------------------------------------------------------


def test_evaluate_redistribution_returns_gate_decision_type(
    script_module, policy_module,
):
    """``evaluate_redistribution`` is exposed so tests can drive the
    gate path without invoking the CSV download / normalisation
    pipeline. It must return the gate's ``RedistributionGateDecision``
    dataclass.

    We compare by class name + attribute set instead of ``isinstance``,
    because the script imports the gate via the canonical
    ``aurelius.ingestion.redistribution_gate`` path while the test
    fixture loads the gate module under a different name — two module
    objects → two class objects → ``isinstance`` is False even though
    the instance is the genuine gate output.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert type(decision).__name__ == "RedistributionGateDecision", (
        f"evaluate_redistribution must return a RedistributionGateDecision, "
        f"got {type(decision).__name__!r}"
    )
    for field in (
        "permitted",
        "reason_code",
        "reason_detail",
        "license_status",
        "license_observed",
        "scope",
        "operator_grant_dataset_id",
    ):
        assert hasattr(decision, field), (
            f"gate decision missing field {field!r}; the wiring may have "
            f"diverged from the canonical decision schema"
        )


def test_evaluate_redistribution_default_tag_under_empty_ledger_denies(
    script_module, policy_module,
):
    """Default tag (``None`` — no upstream license) + empty ledger →
    deny with the canonical unspecified status code. This is the
    verdict pinned by
    ``test_no_committed_normalized_sample_under_unspecified_license``
    in ``tests/test_hf_h200_quantization_ingest.py`` and the verdict
    already embedded in the committed summary.json (no normalised
    sample committed).
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(ledger=ledger)
    assert decision.permitted is False
    assert decision.license_status == "unspecified_no_committed_sample"
    assert decision.reason_code == "no_grant_recorded"
    assert decision.operator_grant_dataset_id is None
    assert decision.scope == "committed_normalized_sample"


def test_evaluate_redistribution_permissive_tag_under_empty_ledger_permits(
    script_module, policy_module,
):
    """Swap the tag to an explicit permissive value (``"mit"``) under
    the default-empty ledger → the gate permits with
    ``permitted_declared_permissive_license`` and status
    ``permissive_mit``. This proves the wiring actually consults the
    gate; if the script were carrying a hard-coded ``deny`` for this
    dataset (the v1 behaviour pre-wiring), this test would fail.
    """

    ledger = policy_module.OperatorPolicyLedger.empty()
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag="mit",
    )
    assert decision.permitted is True
    assert decision.license_status == "permissive_mit"
    assert decision.reason_code == "permitted_declared_permissive_license"
    assert decision.operator_grant_dataset_id is None


def test_evaluate_redistribution_operator_grant_for_none_license_permits(
    script_module, policy_module, gate_module,
):
    """In-memory operator grant for ``license = None`` flips the gate
    to PERMIT with ``permitted_operator_grant``. The grant is never
    written to disk — this only verifies that the script's
    ``evaluate_redistribution`` consults the ledger.
    """

    grant = policy_module.OperatorGrant(
        dataset_id=DATASET_ID,
        granted=True,
        granted_by="test-operator-in-memory",
        granted_at_iso="2026-06-02T00:00:00Z",
        allowed_scopes=("committed_normalized_sample",),
        notes=(
            "in-memory test grant — never written to disk; verifies "
            "the fourth gate consumer actually reads the ledger"
        ),
    )
    ledger = policy_module.OperatorPolicyLedger(
        doc_version=policy_module.POLICY_DOC_VERSION,
        policy_default="deny_all",
        grants=(grant,),
    )
    decision = script_module.evaluate_redistribution(
        ledger=ledger,
        license_tag=None,
    )
    assert decision.permitted is True, (
        "wiring is broken: an operator grant for license=None must flip "
        "the gate's verdict to permit. The script's evaluate_redistribution "
        "is not consulting the ledger."
    )
    assert decision.reason_code == (
        gate_module.REASON_PERMITTED_OPERATOR_GRANT
    )
    assert decision.operator_grant_dataset_id == DATASET_ID


# ---------------------------------------------------------------------------
# 4. Committed per-dataset summary.json carries the gate fields
# ---------------------------------------------------------------------------


def test_committed_summary_carries_redistribution_gate_metadata():
    s = json.loads(_summary_path().read_text())
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
        f"committed summary.json missing gate-derived fields: "
        f"{sorted(missing)!r}"
    )
    assert s["redistribution_gate_reason_code"] == "no_grant_recorded"
    assert s["redistribution_gate_permitted"] is False
    assert s["redistribution_gate_operator_grant_dataset_id"] is None
    assert s["redistribution_gate_scope"] == "committed_normalized_sample"
    assert s["license_redistribution_status"] == (
        "unspecified_no_committed_sample"
    )
    # The detail string is a free-form audit trail; assert only its
    # non-emptiness and that it mentions the dataset id + the
    # canonical reason wording so the field cannot be silently zeroed.
    detail = s["redistribution_gate_reason_detail"]
    assert isinstance(detail, str) and detail
    assert DATASET_ID in detail
    assert "no operator grant recorded" in detail


def test_committed_summary_status_matches_gate_classification(
    gate_module,
):
    """The status label in summary.json equals what the gate classifies
    the recorded ``license`` tag (``None``) into. Pinning this gives
    zero behavioural drift on the already-committed artefacts.
    """

    s = json.loads(_summary_path().read_text())
    expected_status = gate_module.classify_license(s["license"])
    assert s["license_redistribution_status"] == expected_status, (
        f"summary status {s['license_redistribution_status']!r} != "
        f"gate classification {expected_status!r} of license tag "
        f"{s['license']!r}"
    )


def test_committed_summary_license_redistribution_source_recorded():
    """The human-curated provenance string lives in the committed
    summary so a future maintainer can audit *why* the license tag is
    recorded as ``None``.
    """

    s = json.loads(_summary_path().read_text())
    assert s["license_redistribution_source"] == (
        "HF card frontmatter has no `license:` field; recorded as unspecified"
    )


def test_committed_summary_preserves_v1_skip_reason_verbatim():
    """The pre-wiring script-level skip reason string is preserved
    verbatim — downstream tests in
    ``tests/test_hf_h200_quantization_ingest.py`` pin this exact
    value. The gate fields are additive.
    """

    s = json.loads(_summary_path().read_text())
    assert s["committed_normalized_sample_reason_skipped"] == (
        "license_unspecified_no_redistribution_promise"
    )
    # And no normalised sample committed.
    assert s["committed_normalized_sample_rows"] == 0
    assert s["committed_normalized_sample_bytes"] == 0
    assert s["committed_normalized_sample_path"] is None


# ---------------------------------------------------------------------------
# 5. Round-6 audit summary carries the new gate-derived fields + v2 doc
# ---------------------------------------------------------------------------


def test_audit_summary_doc_version_bumped_to_v2():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["doc_version"] == (
        "round6_broadened_discovery_audit_summary_v2"
    ), (
        f"audit summary doc_version is {a['doc_version']!r}; bump to v2 "
        f"when the gate wiring lands so consumers can detect the new "
        f"redistribution_gate_* fields"
    )


def test_audit_summary_records_redistribution_gate_metadata():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["redistribution_gate_scope"] == "committed_normalized_sample"
    assert a["redistribution_gate_policy_default"] == "deny_all"
    assert a["redistribution_gate_policy_grant_count"] == 0


def test_audit_summary_per_dataset_has_gate_reason_code():
    a = json.loads(AUDIT_PATH.read_text())
    assert a["ingested"], "audit summary has no ingested entries"
    for entry in a["ingested"]:
        for key in (
            "redistribution_gate_reason_code",
            "redistribution_gate_permitted",
            "redistribution_gate_operator_grant_dataset_id",
            "license_redistribution_status",
        ):
            assert key in entry, (
                f"audit entry {entry.get('dataset_id')!r} missing {key!r}"
            )
        assert entry["redistribution_gate_reason_code"] == (
            "no_grant_recorded"
        )
        assert entry["redistribution_gate_permitted"] is False
        assert entry["license_redistribution_status"] == (
            "unspecified_no_committed_sample"
        )
        assert entry["redistribution_gate_operator_grant_dataset_id"] is None


# ---------------------------------------------------------------------------
# 6. Safety — no HF_TOKEN literal in the refactored script
# ---------------------------------------------------------------------------


def test_no_hf_token_literal_in_script(script_source: str):
    """No ``hf_`` token literal, no ``HF_TOKEN`` value embedded.

    Real HF tokens are mixed-case base62 strings of the form
    ``hf_<lowercase + UPPERCASE + digits, length 30+>``. We require
    both a lowercase and an uppercase character in the suffix to
    distinguish actual tokens from the script's logger name
    (``aurelius.hf_h200_quantization_ingest``), which is lowercase
    + digits + underscores only.
    """

    candidates = re.findall(r"\bhf_[A-Za-z0-9]{20,}\b", script_source)
    suspicious = [
        c for c in candidates
        if any(ch.isupper() for ch in c[3:])
        and any(ch.islower() for ch in c[3:])
    ]
    assert not suspicious, (
        f"script contains a literal that looks like an HF token: "
        f"{suspicious!r}"
    )
    bad_assignment = re.search(
        r'HF_TOKEN\s*=\s*["\']hf_', script_source,
    )
    assert bad_assignment is None, (
        "HF_TOKEN appears to be assigned a literal hf_ value"
    )


# ---------------------------------------------------------------------------
# 7. ingest / _write_round6_audit_summary accept ledger keyword arg
# ---------------------------------------------------------------------------


def test_ingest_accepts_ledger_keyword_arg(script_module):
    """The refactored ``ingest`` signature must accept ``ledger`` as a
    keyword-only argument so callers (including main()) can supply a
    pre-loaded ledger without re-reading the policy file.

    The default value must be ``None`` so existing call sites that
    don't pass the ledger still work — the function falls back to
    ``_load_ledger()`` internally.
    """

    import inspect

    sig = inspect.signature(script_module.ingest)
    params = sig.parameters
    assert "ledger" in params, (
        "ingest must accept a 'ledger' parameter for the gate wiring"
    )
    p = params["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"ledger must be keyword-only (got {p.kind!r}) to keep the "
        f"existing positional call signature stable"
    )
    assert p.default is None, (
        "ledger default must be None so existing call sites without the "
        "wiring fall back to _load_ledger()"
    )


def test_write_round6_audit_summary_accepts_ledger_keyword_arg(
    script_module,
):
    """``_write_round6_audit_summary`` must accept ``ledger`` so the
    round-6 ``main()`` can load the ledger once and pass it both to
    ``ingest`` and to the audit summary writer (single source of
    truth for ``redistribution_gate_policy_default`` /
    ``redistribution_gate_policy_grant_count``).
    """

    import inspect

    sig = inspect.signature(script_module._write_round6_audit_summary)
    params = sig.parameters
    assert "ledger" in params, (
        "_write_round6_audit_summary must accept a 'ledger' parameter"
    )
    p = params["ledger"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"ledger must be keyword-only (got {p.kind!r})"
    )
    assert p.default is None, "ledger default must be None"


def test_load_ledger_returns_empty_when_policy_path_missing(
    script_module, policy_module, tmp_path,
):
    """``_load_ledger`` must fall back to ``OperatorPolicyLedger.empty()``
    when the policy file is absent — that's the documented
    self-sufficiency rail (a fresh checkout without the committed JSON
    pulled still produces correct deny decisions instead of crashing).
    """

    nonexistent = tmp_path / "no_such_file.json"
    assert not nonexistent.exists()
    ledger = script_module._load_ledger(nonexistent)
    assert ledger.policy_default == "deny_all"
    assert ledger.grants == ()
