"""Action registry — the single source of truth for what the planner may optimize.

Wraps the canonical ``ACTION_SPECS`` (``actions.py``) with query + enumeration helpers. The
controller asks the registry for candidate :class:`ActionBundle`s; by default the registry
enumerates **only CONNECTED surfaces** (the ones that change the scored reward), so a planner
*cannot* optimize a fake or not-yet-wired knob. SIMULATED_ONLY surfaces are enumerated only
when explicitly opted in; PLANNED / REQUIRES_PILOT_TELEMETRY / REJECTED surfaces are never
enumerated and are rejected if a bundle tries to set them away from their no-op default.
"""

from __future__ import annotations

from itertools import product

from .actions import (
    ACTION_SPECS,
    CONNECTED,
    CONNECTED_SURFACES,
    PLANNED,
    REQUIRES_PILOT_TELEMETRY,
    SIMULATED_ONLY,
    SIMULATED_SURFACES,
    ActionBundle,
    ActionSpec,
)


def list_all_actions() -> list:
    """Every action surface Aurelius represents (all statuses)."""
    return list(ACTION_SPECS.values())


def list_connected_actions() -> list:
    """Surfaces that change the scored reward today (optimized by default)."""
    return [s for s in ACTION_SPECS.values() if s.status == CONNECTED]


def list_simulated_actions() -> list:
    """Surfaces with a real model but not yet in the reward path (opt-in)."""
    return [s for s in ACTION_SPECS.values() if s.status == SIMULATED_ONLY]


def list_planned_actions() -> list:
    """Surfaces desired but not simulatable today (never optimized)."""
    return [s for s in ACTION_SPECS.values()
            if s.status in (PLANNED, REQUIRES_PILOT_TELEMETRY)]


def optimizable_surfaces(*, include_simulated: bool = False) -> tuple:
    """The surface names the planner is allowed to vary."""
    return CONNECTED_SURFACES + (SIMULATED_SURFACES if include_simulated else ())


def validate_action_bundle(bundle: ActionBundle) -> dict:
    """Return ``{ok, problems}``. A bundle is invalid if any field holds a value outside its
    spec, or if a non-CONNECTED/non-SIMULATED surface is set away from its no-op default
    (you cannot *use* a PLANNED/REJECTED action — that would be a fake claim)."""
    problems = []
    for name, spec in ACTION_SPECS.items():
        value = getattr(bundle, name)
        if not spec.validate(value):
            problems.append(f"{name}={value!r} not in allowed options {spec.options}")
        elif value != spec.default and spec.status not in (CONNECTED, SIMULATED_ONLY):
            problems.append(
                f"{name} is {spec.status} (not actuatable); cannot be set to {value!r} "
                f"— {spec.limitation}")
    return {"ok": not problems, "problems": problems}


def enumerate_candidate_bundles(*, connected_only: bool = True) -> list:
    """Cartesian product over the optimizable surfaces' options; all other surfaces stay at
    their no-op default. ``connected_only=True`` (default) → only CONNECTED surfaces vary
    (the 12 capacity×ordering×admission bundles); ``False`` also varies SIMULATED_ONLY
    surfaces (explicit opt-in). PLANNED surfaces are never enumerated."""
    names = list(optimizable_surfaces(include_simulated=not connected_only))
    option_lists = [ACTION_SPECS[n].options for n in names]
    return [ActionBundle(**dict(zip(names, combo))) for combo in product(*option_lists)]


def spec_for(name: str) -> ActionSpec:
    return ACTION_SPECS[name]


def status_counts() -> dict:
    counts: dict = {}
    for s in ACTION_SPECS.values():
        counts[s.status] = counts.get(s.status, 0) + 1
    return counts


def planned_report() -> list:
    """The surfaces a controller should report as *understood but not yet available*."""
    return [{"surface": s.surface, "field": s.name, "status": s.status,
             "roadmap": s.roadmap, "limitation": s.limitation}
            for s in ACTION_SPECS.values()
            if s.status in (SIMULATED_ONLY, PLANNED, REQUIRES_PILOT_TELEMETRY)]


__all__ = [
    "list_all_actions", "list_connected_actions", "list_simulated_actions",
    "list_planned_actions", "optimizable_surfaces", "validate_action_bundle",
    "enumerate_candidate_bundles", "spec_for", "status_counts", "planned_report",
]
