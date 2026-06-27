"""Full-source ingestion for the canonical environment.

Each adapter ingests the complete available public source, computes train/holdout
splits + calibration distributions, and reports its data tier honestly
(FULL_TRACE / SUBSET_TRACE / SAMPLE_FIXTURE / MOCK / BLOCKED). Nothing is silently
downgraded; a blocked source records the exact manual unblock step.
"""

from . import azure, electricity, mooncake, v2026
from .azure import hourly_arrival_frames, ingest_azure, to_serving_raw
from .electricity import load_prices
from .mooncake import ingest_mooncake, reuse_distribution, split_reuse
from .v2026 import v2026_status


def source_statuses() -> dict:
    """One-call data-tier report for every canonical source (for the manifest/PR)."""
    _, az = ingest_azure(limit=1)
    _, mc = ingest_mooncake()
    _, el = load_prices("CAISO")
    return {
        "azure_llm": az.to_dict(),
        "mooncake": mc.to_dict(),
        "alibaba_gpu_v2026": v2026_status().to_dict(),
        "electricity": el.to_dict(),
    }


__all__ = [
    "azure", "mooncake", "v2026", "electricity",
    "ingest_azure", "hourly_arrival_frames", "to_serving_raw", "ingest_mooncake",
    "reuse_distribution", "split_reuse", "load_prices", "v2026_status", "source_statuses",
]
