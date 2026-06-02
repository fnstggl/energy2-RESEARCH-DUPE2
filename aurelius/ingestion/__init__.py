"""Data ingestion modules for Aurelius.

Top-level re-exports are lazy so that lightweight submodules (e.g. the
operator redistribution policy) can be imported in environments where
the heavy data-science dependencies (pandas, sqlalchemy, ...) are not
installed. Direct ``from aurelius.ingestion.<submodule> import ...`` is
the preferred call pattern; the lazy ``__getattr__`` below is kept for
backwards compatibility with the older ``from aurelius.ingestion import
EnergyPriceIngester`` style.
"""

from typing import Any

__all__ = ["EnergyPriceIngester", "JobLogIngester"]


def __getattr__(name: str) -> Any:
    if name == "EnergyPriceIngester":
        from .energy_prices import EnergyPriceIngester
        return EnergyPriceIngester
    if name == "JobLogIngester":
        from .job_logs import JobLogIngester
        return JobLogIngester
    raise AttributeError(f"module 'aurelius.ingestion' has no attribute {name!r}")
