"""ElectricityState — first-class electricity economics for the canonical world model.

Turns electricity from a passive constant cost label into an active, per-period economic signal the MPC can
see and respond to. Built on the real day-ahead price series (`price_series`: PJM/ERCOT/CAISO) and the
canonical region→market registry (`aurelius.ingestion.region_registry`).

Honesty (see research/ELECTRICITY_PRODUCTION_REALISM_AUDIT.md):
  * historical market prices are TRACE_DERIVED;
  * the per-period forecast is FORECAST_DERIVED (the existing ForecastingModel already forecasts
    `electricity_price` as a target — a real diurnal price in the frames makes that forecast vary);
  * the flat fallback price is SIMULATOR_INFERENCE.
Real prices are aligned to the trace by HOUR-OF-DAY (the diurnal profile), because the Azure serving trace has
no wall-clock timestamp to calendar-align against — documented, not silently calendar-faked. A market with no
wired data raises rather than fabricating (price_series already enforces this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .price_series import diurnal_profile, load_price_series, price_percentiles

# canonical region → ISO market key (the price_series markets). Mirrors aurelius/ingestion/region_registry.py.
MARKET_BY_REGION = {"us-east": "pjm", "us-south": "ercot", "us-west": "caiso"}
REGION_BY_MARKET = {v: k for k, v in MARKET_BY_REGION.items()}


@dataclass
class ElectricityState:
    """Persistent per-period electricity view (clones with CanonicalWorldState via deepcopy)."""
    market: str = "flat"                 # "pjm" | "ercot" | "caiso" | "flat" (constant fallback)
    region: str = "unknown"
    current_price: float = 0.06          # $/kWh for the current period (realized)
    forecast_price: float = 0.06         # $/kWh forecast for the current period (FORECAST_DERIVED)
    price_percentile: float = 0.5        # where current_price sits in the market's own distribution [0,1]
    volatility: float = 0.0              # coefficient of variation of the diurnal profile
    spike: bool = False                  # current_price ≥ market p90 (an expensive period)
    forecast_error: float = 0.0          # |forecast − realized| / realized for the current period
    provenance: str = "SIMULATOR_INFERENCE"   # TRACE_DERIVED for real markets, else SIMULATOR_INFERENCE

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class PowerState:
    """Persistent power/energy ledger (clones with CanonicalWorldState). Formalises the per-period power/energy
    the world simulator already computes (`PeriodOutcome.power_w` / `energy_j` from the DVFS roofline action):

        clock_state → compute throughput factor → phase latency → watts → joules → kWh → × price → $.

    The DVFS power curve (power_w = TDP·(0.4 + 0.6·clock^2.4)) is SIMULATOR_INFERENCE; cumulative energy is an
    accumulation of the real per-period outcomes. Distinguishes the modelled lever (clock-locking, which shapes
    memory-bound decode energy) from power-capping (NOT modelled — see ELECTRICITY_PRODUCTION_REALISM_AUDIT.md:
    a cap below the decode draw never engages, so it would book phantom savings).
    """
    mean_power_w: float = 0.0            # last period's mean GPU power under the clock action
    clock_state: str = "base"            # last applied clock policy (base/low/high)
    cumulative_energy_kwh: float = 0.0   # serving energy accumulated across periods
    cumulative_energy_cost: float = 0.0  # $ accumulated (energy_kwh × price)
    lever: str = "clock_locking"         # the modelled DVFS lever (NOT power_cap — see audit)

    def accumulate(self, *, power_w: float, energy_j: float, price_per_kwh: float, clock_state: str) -> None:
        kwh = max(0.0, energy_j) / 3.6e6
        self.mean_power_w = round(power_w, 1)
        self.clock_state = clock_state
        self.cumulative_energy_kwh = round(self.cumulative_energy_kwh + kwh, 6)
        self.cumulative_energy_cost = round(self.cumulative_energy_cost + kwh * price_per_kwh, 6)

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class PriceProfile:
    """A real (or flat) diurnal price path + its distribution stats, aligned to the trace's cycle position."""
    market: str
    region: str
    by_cycle: dict                       # cycle_pos (0..cycle_len-1) → price_per_kwh ($/kWh)
    percentiles: dict = field(default_factory=dict)   # p05..p95 of the FULL market series ($/kWh)
    volatility: float = 0.0
    provenance: str = "SIMULATOR_INFERENCE"

    def price_at(self, period_index: int, cycle_len: int) -> float:
        if not self.by_cycle:
            return 0.06
        return self.by_cycle.get(period_index % cycle_len, next(iter(self.by_cycle.values())))

    def percentile_of(self, price: float) -> float:
        """Fraction of the market distribution at or below ``price`` (from the percentile thresholds)."""
        if not self.percentiles:
            return 0.5
        qs = sorted((int(k[1:]) / 100.0, v) for k, v in self.percentiles.items() if k.startswith("p"))
        below = [q for q, v in qs if v <= price]
        return max(below) if below else 0.0

    def is_spike(self, price: float) -> bool:
        return bool(self.percentiles) and price >= self.percentiles.get("p90", float("inf"))


def _scale_to_cycle(hourly: dict, cycle_len: int) -> dict:
    """Map a 24-hour diurnal profile onto ``cycle_len`` control steps (hourly trace → cycle_len 24 = identity)."""
    if cycle_len == 24:
        return dict(hourly)
    return {c: hourly.get(round(c * 24 / max(1, cycle_len)) % 24, 0.06) for c in range(cycle_len)}


def build_price_profile(market: str | None, cycle_len: int, *, flat_price: float = 0.06) -> PriceProfile:
    """Build the diurnal price path for ``market`` (real, TRACE_DERIVED) or a flat fallback.

    ``market=None`` (or "flat") reproduces the constant-price behaviour EXACTLY (every cycle position =
    ``flat_price``), so the flat-price baseline is unchanged. A real market name loads the committed
    day-ahead series, takes its hour-of-day mean profile, and exposes the market distribution percentiles.
    """
    if not market or market == "flat":
        return PriceProfile("flat", "unknown", {c: flat_price for c in range(cycle_len)},
                            percentiles={}, volatility=0.0, provenance="SIMULATOR_INFERENCE")
    series = load_price_series(market)                  # raises if the market is not wired — never fabricates
    hourly = diurnal_profile(series)                    # hour-of-day mean ($/kWh), TRACE_DERIVED
    by_cycle = _scale_to_cycle(hourly, cycle_len)
    pct = price_percentiles(series)
    vals = list(by_cycle.values())
    mean = sum(vals) / len(vals) if vals else 0.0
    vol = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / mean if mean > 0 else 0.0
    return PriceProfile(market, REGION_BY_MARKET.get(market, "unknown"), by_cycle, percentiles=pct,
                        volatility=round(vol, 4), provenance="TRACE_DERIVED")


def electricity_state_for_period(profile: PriceProfile, period_index: int, cycle_len: int, *,
                                 forecast_price: float | None = None) -> ElectricityState:
    """Build the ElectricityState for one period from the price profile (realized) and an optional forecast."""
    price = profile.price_at(period_index, cycle_len)
    fc = forecast_price if forecast_price is not None else price
    return ElectricityState(
        market=profile.market, region=profile.region, current_price=round(price, 6),
        forecast_price=round(fc, 6), price_percentile=round(profile.percentile_of(price), 3),
        volatility=profile.volatility, spike=profile.is_spike(price),
        forecast_error=round(abs(fc - price) / price, 4) if price > 0 else 0.0,
        provenance=profile.provenance)
