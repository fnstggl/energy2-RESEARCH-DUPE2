# Weather Integration (Roadmap Phase 3)

## Provider: Open-Meteo

No API key required. Free for non-commercial use, no rate limits.
For commercial deployment, contact Open-Meteo for a license.

- Forecast API: https://open-meteo.com/en/docs
- Historical API: https://open-meteo.com/en/docs/historical-weather-api

## Purpose

Weather data feeds the Phase 3 cooling/PUE optimizer:
- Predict facility cooling load from temperature and humidity
- Estimate PUE (Power Usage Effectiveness) penalty in hot weather
- Detect heat-wave risk for ERCOT (Texas grid especially vulnerable)
- Forecast future grid stress from weather patterns

Weather is NOT a substitute for GPU telemetry (DCGM).
DCGM tells current GPU state. Weather predicts future cooling/grid conditions.

## Variables to request

| Variable               | Unit  | Use                              |
|------------------------|-------|----------------------------------|
| temperature_2m         | °C    | Air temp at 2m — cooling load    |
| relativehumidity_2m    | %     | Humidity — cooling efficiency    |
| windspeed_10m          | km/h  | Wind cooling effect              |
| apparent_temperature   | °C    | Feels-like — cooling load proxy  |
| precipitation          | mm    | Extreme weather signal           |

## DC locations to monitor

| Region   | Location       | Latitude | Longitude |
|----------|----------------|----------|-----------|
| us-west  | Sacramento, CA |  38.55   | -121.47   |
| us-east  | Ashburn, VA    |  39.04   |  -77.49   |
| us-south | Dallas, TX     |  32.78   |  -96.80   |

## Planned implementation

`aurelius/ingestion/weather.py` — not yet implemented.
