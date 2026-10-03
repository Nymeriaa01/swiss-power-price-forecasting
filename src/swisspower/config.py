"""Project-wide settings. Everything that changes a result lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"

TZ = "Europe/Zurich"

# Full history downloaded. 2022 is kept on purpose: it is the energy-crisis
# regime and the models have to cope with it in their calibration window.
START = "2022-01-01"
END = "2026-09-30"

# Out-of-sample period. Nothing after TEST_START is ever used for fitting
# before the day it is forecast.
TEST_START = "2024-01-01"

# Walk-forward setup
CALIBRATION_DAYS = 730   # rolling window (2 years)
REFIT_EVERY_DAYS = 7     # models are re-estimated once a week

# Weather points (lat, lon). German wind and solar drive a large part of the
# Swiss price through market coupling, Swiss temperature drives demand.
WEATHER_POINTS = {
    "ch_zurich": (47.37, 8.54),
    "ch_geneva": (46.20, 6.14),
    "de_north_hamburg": (53.55, 9.99),
    "de_north_coast": (54.30, 8.60),
    "de_east_brandenburg": (52.40, 13.00),
    "de_south_munich": (48.14, 11.58),
    "de_south_stuttgart": (48.78, 9.18),
    "de_west_frankfurt": (50.11, 8.68),
}
WEATHER_VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]

# Storage asset used in the trading backtest (a small battery / pumped-storage proxy)
STORAGE = {
    "power_mw": 1.0,
    "energy_mwh": 2.0,
    "round_trip_eff": 0.85,
    "cost_eur_per_mwh": 5.0,  # wear / variable cost per MWh discharged
}

SEED = 42
