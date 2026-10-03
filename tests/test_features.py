import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swisspower import config, features  # noqa: E402


def synthetic_raw(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    hours = idx.hour.to_numpy()
    base = 80 + 20 * np.sin(2 * np.pi * hours / 24)
    w_cols = [f"{p}__{v}" for p in config.WEATHER_POINTS for v in config.WEATHER_VARS]
    return {
        "ch": pd.Series(base + rng.normal(0, 5, len(idx)), index=idx),
        "de": pd.Series(base + rng.normal(0, 8, len(idx)), index=idx),
        "weather": pd.DataFrame(rng.normal(10, 3, (len(idx), len(w_cols))), index=idx, columns=w_cols),
        "ttf": pd.Series(35 + rng.normal(0, 1, 366),
                         index=pd.date_range("2024-01-01", "2024-12-31", freq="D")),
    }


def corrupt_future(raw, day):
    """Overwrite everything that is NOT known on day-1 at 10:00 for delivery day `day`."""
    raw = {k: v.copy() for k, v in raw.items()}
    local_start = pd.Timestamp(day).tz_localize(config.TZ).tz_convert("UTC")
    for k in ["ch", "de"]:
        raw[k][raw[k].index >= local_start] = 9999.0                   # prices of D and after
    raw["ttf"][raw["ttf"].index >= pd.Timestamp(day) - pd.Timedelta(days=1)] = 9999.0  # gas settles D-1 evening
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(config.TZ).tz_convert("UTC")
    raw["weather"][raw["weather"].index >= next_day] = 9999.0       # weather forecast for D is allowed
    return raw


@pytest.mark.parametrize("day", ["2024-03-15", "2024-10-28", "2024-12-02"])
def test_no_lookahead(day):
    raw = synthetic_raw()
    m_clean = features.build_matrices(raw)
    m_dirty = features.build_matrices(corrupt_future(raw, day))
    d = pd.Timestamp(day)

    pd.testing.assert_series_equal(features.wide_features(m_clean).loc[d],
                                   features.wide_features(m_dirty).loc[d])
    pd.testing.assert_frame_equal(features.long_features(m_clean).xs(d, level="date"),
                                  features.long_features(m_dirty).xs(d, level="date"))
    pd.testing.assert_series_equal(features.naive_forecast(m_clean["ch"]).loc[d],
                                   features.naive_forecast(m_dirty["ch"]).loc[d])


def test_daily_matrix_handles_dst():
    raw = synthetic_raw()
    m = features.to_daily_matrix(raw["ch"])
    for dst_day in ["2024-03-31", "2024-10-27"]:  # 23-hour and 25-hour days
        row = m.loc[dst_day]
        assert row.shape == (24,)
        assert row.notna().all()
