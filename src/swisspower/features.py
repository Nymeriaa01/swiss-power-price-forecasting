"""Feature engineering.

Everything is built on a "daily matrix" (one row per delivery day, one column
per local hour 0..23), which is the usual layout in electricity price
forecasting and makes lags in days trivial and leak-free.

Information set: the forecast for delivery day D is issued on D-1 around
10:00 (local), before the 12:00 day-ahead auction. At that time we know
- all day-ahead prices up to and including day D-1 (they were published on D-2),
- the weather forecast for day D,
- the gas settlement of D-2 at the latest (D-1 settles in the evening).
`tests/test_no_lookahead.py` checks this property.
"""
import holidays
import numpy as np
import pandas as pd

from . import config

HOURS = list(range(24))
PRICE_LAGS = [1, 2, 3, 7]


def to_daily_matrix(s):
    """UTC hourly series -> DataFrame(index=local date, columns=local hour 0..23).

    DST: the missing hour of the spring day is interpolated, the doubled hour
    of the autumn day is averaged.
    """
    local = s.tz_convert(config.TZ)
    df = pd.DataFrame({
        "date": local.index.tz_localize(None).normalize(),
        "hour": local.index.hour,
        "v": local.to_numpy(),
    })
    m = df.pivot_table(index="date", columns="hour", values="v", aggfunc="mean", dropna=False)
    m = m.reindex(columns=HOURS)
    m = m.interpolate(axis=1, limit=1, limit_area="inside")
    m.index.name = "date"
    m.columns.name = None
    return m


def weather_aggregates(w):
    """Average the weather points into four drivers (UTC hourly)."""
    def mean_of(prefix, var):
        cols = [c for c in w.columns if c.startswith(prefix) and c.endswith(var)]
        return w[cols].mean(axis=1)

    return pd.DataFrame({
        "ch_temp": mean_of("ch_", "temperature_2m"),
        "ch_solar": mean_of("ch_", "shortwave_radiation"),
        "de_wind": mean_of("de_", "wind_speed_100m"),
        "de_solar": mean_of("de_", "shortwave_radiation"),
    })


def gas_for_delivery_day(ttf, days):
    """For each delivery day D, the last TTF close available on D-1 morning (= close of D-2 or before)."""
    cal = ttf.reindex(pd.date_range(ttf.index.min(), days.max())).ffill()
    return cal.reindex(days - pd.Timedelta(days=2)).set_axis(days)


def build_matrices(raw):
    """Raw downloads -> dict of daily matrices sharing the same date index."""
    ch = to_daily_matrix(raw["ch"])
    days = ch.index
    mats = {"ch": ch, "de": to_daily_matrix(raw["de"]).reindex(days)}
    for name, s in weather_aggregates(raw["weather"]).items():
        mats[name] = to_daily_matrix(s).reindex(days)
    mats["ttf"] = gas_for_delivery_day(raw["ttf"], days)
    return mats


def calendar(days):
    ch_holidays = holidays.Switzerland(years=range(days.min().year, days.max().year + 1))
    return pd.DataFrame({
        "dow": days.dayofweek,
        "month": days.month,
        "holiday": [int(d in ch_holidays) for d in days.date],
    }, index=days)


def wide_features(m):
    """One row per delivery day. Used by the LEAR model (24 hour-specific regressions)."""
    days = m["ch"].index
    blocks = []
    for lag in PRICE_LAGS:
        blocks.append(m["ch"].shift(lag).add_prefix(f"ch_d{lag}_h"))
    blocks.append(m["de"].shift(1).add_prefix("de_d1_h"))
    for w in ["ch_temp", "ch_solar", "de_wind", "de_solar"]:
        blocks.append(m[w].add_prefix(f"{w}_h"))  # forecast for day D itself
    blocks.append(m["ttf"].rename("ttf").to_frame())
    cal = calendar(days)
    blocks.append(pd.get_dummies(cal["dow"], prefix="dow", dtype=float).reindex(columns=[f"dow_{i}" for i in range(7)], fill_value=0.0))
    blocks.append(cal[["holiday"]].astype(float))
    return pd.concat(blocks, axis=1)


def long_features(m):
    """One row per (delivery day, hour). Used by the gradient-boosting model."""
    days = m["ch"].index
    ch, de = m["ch"], m["de"]
    d1 = ch.shift(1)

    daily = pd.DataFrame({
        "ch_d1_mean": d1.mean(axis=1),
        "ch_d1_max": d1.max(axis=1),
        "ch_d1_min": d1.min(axis=1),
        "ch_d1_std": d1.std(axis=1),
        "de_d1_mean": de.shift(1).mean(axis=1),
        "ch_de_spread_d1": d1.mean(axis=1) - de.shift(1).mean(axis=1),
        "ch_d7_mean": ch.shift(7).mean(axis=1),
        "ttf": m["ttf"],
        "de_wind_mean": m["de_wind"].mean(axis=1),
        "de_solar_mean": m["de_solar"].mean(axis=1),
        "ch_temp_mean": m["ch_temp"].mean(axis=1),
    }, index=days).join(calendar(days))

    hourly = {
        "ch_d1": d1, "ch_d2": ch.shift(2), "ch_d7": ch.shift(7),
        "de_d1": de.shift(1),
        "ch_temp": m["ch_temp"], "ch_solar": m["ch_solar"],
        "de_wind": m["de_wind"], "de_solar": m["de_solar"],
    }
    rows = []
    for h in HOURS:
        df = pd.DataFrame({k: v[h] for k, v in hourly.items()}, index=days)
        df["hour"] = h
        rows.append(df.join(daily))
    X = pd.concat(rows).set_index("hour", append=True).sort_index()
    return X


def target_long(m):
    return m["ch"].stack(future_stack=True).rename("price").rename_axis(["date", "hour"])


def naive_forecast(ch):
    """Standard EPF naive benchmark: D-7 for Monday/Saturday/Sunday, D-1 otherwise."""
    f = ch.shift(1).copy()
    weekly = ch.index.dayofweek.isin([0, 5, 6])
    f.loc[weekly] = ch.shift(7).loc[weekly]
    return f


def complete_days(m, days):
    """Days where the target and the main inputs are all present."""
    ok = m["ch"].loc[days].notna().all(axis=1)
    ok &= m["ch"].shift(7).loc[days].notna().all(axis=1)
    return days[ok.to_numpy()]
