"""Download and cache the raw data.

Sources (all public, no API key needed):
- Day-ahead prices: Energy-Charts API (Fraunhofer ISE), bidding zones CH and DE-LU
- Weather forecasts: Open-Meteo historical forecast API (archived NWP forecasts,
  not reanalysis, so closer to what a trader actually had)
- Gas: TTF front-month futures close from Yahoo Finance
"""
import time

import pandas as pd
import requests

from . import config

ENERGY_CHARTS = "https://api.energy-charts.info/price"
OPEN_METEO = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _get_json(url, params, retries=6):
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:  # rate limited, back off
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Too many retries for {url} {params}")


def _year_chunks(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for year in range(start.year, end.year + 1):
        a = max(start, pd.Timestamp(year, 1, 1))
        b = min(end, pd.Timestamp(year, 12, 31))
        yield a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")


def download_prices(zone, start=config.START, end=config.END, force=False):
    """Hourly day-ahead price for a bidding zone, UTC index, EUR/MWh.

    DE-LU moved to 15-minute products in late 2025, so everything is averaged
    to hourly values to keep a single resolution.
    """
    path = config.DATA_RAW / f"prices_{zone}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)["price"]

    parts = []
    for a, b in _year_chunks(start, end):
        js = _get_json(ENERGY_CHARTS, {"bzn": zone, "start": a, "end": b})
        s = pd.Series(js["price"], index=pd.to_datetime(js["unix_seconds"], unit="s", utc=True), dtype="float64")
        parts.append(s)
        time.sleep(5)
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    s = s.resample("1h").mean()

    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame("price").to_parquet(path)
    return s


def download_weather(start=config.START, end=config.END, force=False):
    """Hourly weather forecasts for every point in config.WEATHER_POINTS (UTC index)."""
    path = config.DATA_RAW / "weather.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    frames = []
    for name, (lat, lon) in config.WEATHER_POINTS.items():
        js = _get_json(OPEN_METEO, {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ",".join(config.WEATHER_VARS),
            "timezone": "GMT",
        })
        df = pd.DataFrame(js["hourly"])
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        df.columns = [f"{name}__{c}" for c in df.columns]
        frames.append(df)
        time.sleep(1.0)
    w = pd.concat(frames, axis=1).astype("float64")

    path.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(path)
    return w


def download_gas(start=config.START, end=config.END, force=False):
    """Daily TTF front-month settlement (EUR/MWh), indexed by trading date."""
    path = config.DATA_RAW / "ttf.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)["ttf"]

    import yfinance as yf

    df = yf.download("TTF=F", start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                     progress=False, auto_adjust=False)
    s = df["Close"].squeeze().dropna().rename("ttf")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()

    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def load_all(force=False):
    return {
        "ch": download_prices("CH", force=force),
        "de": download_prices("DE-LU", force=force),
        "weather": download_weather(force=force),
        "ttf": download_gas(force=force),
    }
