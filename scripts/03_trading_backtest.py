"""Storage scheduling backtest on the walk-forward forecasts + figures."""
import pandas as pd

from _setup import *  # noqa: F401,F403
from swisspower import config, data, metrics, plots, trading

MODELS = ["naive", "lear", "gbm", "ensemble"]

if __name__ == "__main__":
    fc = pd.read_parquet(config.DATA_PROCESSED / "forecasts.parquet")

    pnl = trading.run_strategy(fc, MODELS)
    pnl.to_csv(config.RESULTS / "storage_daily_pnl.csv")
    table = trading.strategy_table(pnl)
    table.round(2).to_csv(config.RESULTS / "storage_summary.csv")
    print(table.round(2).to_string())

    raw = data.load_all()
    plots.price_history(raw["ch"], raw["de"])
    plots.example_week(fc, "2025-05-12")
    plots.error_by_hour(metrics.mae_by(fc, ["naive", "lear", "gbm"], "hour"))
    plots.cumulative_pnl(pnl)
    print("figures written to", config.FIGURES)
