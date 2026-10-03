"""Walk-forward forecasts on the out-of-sample period + accuracy tables."""
import argparse

import pandas as pd

from _setup import *  # noqa: F401,F403
from swisspower import backtest, config, data, features, metrics

MODELS = ["naive", "lear", "gbm", "ensemble"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-days", type=int, default=config.CALIBRATION_DAYS)
    args = ap.parse_args()

    m = features.build_matrices(data.load_all())
    print(f"Walk-forward from {config.TEST_START}, calibration window {args.calib_days} days")
    fc, _ = backtest.walk_forward(m, calib_days=args.calib_days)

    tag = "" if args.calib_days == config.CALIBRATION_DAYS else f"_calib{args.calib_days}"
    for d in (config.RESULTS, config.DATA_PROCESSED):
        d.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(config.DATA_PROCESSED / f"forecasts{tag}.parquet")

    acc = metrics.accuracy_table(fc, MODELS)
    acc.round(3).to_csv(config.RESULTS / f"accuracy{tag}.csv")
    print("\nAccuracy, out-of-sample", fc.index.get_level_values("date").min().date(), "->",
          fc.index.get_level_values("date").max().date(), f"({fc.index.get_level_values('date').nunique()} days)")
    print(acc.round(3).to_string())

    by_year = metrics.mae_by(fc, MODELS, "year")
    by_year.round(2).to_csv(config.RESULTS / f"mae_by_year{tag}.csv")
    print("\nMAE by year\n", by_year.round(2).to_string())

    rows = []
    for a, b in [("lear", "naive"), ("gbm", "naive"), ("gbm", "lear"), ("lear", "gbm"), ("ensemble", "gbm"), ("ensemble", "lear")]:
        stat, p = metrics.dm_test(fc, a, b)
        rows.append({"model": a, "vs": b, "DM_stat": round(stat, 2), "p_value": round(p, 4)})
    dm = pd.DataFrame(rows)
    dm.to_csv(config.RESULTS / f"dm_tests{tag}.csv", index=False)
    print("\nDiebold-Mariano (H0: model not better than 'vs')\n", dm.to_string(index=False))
