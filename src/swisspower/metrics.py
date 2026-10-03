"""Forecast accuracy metrics and the Diebold-Mariano test."""
import numpy as np
import pandas as pd
from scipy import stats


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def smape(y, f):
    denom = (np.abs(y) + np.abs(f)) / 2
    ok = denom > 1e-9
    return float(np.mean(np.abs(y - f)[ok] / denom[ok]) * 100)


def accuracy_table(fc, models, benchmark="naive"):
    rows = {}
    base = mae(fc["actual"], fc[benchmark])
    for m in models:
        rows[m] = {
            "MAE": mae(fc["actual"], fc[m]),
            "rMAE": mae(fc["actual"], fc[m]) / base,
            "RMSE": rmse(fc["actual"], fc[m]),
            "sMAPE_%": smape(fc["actual"].to_numpy(), fc[m].to_numpy()),
        }
    return pd.DataFrame(rows).T


def dm_test(fc, model_a, model_b):
    """Multivariate DM test (Ziel & Weron, 2018) on daily mean absolute errors.

    H0: model_a is not more accurate than model_b. A small p-value means
    model_a is significantly better. Daily aggregation removes most of the
    intra-day correlation of errors, so a plain t-type statistic is used.
    """
    err = (fc[[model_a, model_b]].sub(fc["actual"], axis=0)).abs()
    daily = err.groupby(level="date").mean()
    d = daily[model_b] - daily[model_a]  # > 0 when model_a is better
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(stat), float(1 - stats.norm.cdf(stat))


def mae_by(fc, models, level):
    err = fc[models].sub(fc["actual"], axis=0).abs()
    if level == "year":
        key = fc.index.get_level_values("date").year
    else:
        key = fc.index.get_level_values(level)
    return err.groupby(key).mean()
