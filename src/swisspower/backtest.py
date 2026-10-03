"""Walk-forward evaluation.

Every REFIT_EVERY_DAYS the models are re-estimated on the last
CALIBRATION_DAYS days that were known at that point, then used to forecast
the following days. No test-period information is used for fitting before
the day it is forecast, and no hyper-parameter was tuned on the test period.
"""
import time

import numpy as np
import pandas as pd

from . import config, features
from .models import GBM, LEAR


def walk_forward(m, test_start=config.TEST_START, test_end=config.END,
                 calib_days=config.CALIBRATION_DAYS, refit_every=config.REFIT_EVERY_DAYS,
                 verbose=True):
    Xw = features.wide_features(m)
    Xl = features.long_features(m)
    y_long = features.target_long(m)
    Yw = m["ch"]

    all_days = m["ch"].index
    test_days = all_days[(all_days >= test_start) & (all_days <= test_end)]
    test_days = features.complete_days(m, test_days)

    out = []
    t0 = time.time()
    for i in range(0, len(test_days), refit_every):
        block = test_days[i:i + refit_every]
        first = block[0]
        # Training days: the calibration window that ends the day before the
        # first forecast day (its prices were known on the issue date).
        train = all_days[(all_days < first) & (all_days >= first - pd.Timedelta(days=calib_days))]
        train = features.complete_days(m, train)

        lear = LEAR().fit(Xw.loc[train], Yw.loc[train])
        tr_idx = Xl.index.get_level_values("date").isin(train)
        gbm = GBM().fit(Xl[tr_idx], y_long[tr_idx])

        te_idx = Xl.index.get_level_values("date").isin(block)
        res = pd.DataFrame({
            "actual": y_long[te_idx],
            "lear": lear.predict(Xw.loc[block]).stack(),
            "gbm": gbm.predict(Xl[te_idx]),
        })
        out.append(res)
        if verbose and (i // refit_every) % 10 == 0:
            print(f"  {first.date()}  ({i}/{len(test_days)} days, {time.time() - t0:.0f}s)")

    fc = pd.concat(out).sort_index()
    fc.index = fc.index.set_names(["date", "hour"])
    naive = features.naive_forecast(m["ch"]).stack(future_stack=True)
    fc["naive"] = naive.reindex(fc.index).to_numpy()
    fc["ensemble"] = fc[["lear", "gbm"]].mean(axis=1)
    return fc[["actual", "naive", "lear", "gbm", "ensemble"]], (lear, gbm, Xl[tr_idx], y_long[tr_idx])
