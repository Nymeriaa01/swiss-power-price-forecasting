"""Forecasting models.

- LEAR: 24 hour-specific LASSO regressions on an asinh-transformed problem,
  the standard linear benchmark in day-ahead price forecasting
  (Lago, Marcjasz, De Schutter, Weron, 2021).
- GBM: one LightGBM model on the long (day, hour) table, trained with an L1
  loss, since price spikes would otherwise dominate the fit.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC

from . import config


class AsinhScaler:
    """Median/MAD normalisation followed by asinh (Uniejewski et al., 2018).

    Handles negative prices and dampens spikes, which plain log cannot do.
    """

    def fit(self, x):
        self.med = np.nanmedian(x, axis=0)
        mad = np.nanmedian(np.abs(x - self.med), axis=0) / 0.6745
        self.mad = np.where(mad > 1e-9, mad, 1.0)
        return self

    def transform(self, x):
        return np.arcsinh((x - self.med) / self.mad)

    def inverse(self, z):
        return np.sinh(z) * self.mad + self.med


class LEAR:
    name = "lear"

    def __init__(self, binary_prefixes=("dow_", "holiday")):
        self.binary_prefixes = binary_prefixes

    def fit(self, X, Y):
        self.cols = X.columns
        cont = [not c.startswith(self.binary_prefixes) for c in X.columns]
        self.cont = np.array(cont)
        x = X.to_numpy(dtype=float)
        # The rare gaps in inputs (weather, gas, DE prices) are filled with the
        # training median; the target itself is never imputed.
        self.fill = np.nanmedian(x, axis=0)
        self.x_scaler = AsinhScaler().fit(x[:, self.cont])
        self.y_scaler = AsinhScaler().fit(Y.to_numpy(dtype=float))
        xs = self._tx(x)
        ys = self.y_scaler.transform(Y.to_numpy(dtype=float))
        self.models = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for h in range(ys.shape[1]):
                self.models.append(LassoLarsIC(criterion="aic", max_iter=2500).fit(xs, ys[:, h]))
        return self

    def _tx(self, x):
        x = np.where(np.isnan(x), self.fill, x)
        x[:, self.cont] = self.x_scaler.transform(x[:, self.cont])
        return x

    def predict(self, X):
        xs = self._tx(X[self.cols].to_numpy(dtype=float))
        z = np.column_stack([m.predict(xs) for m in self.models])
        return pd.DataFrame(self.y_scaler.inverse(z), index=X.index, columns=range(z.shape[1]))


GBM_PARAMS = dict(
    objective="l1",
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=config.SEED,
    verbose=-1,
)


class GBM:
    name = "gbm"

    def __init__(self, **params):
        self.params = {**GBM_PARAMS, **params}

    def fit(self, X, y):
        self.cols = X.columns
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X.reset_index(level="hour"), y, categorical_feature=["hour", "dow"])
        return self

    def predict(self, X):
        p = self.model.predict(X[self.cols].reset_index(level="hour"))
        return pd.Series(p, index=X.index)
