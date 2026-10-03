"""SHAP explanation of the GBM refitted on the last calibration window."""
import numpy as np
import pandas as pd

from _setup import *  # noqa: F401,F403
from swisspower import config, data, features, plots
from swisspower.models import GBM

if __name__ == "__main__":
    import shap

    m = features.build_matrices(data.load_all())
    X, y = features.long_features(m), features.target_long(m)
    days = m["ch"].index
    last = days[days <= config.END][-1]
    train = features.complete_days(m, days[(days <= last) & (days > last - pd.Timedelta(days=config.CALIBRATION_DAYS))])
    idx = X.index.get_level_values("date").isin(train)

    gbm = GBM().fit(X[idx], y[idx])
    Xt = X[idx].reset_index(level="hour")[["hour", *gbm.cols]]
    sample = Xt.sample(4000, random_state=config.SEED)
    sv = shap.TreeExplainer(gbm.model).shap_values(sample)

    imp = pd.Series(np.abs(sv).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    imp.round(3).to_csv(config.RESULTS / "shap_importance.csv", header=["mean_abs_shap_eur"])
    print(imp.round(2).head(15).to_string())
    plots.shap_summary(sv, sample)
