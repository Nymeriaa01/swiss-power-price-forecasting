"""Using the forecasts: day-ahead scheduling of a small storage asset.

Each day, the asset is scheduled on the *forecast* prices (linear program),
then the schedule is settled at the *actual* day-ahead prices. Comparing with
perfect foresight tells how much of the available spread a forecast captures.

Note what matters here: the timing of cheap and expensive hours (the shape of
the day), much more than the price level. A forecast can have a good MAE and
still pick the wrong hours.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from . import config


def schedule(prices, power_mw, energy_mwh, round_trip_eff, cost_eur_per_mwh):
    """Optimal charge/discharge plan for one day given a price vector.

    Variables: charge c_t, discharge d_t, state of charge s_t (t = 0..T-1).
    s_t = s_{t-1} + eta * c_t - d_t / eta, empty at the start of the day,
    at most one full cycle per day.
    """
    p = np.asarray(prices, dtype=float)
    T = len(p)
    eta = np.sqrt(round_trip_eff)
    # x = [c (T), d (T), s (T)], linprog minimises
    cost = np.concatenate([p, -p + cost_eur_per_mwh, np.zeros(T)])

    A_eq = np.zeros((T, 3 * T))
    for t in range(T):
        A_eq[t, t] = -eta
        A_eq[t, T + t] = 1 / eta
        A_eq[t, 2 * T + t] = 1
        if t > 0:
            A_eq[t, 2 * T + t - 1] = -1
    b_eq = np.zeros(T)

    A_ub = np.zeros((1, 3 * T))
    A_ub[0, T:2 * T] = 1  # total discharge <= one cycle
    b_ub = [energy_mwh]

    bounds = [(0, power_mw)] * (2 * T) + [(0, energy_mwh)] * T
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    c, d = res.x[:T], res.x[T:2 * T]
    return c, d


def daily_pnl(actual, c, d, cost_eur_per_mwh):
    return float(np.sum(actual * (d - c)) - cost_eur_per_mwh * np.sum(d))


def run_strategy(fc, models, storage=config.STORAGE):
    """Daily P&L (EUR) of the storage asset scheduled on each model's forecast."""
    pnl = {m: {} for m in ["perfect_foresight", *models]}
    for day, g in fc.groupby(level="date"):
        actual = g["actual"].to_numpy()
        for m in pnl:
            signal = actual if m == "perfect_foresight" else g[m].to_numpy()
            c, d = schedule(signal, **storage)
            pnl[m][day] = daily_pnl(actual, c, d, storage["cost_eur_per_mwh"])
    return pd.DataFrame(pnl)


def max_drawdown(cum):
    return float((cum - cum.cummax()).min())


def strategy_table(pnl):
    perfect = pnl["perfect_foresight"].sum()
    rows = {}
    for m in pnl.columns:
        s = pnl[m]
        rows[m] = {
            "total_pnl_eur": s.sum(),
            "capture_%": 100 * s.sum() / perfect,
            "mean_daily_eur": s.mean(),
            "sharpe_ann": s.mean() / s.std(ddof=1) * np.sqrt(365),
            "max_drawdown_eur": max_drawdown(s.cumsum()),
            "losing_days_%": 100 * (s < 0).mean(),
        }
    return pd.DataFrame(rows).T
