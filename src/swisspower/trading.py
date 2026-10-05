"""Using the forecasts: day-ahead scheduling of a small storage asset.

Each day, the asset is scheduled on the *forecast* prices (mixed-integer
linear program),
then the schedule is settled at the *actual* day-ahead prices. Comparing with
perfect foresight tells how much of the available spread a forecast captures.

Note what matters here: the timing of cheap and expensive hours (the shape of
the day), much more than the price level. A forecast can have a good MAE and
still pick the wrong hours.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import config


def schedule(prices, power_mw, energy_mwh, round_trip_eff, cost_eur_per_mwh):
    """Optimal charge/discharge plan for one day given a price vector.

    Variables: charge c_t, discharge d_t, state of charge s_t and a binary
    mode u_t (1 = charging) for t = 0..T-1.
    s_t = s_{t-1} + eta * c_t - d_t / eta, empty at the start of the day,
    at most one full cycle per day.

    The binary is needed: with a pure LP, at negative prices the optimiser
    charges and discharges in the same hour to "burn" energy through the
    efficiency losses and get paid for it, which a real asset cannot do.
    """
    p = np.asarray(prices, dtype=float)
    T = len(p)
    eta = np.sqrt(round_trip_eff)
    n = 4 * T
    c_, d_, s_, u_ = (slice(k * T, (k + 1) * T) for k in range(4))

    # x = [c, d, s, u], milp minimises
    cost = np.zeros(n)
    cost[c_] = p
    cost[d_] = -p + cost_eur_per_mwh

    soc = np.zeros((T, n))
    for t in range(T):
        soc[t, c_.start + t] = -eta
        soc[t, d_.start + t] = 1 / eta
        soc[t, s_.start + t] = 1
        if t > 0:
            soc[t, s_.start + t - 1] = -1

    cycle = np.zeros((1, n))
    cycle[0, d_] = 1

    mode = np.zeros((2 * T, n))
    for t in range(T):
        mode[t, c_.start + t] = 1                  # c_t <= P * u_t
        mode[t, u_.start + t] = -power_mw
        mode[T + t, d_.start + t] = 1              # d_t <= P * (1 - u_t)
        mode[T + t, u_.start + t] = power_mw

    constraints = [
        LinearConstraint(soc, 0, 0),
        LinearConstraint(cycle, 0, energy_mwh),
        LinearConstraint(mode, -np.inf, np.r_[np.zeros(T), np.full(T, power_mw)]),
    ]
    upper = np.r_[np.full(2 * T, power_mw), np.full(T, energy_mwh), np.ones(T)]
    integrality = np.r_[np.zeros(3 * T), np.ones(T)]

    res = milp(cost, constraints=constraints, bounds=Bounds(np.zeros(n), upper), integrality=integrality)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[c_], res.x[d_]


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
