import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swisspower import trading  # noqa: E402

ASSET = dict(power_mw=1.0, energy_mwh=2.0, round_trip_eff=0.85, cost_eur_per_mwh=5.0)


def test_flat_prices_no_trade():
    c, d = trading.schedule(np.full(24, 50.0), **ASSET)
    assert np.allclose(c, 0) and np.allclose(d, 0)


def test_buys_low_sells_high_within_limits():
    p = np.full(24, 50.0)
    p[2:4] = 10.0
    p[18:20] = 150.0
    c, d = trading.schedule(p, **ASSET)
    assert c[2:4].sum() > 1.5 and d[18:20].sum() > 1.5
    assert d.sum() <= ASSET["energy_mwh"] + 1e-6
    assert (c <= ASSET["power_mw"] + 1e-6).all() and (d <= ASSET["power_mw"] + 1e-6).all()
    assert trading.daily_pnl(p, c, d, ASSET["cost_eur_per_mwh"]) > 0


def test_perfect_foresight_never_loses():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.normal(80, 40, 24)
        c, d = trading.schedule(p, **ASSET)
        assert trading.daily_pnl(p, c, d, ASSET["cost_eur_per_mwh"]) >= -1e-6
