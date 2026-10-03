import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

COLORS = {"actual": "#222222", "naive": "#9e9e9e", "lear": "#1f77b4",
          "gbm": "#ff7f0e", "ensemble": "#2ca02c", "perfect_foresight": "#222222"}

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def price_history(ch, de):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ch.tz_convert(config.TZ).resample("W").mean().plot(ax=ax, label="Switzerland (CH)", color="#c0392b")
    de.tz_convert(config.TZ).resample("W").mean().plot(ax=ax, label="Germany-Luxembourg (DE-LU)", color="#555555", alpha=0.7)
    ax.axvline(pd.Timestamp(config.TEST_START, tz=config.TZ), color="k", ls="--", lw=1)
    ax.text(pd.Timestamp(config.TEST_START, tz=config.TZ), ax.get_ylim()[1] * 0.92, "  out-of-sample →", fontsize=9)
    ax.set_ylabel("EUR/MWh (weekly mean)")
    ax.set_xlabel("")
    ax.set_title("Day-ahead prices: the 2022 crisis and the return to a lower regime")
    ax.legend(frameon=False)
    _save(fig, "01_price_history.png")


def example_week(fc, start):
    week = fc.loc[pd.Timestamp(start):pd.Timestamp(start) + pd.Timedelta(days=6)]
    x = range(len(week))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for col in ["actual", "naive", "lear", "gbm"]:
        ax.plot(x, week[col].to_numpy(), label=col, color=COLORS[col],
                lw=2 if col == "actual" else 1.3, ls=":" if col == "naive" else "-")
    days = week.index.get_level_values("date").unique()
    ax.set_xticks(range(0, len(week), 24), [d.strftime("%a %d %b") for d in days])
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Example out-of-sample week (CH), from {pd.Timestamp(start):%d %b %Y}")
    ax.legend(frameon=False, ncol=4)
    _save(fig, "02_example_week.png")


def error_by_hour(mae_hour):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for col in mae_hour.columns:
        ax.plot(mae_hour.index, mae_hour[col], marker="o", ms=3, label=col, color=COLORS[col])
    ax.set_xlabel("Delivery hour (local time)")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.set_title("Errors peak at midday, when solar output sets the price")
    ax.legend(frameon=False)
    _save(fig, "03_mae_by_hour.png")


def cumulative_pnl(pnl):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for col in pnl.columns:
        ax.plot(pnl.index, pnl[col].cumsum() / 1000, label=col, color=COLORS.get(col),
                lw=2 if col == "perfect_foresight" else 1.4, ls="--" if col == "perfect_foresight" else "-")
    ax.set_ylabel("Cumulative P&L (kEUR)")
    ax.set_title("1 MW / 2 MWh storage scheduled on each forecast, settled at actual prices")
    ax.legend(frameon=False)
    _save(fig, "04_storage_cumulative_pnl.png")


def shap_summary(shap_values, X):
    import shap

    shap.summary_plot(shap_values, X, show=False, max_display=15, plot_size=(8, 5))
    fig = plt.gcf()
    plt.title("What drives the GBM forecast (SHAP, last calibration window)")
    _save(fig, "05_shap_summary.png")
