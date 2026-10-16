"""Charts saved as PNG files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HISTORY = "#2a78d6"   # categorical slot 1
FORECAST = "#eb6834"  # categorical slot 2
INK = "#3d3d3a"
MUTED = "#8a8980"
GRID = "#e6e5e0"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=30)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_forecast(prices: pd.Series, fc: pd.DataFrame, metal: str, out: Path, history_days: int = 90) -> Path:
    hist = prices.iloc[-history_days:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist.index, hist.values, color=HISTORY, linewidth=2, label="Observed price")
    joined = pd.concat([hist.iloc[[-1]].rename("forecast"), fc["forecast"]])
    ax.plot(joined.index, joined.values, color=FORECAST, linewidth=2, linestyle="--", label="Forecast")
    if fc["lower"].notna().all():
        ax.fill_between(fc.index, fc["lower"], fc["upper"], color=FORECAST, alpha=0.15, linewidth=0,
                        label="Approx. 95% band (backtest errors)")
    ax.axvline(hist.index[-1], color=MUTED, linewidth=1)
    ax.annotate(f"₹{fc['forecast'].iloc[-1]:,.2f}", (fc.index[-1], fc["forecast"].iloc[-1]),
                xytext=(6, 0), textcoords="offset points", color=INK, va="center", fontsize=10)
    _style(ax, f"{metal.title()} price forecast, next {len(fc)} days", "₹ per kg")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK, borderaxespad=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_backtest(preds: pd.DataFrame, model: str, metal: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(preds.index, preds["actual"], color=HISTORY, linewidth=2, label="Actual next-day price")
    ax.plot(preds.index, preds[model], color=FORECAST, linewidth=2, label=f"Predicted: {model}")
    _style(ax, f"{metal.title()}: walk-forward backtest", "₹ per kg")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK, borderaxespad=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
