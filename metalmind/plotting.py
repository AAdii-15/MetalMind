"""Charts saved as PNG files."""

from __future__ import annotations

import re
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
    _style(ax, f"{metal.replace('_', ' ').title()} price forecast, next {len(fc)} days", "₹ per kg")
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
    _style(ax, f"{metal.replace('_', ' ').title()}: walk-forward backtest", "₹ per kg")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, labelcolor=INK, borderaxespad=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


FEATURE_SET_COLORS = {"price": "#2a78d6", "price+sentiment": "#eb6834", "price+sentiment+events": "#1baf7a"}


def plot_skill(summary: pd.DataFrame, horizon: int, out: Path, ncols: int | None = None) -> Path:
    """Small multiples: % MAE change vs the naive baseline for every model, one panel per metal."""
    data = summary[(summary["horizon (days)"] == horizon) & (summary["model"] != "Naive (no change)")]
    metals = list(dict.fromkeys(data["metal"]))
    ncols = ncols or len(metals)
    nrows = -(-len(metals) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.2 * nrows + 0.8), sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(metals):]:
        ax.set_visible(False)
    order = list(dict.fromkeys(data["model"]))[::-1]
    for ax, metal in zip(flat, metals):
        d = data[data["metal"] == metal].set_index("model").reindex(order)
        parsed = [re.match(r"^(.*) \[(.*)\]$", m) for m in d.index]
        colors = [FEATURE_SET_COLORS.get(p.group(2), MUTED) if p else MUTED for p in parsed]
        ax.barh(range(len(d)), d["MAE vs naive (%)"], color=colors, height=0.7)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([p.group(1) if p else m for p, m in zip(parsed, d.index)], fontsize=9, color=INK)
        ax.set_title(metal.replace("_", " ").title(), loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(axis="x", colors=MUTED)
        ax.set_xlabel("MAE vs naive (%)  ← better | worse →", color=MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FEATURE_SET_COLORS.values()] + \
              [plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    lo = min(-0.5, float(data["MAE vs naive (%)"].min()) * 1.15)
    hi = max(0.5, float(data["MAE vs naive (%)"].max()) * 1.15)
    for ax in flat:
        ax.set_xlim(lo, hi)
    fig.suptitle(f"Forecast error vs the naive no-change baseline, {horizon}-day horizon",
                 x=0.01, ha="left", y=1 - 0.015 / nrows, fontsize=13, color=INK, fontweight="bold")
    fig.legend(handles, ["features: " + k for k in FEATURE_SET_COLORS] + ["baseline (drift)"],
               loc="upper left", bbox_to_anchor=(0.005, 1 - 0.07 / nrows), ncol=2 if ncols <= 2 else 4,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.16 if ncols > 2 else 0.12) / nrows))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
