"""Phase 2: can news predict how MUCH metal prices will move?

Phase 1 showed that neither models nor news predict the *direction* of the next
price move better than "no change". For procurement, the size of the move
matters as much: it decides how wide a price-escalation margin in a tender
quote must be. This module asks:

1. Can models forecast next-week volatility better than simply assuming
   "recent volatility continues" (persistence)?
2. Does news (volume, sentiment, event tags) add information to that forecast?
3. Which way of sizing a 95% price band for the next 5 trading days is
   actually right about 95% of the time, and how wide is it?

Target: realised volatility over the next h trading days,
    sigma_fwd(t) = sqrt( mean( r(t+1)^2 ... r(t+h)^2 ) )   (daily, in %)
Models predict log(sigma_fwd) and are scored on sigma itself.
All features in row t use data up to day t only; training rows stop h days
before each test day, so no target overlaps the future.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metalmind import news, prices as prices_mod
from metalmind.config import get_metal
from metalmind.evaluate import diebold_mariano
from metalmind.events import CATEGORIES
from metalmind.models import FeatureModel

VOL_FEATURES = ["vol_5", "vol_20", "vol_60", "abs_ret_1", "abs_ret_5", "vol_ratio"]
NEWS_VOL_FEATURES = ["log_news_1d", "log_news_5d", "news_surprise", "abs_sent_3d", "sentiment_3d"] + \
    [f"ev_{c}_5d" for c in CATEGORIES]
PERSIST = "Persistence (last 20-day volatility)"
LONGRUN = "Long-run average volatility"
Z95 = 1.959964


def build(prices: pd.Series, signals: pd.DataFrame | None, horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Features (row t, information up to t) and the forward realised-volatility target (in %)."""
    p = prices.astype(float)
    r = p.pct_change() * 100
    f = pd.DataFrame(index=p.index)
    f["vol_5"] = np.sqrt((r**2).rolling(5).mean())
    f["vol_20"] = np.sqrt((r**2).rolling(20).mean())
    f["vol_60"] = np.sqrt((r**2).rolling(60).mean())
    f["abs_ret_1"] = r.abs()
    f["abs_ret_5"] = (p.pct_change(5) * 100).abs()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"]
    if signals is not None:
        s = signals.reindex(p.index).fillna(0)
        n = s["news_count"].astype(float)
        f["log_news_1d"] = np.log1p(n)
        f["log_news_5d"] = np.log1p(n.rolling(5, min_periods=1).sum())
        base = n.rolling(60, min_periods=20).mean()
        f["news_surprise"] = np.log1p(n.rolling(5, min_periods=1).mean()) - np.log1p(base)
        f["abs_sent_3d"] = s["sentiment"].abs().rolling(3, min_periods=1).mean()
        f["sentiment_3d"] = s["sentiment"].rolling(3, min_periods=1).mean()
        for c in CATEGORIES:
            col = f"ev_{c}"
            f[f"{col}_5d"] = np.log1p(s[col].astype(float).rolling(5, min_periods=1).sum()) if col in s else 0.0
    fwd = np.sqrt(sum(r.shift(-k) ** 2 for k in range(1, horizon + 1)) / horizon)
    fwd.name = "sigma_fwd"
    data = f.join(fwd).replace([np.inf, -np.inf], np.nan).dropna()
    return data[f.columns], data["sigma_fwd"]


def model_grid(with_news: bool) -> dict[str, object]:
    grid = {}
    sets = {"volatility history": VOL_FEATURES}
    if with_news:
        sets["volatility history + news"] = VOL_FEATURES + NEWS_VOL_FEATURES
    for kind, label in (("ridge", "Ridge regression"), ("forest", "Random Forest"), ("boosting", "Gradient Boosting")):
        for fs, cols in sets.items():
            name = f"{label} [{fs}]"
            grid[name] = lambda kind=kind, cols=cols, name=name: FeatureModel(kind, cols, name)
    return grid


def walk_forward(X: pd.DataFrame, y: pd.Series, grid: dict, horizon: int = 5,
                 min_train: int = 250, refit_every: int = 20) -> pd.DataFrame:
    """Expanding-window backtest. Predictions are sigma (in %), one column per method."""
    if len(X) <= min_train + horizon:
        raise ValueError(f"Need more than {min_train + horizon} usable days, have {len(X)}.")
    first = min_train + horizon - 1
    blocks = []
    logy = np.log(y.clip(lower=1e-6))
    for i in range(first, len(X), refit_every):
        train_end = i - horizon + 1                       # targets of rows < train_end are known on day i
        block = slice(i, min(i + refit_every, len(X)))
        Xb = X.iloc[block]
        out = pd.DataFrame({"actual": y.iloc[block].to_numpy(), PERSIST: Xb["vol_20"].to_numpy(),
                            LONGRUN: np.full(len(Xb), float(y.iloc[:train_end].mean()))}, index=Xb.index)
        for name, make in grid.items():
            m = make().fit(X.iloc[:train_end], logy.iloc[:train_end])
            out[name] = np.exp(np.asarray(m.predict(Xb), dtype=float))
        blocks.append(out)
    preds = pd.concat(blocks)
    preds.index.name = "date"
    return preds


def summarise(preds: pd.DataFrame, horizon: int) -> pd.DataFrame:
    a = preds["actual"].to_numpy()
    e_p = preds[PERSIST].to_numpy() - a
    rows = {}
    for col in preds.columns.drop("actual"):
        e = preds[col].to_numpy() - a
        stat, p = diebold_mariano(e, e_p, horizon) if col != PERSIST else (np.nan, np.nan)
        rows[col] = {"MAE (vol %)": np.mean(np.abs(e)), "RMSE (vol %)": np.sqrt(np.mean(e**2)),
                     "Correlation with actual": stats.spearmanr(preds[col], a)[0],
                     "DM stat vs persistence": stat, "DM p-value": p}
    out = pd.DataFrame(rows).T
    out["MAE vs persistence (%)"] = (out["MAE (vol %)"] / out.loc[PERSIST, "MAE (vol %)"] - 1) * 100
    return out


CALIBRATED = "Best model, calibrated multiplier"


def band_coverage(prices: pd.Series, preds: pd.DataFrame, methods: list[str], horizon: int = 5,
                  calibrate: str | None = None, warmup: int = 60) -> pd.DataFrame:
    """How often the actual h-day price change fell inside a 95% band sized by each method.

    Normal-theory bands (1.96 x sigma x sqrt(h)) are too narrow for fat-tailed metal prices.
    With `calibrate`, an extra band uses that method's sigma but replaces 1.96 with the 95th
    percentile of past |move| / (sigma x sqrt(h)), using only days whose outcome was already known.
    """
    p = prices.astype(float)
    fwd_ret = (p.shift(-horizon) / p - 1).reindex(preds.index) * 100
    valid = fwd_ret.notna()
    halves = {m: Z95 * preds[m] * np.sqrt(horizon) for m in methods}
    if calibrate:
        z = (fwd_ret.abs() / (preds[calibrate] * np.sqrt(horizon))).to_numpy()
        mult = np.full(len(z), Z95)
        for i in range(len(z)):
            known = z[: max(0, i - horizon + 1)]            # outcomes known by day i
            known = known[~np.isnan(known)]
            if len(known) >= warmup:
                mult[i] = np.quantile(known, 0.95)
        halves[CALIBRATED] = pd.Series(mult, index=preds.index) * preds[calibrate] * np.sqrt(horizon)
    rows = {}
    for m, half in halves.items():
        inside = fwd_ret.abs() <= half
        rows[m] = {"Coverage (%)": inside[valid].mean() * 100, "Average band (± %)": half[valid].mean(),
                   "Misses (days)": int((~inside[valid]).sum()), "Test days": int(valid.sum())}
    return pd.DataFrame(rows).T


def news_link(X: pd.DataFrame, y: pd.Series) -> dict:
    """Is news volume related to next-week volatility, beyond what recent volatility already says?"""
    if "log_news_5d" not in X:
        return {}
    surprise_vol = np.log(y) - np.log(X["vol_20"])               # volatility above / below its recent level
    r1, p1 = stats.spearmanr(X["log_news_5d"], y)
    r2, p2 = stats.spearmanr(X["news_surprise"], surprise_vol)
    return {"rho(news volume, next-week vol)": r1, "p1": p1,
            "rho(news surprise, vol above recent level)": r2, "p2": p2, "days": len(X)}


def _md(df: pd.DataFrame) -> str:
    cols = ["method"] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, r in df.iterrows():
        cells = [str(idx)] + ["" if (isinstance(v, float) and np.isnan(v)) else
                              (f"{v:,.3f}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.to_numpy()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(metals: list[str], start: str = "2021-01-01", horizon: int = 5, min_train: int = 250,
        refit_every: int = 20, data_dir: Path = Path("data"), results_dir: Path = Path("results"),
        source: str = "yahoo", prices_csv: dict[str, str] | None = None, progress=print,
        sentiment: str = "vader") -> Path:
    from metalmind import plotting

    out = results_dir / f"volatility_{datetime.now():%Y%m%d_%H%M}" if sentiment == "vader" else \
        results_dir / f"volatility_{datetime.now():%Y%m%d_%H%M}_{sentiment}"
    out.mkdir(parents=True, exist_ok=True)
    summaries, coverages, links, figs = [], [], [], {}
    for metal in metals:
        spec = get_metal(metal)
        if prices_csv and metal in prices_csv:
            series = prices_mod.load_csv(prices_csv[metal])
        else:
            series, _ = prices_mod.get_price_history(metal, data_dir=data_dir, source=source, start=start)
        series = series[series.index >= pd.Timestamp(start)]
        archive = news.load_archive(metal, data_dir)
        signals = news.daily_signals(archive, series.index, news.SENTIMENT_COLUMNS[sentiment]) if not archive.empty else None
        X, y = build(series, signals, horizon)
        grid = model_grid(signals is not None)
        progress(f"{spec.name}: {len(X)} usable days, backtesting {len(grid) + 2} volatility forecasts...")
        preds = walk_forward(X, y, grid, horizon, min_train, refit_every)
        preds.to_csv(out / f"vol_predictions_{spec.name}.csv")
        s = summarise(preds, horizon)
        s.insert(0, "metal", spec.name)
        summaries.append(s)
        learned = [c for c in preds.columns if "[" in c]
        best = s.loc[learned, "MAE (vol %)"].astype(float).idxmin()
        cov = band_coverage(series, preds, [LONGRUN, PERSIST, best], horizon, calibrate=best)
        cov.insert(0, "metal", spec.name)
        coverages.append(cov)
        lk = news_link(X.loc[preds.index], y.loc[preds.index])
        if lk:
            links.append({"metal": spec.name, **lk})
        figs[spec.name] = plotting.plot_volatility(preds.iloc[-250:], best, spec.name, out / f"vol_{spec.name}.png")
    summary = pd.concat(summaries)
    coverage = pd.concat(coverages)
    summary.to_csv(out / "vol_summary.csv")
    coverage.to_csv(out / "band_coverage.csv")
    link_df = pd.DataFrame(links)
    if not link_df.empty:
        link_df.to_csv(out / "news_volatility_link.csv", index=False)
    plotting.plot_coverage(coverage, out / "band_coverage.png")
    cfg = {"metals": metals, "start": start, "horizon": horizon, "min_train": min_train, "refit_every": refit_every,
           "sentiment": sentiment}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    parts = [f"# METAL MIND phase 2: volatility ({horizon}-day) {out.name}", "",
             f"Target: realised volatility of daily returns over the next {horizon} trading days (in %). "
             f"Walk-forward, first {min_train} days for training, refit every {refit_every} days. "
             "Negative 'MAE vs persistence' = better than assuming recent volatility continues.", ""]
    for metal, g in summary.groupby("metal", sort=False):
        parts += [f"## {metal}", "", _md(g.drop(columns="metal").sort_values("MAE (vol %)")), ""]
    parts += ["## 95% price-band coverage (next 5 trading days)", "", _md(coverage), ""]
    if not link_df.empty:
        parts += ["## News volume vs volatility (Spearman)", "", _md(link_df.set_index("metal")), ""]
    (out / "report.md").write_text("\n".join(parts))
    return out
