"""The METAL MIND research experiment.

For every metal and forecast horizon it backtests a grid of models:

    baselines:  naive (no change), drift (mean return)
    models:     Ridge regression, Random Forest, Gradient Boosting
    features:   price only | price + sentiment | price + sentiment + events

and answers the project's research questions:

1. Can any model beat the naive baseline out of sample?        (summary table, DM test)
2. Does news sentiment add information beyond price features?  (feature-set comparison)
3. Which event types are followed by larger or directional
   price moves (weather, supply, policy, demand, macro)?       (event study)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from metalmind import evaluate, features, news, plotting, prices as prices_mod
from metalmind.config import get_metal
from metalmind.events import CATEGORIES
from metalmind.models import MODEL_LABELS, DriftModel, FeatureModel, NaiveModel

FEATURE_SETS = {
    "price": features.PRICE_FEATURES,
    "price+sentiment": features.PRICE_FEATURES + features.SENTIMENT_FEATURES,
    "price+sentiment+events": features.PRICE_FEATURES + features.SENTIMENT_FEATURES + features.EVENT_FEATURES,
}


def model_grid(with_news: bool) -> dict[str, object]:
    grid: dict[str, object] = {evaluate.NAIVE: NaiveModel, evaluate.DRIFT: DriftModel}
    sets = FEATURE_SETS if with_news else {"price": FEATURE_SETS["price"]}
    for kind, label in MODEL_LABELS.items():
        for fs_name, cols in sets.items():
            name = f"{label} [{fs_name}]"
            grid[name] = lambda kind=kind, cols=cols, name=name: FeatureModel(kind, cols, name)
    return grid


def event_study(prices: pd.Series, signals: pd.DataFrame) -> pd.DataFrame:
    """Next-day returns after days with each event type vs days without.

    Reports the mean signed return (direction) and mean absolute return
    (size of the move) with Welch t-test p-values.
    """
    fwd = (prices.shift(-1) / prices - 1).reindex(signals.index)
    valid = fwd.notna()
    rows = []
    flags = {f"ev_{c}": c for c in CATEGORIES}
    flags["news_count"] = "any news"
    for col, label in flags.items():
        mask = (signals[col] > 0) & valid
        other = (~(signals[col] > 0)) & valid
        a, b = fwd[mask], fwd[other]
        row = {"event": label, "days with event": int(mask.sum()), "days without": int(other.sum())}
        if len(a) >= 10 and len(b) >= 10:
            row["mean next-day return, event (%)"] = a.mean() * 100
            row["mean next-day return, no event (%)"] = b.mean() * 100
            row["p (direction)"] = stats.ttest_ind(a, b, equal_var=False).pvalue
            row["mean |move|, event (%)"] = a.abs().mean() * 100
            row["mean |move|, no event (%)"] = b.abs().mean() * 100
            row["p (size of move)"] = stats.ttest_ind(a.abs(), b.abs(), equal_var=False).pvalue
        rows.append(row)
    has_news = (signals["news_count"] > 0) & valid
    if has_news.sum() >= 30:
        rho, p = stats.spearmanr(signals.loc[has_news, "sentiment"], fwd[has_news])
        rows.append({"event": "sentiment vs next-day return (Spearman rho)", "days with event": int(has_news.sum()),
                     "mean next-day return, event (%)": rho, "p (direction)": p})
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.to_numpy():
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(
    metals: list[str],
    start: str = "2021-01-01",
    horizons: tuple[int, ...] = (1, 5),
    min_train: int = 250,
    refit_every: int = 20,
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    use_news: bool = True,
    prices_csv: dict[str, str] | None = None,
    progress=print,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    out = results_dir / f"experiment_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    summaries, studies, data_rows, importances = [], [], [], []

    for metal in metals:
        spec = get_metal(metal)
        if prices_csv and metal in prices_csv:
            series, source = prices_mod.load_csv(prices_csv[metal]), f"CSV {prices_csv[metal]}"
        else:
            series, source = prices_mod.get_price_history(metal, start=start, source="yahoo")
        series = series[series.index >= pd.Timestamp(start)]
        signals = None
        archive = news.load_archive(metal, data_dir) if use_news else pd.DataFrame()
        if use_news and not archive.empty:
            signals = news.daily_signals(archive, series.index)
        n_art = int(signals["news_count"].sum()) if signals is not None else 0
        data_rows.append({
            "metal": spec.name, "price source": source, "first day": f"{series.index[0]:%Y-%m-%d}",
            "last day": f"{series.index[-1]:%Y-%m-%d}", "trading days": len(series),
            "articles aligned": n_art,
            "days with news (%)": float((signals["news_count"] > 0).mean() * 100) if signals is not None else 0.0,
            **({f"{c} articles": int(signals[f"ev_{c}"].sum()) for c in CATEGORIES} if signals is not None else {}),
        })
        progress(f"{spec.name}: {len(series)} trading days, {n_art} aligned articles")

        grid = model_grid(with_news=signals is not None)
        for h in horizons:
            progress(f"  backtesting {len(grid)} models at {h}-day horizon...")
            bt = evaluate.run(series, signals, min_train=min_train, refit_every=refit_every, horizon=h, models=grid)
            s = bt.summary.copy()
            s.insert(0, "model", s.index)
            s.insert(0, "horizon (days)", h)
            s.insert(0, "metal", spec.name)
            s["test days"] = len(bt.predictions)
            s["test period"] = f"{bt.predictions.index[0]:%Y-%m-%d} to {bt.predictions.index[-1]:%Y-%m-%d}"
            summaries.append(s.reset_index(drop=True))
            bt.predictions.to_csv(out / f"predictions_{spec.name}_h{h}.csv")
            if h == horizons[0]:
                learned = s[~s["model"].isin([evaluate.NAIVE, evaluate.DRIFT])]
                best = learned.sort_values("MAE (₹/kg)").iloc[0]["model"]
                plotting.plot_backtest(bt.predictions.iloc[-250:], best, spec.name, out / f"backtest_{spec.name}.png")

        if signals is not None:
            st = event_study(series, signals)
            st.insert(0, "metal", spec.name)
            studies.append(st)
            X, y = features.build(series, signals, horizon=horizons[0])
            cols = FEATURE_SETS["price+sentiment+events"]
            imp = FeatureModel("forest", cols).fit(X, y).importances()
            importances.append(imp.rename(spec.name))

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    data_summary = pd.DataFrame(data_rows)
    data_summary.to_csv(out / "data_summary.csv", index=False)
    study = pd.concat(studies, ignore_index=True) if studies else pd.DataFrame()
    if not study.empty:
        study.to_csv(out / "event_study.csv", index=False)
    imp_df = pd.concat(importances, axis=1) if importances else pd.DataFrame()
    if not imp_df.empty:
        imp_df.to_csv(out / "feature_importance.csv")
    plotting.plot_skill(summary, horizons[0], out / "skill_vs_naive.png")

    config = {"metals": metals, "start": start, "horizons": list(horizons), "min_train": min_train,
              "refit_every": refit_every, "use_news": use_news, "run_id": run_id}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    _write_report(out, config, data_summary, summary, study, imp_df)
    return out


def _write_report(out: Path, config: dict, data: pd.DataFrame, summary: pd.DataFrame,
                  study: pd.DataFrame, imp: pd.DataFrame) -> None:
    keep = ["metal", "horizon (days)", "model", "MAE (₹/kg)", "RMSE (₹/kg)", "MAPE (%)",
            "Direction hit-rate (%)", "MAE vs naive (%)", "DM p-value"]
    parts = [
        f"# METAL MIND experiment {config['run_id']}",
        "",
        f"Walk-forward backtest, expanding window, first {config['min_train']} trading days for training, "
        f"models refit every {config['refit_every']} days. Horizons: {config['horizons']} trading days. "
        "Negative 'MAE vs naive' = better than the no-change baseline; 'DM p-value' < 0.05 = the difference "
        "in squared error vs naive is statistically significant (Diebold-Mariano).",
        "",
        "## Data",
        "",
        _markdown(data, "{:,.1f}"),
        "",
        "## Forecast accuracy",
        "",
    ]
    for (metal, h), g in summary.groupby(["metal", "horizon (days)"], sort=False):
        parts += [f"### {metal}, {h}-day horizon ({g['test days'].iloc[0]} test days, {g['test period'].iloc[0]})", "",
                  _markdown(g[keep].drop(columns=["metal", "horizon (days)"]).sort_values("MAE (₹/kg)")), ""]
    if not study.empty:
        parts += ["## Event study (next-day returns after news events)", "", _markdown(study), ""]
    if not imp.empty:
        parts += ["## Feature importance (Random Forest, all features, 1-day horizon)", "",
                  _markdown(imp.reset_index().rename(columns={"index": "feature"})), ""]
    (out / "report.md").write_text("\n".join(parts))
