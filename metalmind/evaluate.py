"""Walk-forward backtesting and forecast-accuracy statistics.

For each test day t, models are trained only on rows whose target was already
known at t (an expanding window that stops `horizon` days before t), then asked
for the return from t to t+horizon. Nothing from the future is ever seen, which
a random train/test split would leak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from metalmind import features
from metalmind.models import DriftModel, FeatureModel, NaiveModel

ModelFactory = Callable[[], object]
NAIVE = "Naive (no change)"
DRIFT = "Drift (mean return)"


@dataclass
class Backtest:
    predictions: pd.DataFrame      # one row per test day; price columns per model
    summary: pd.DataFrame          # one row per model: error metrics and significance
    news_coverage: float | None    # share of test days with at least one article
    horizon: int = 1


def default_models(with_news: bool) -> dict[str, ModelFactory]:
    models: dict[str, ModelFactory] = {
        NAIVE: NaiveModel,
        DRIFT: DriftModel,
        "Random Forest: price only": lambda: FeatureModel("forest", features.PRICE_FEATURES, "Random Forest: price only"),
    }
    if with_news:
        cols = features.PRICE_FEATURES + features.NEWS_FEATURES
        models["Random Forest: price + news"] = lambda: FeatureModel("forest", cols, "Random Forest: price + news")
    return models


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.Series,
    models: dict[str, ModelFactory],
    min_train: int = 60,
    refit_every: int = 5,
    horizon: int = 1,
) -> pd.DataFrame:
    if len(X) <= min_train + horizon:
        raise ValueError(
            f"Need more than {min_train + horizon} usable days for a backtest, have {len(X)}. "
            "Supply a longer price history or lower --min-train."
        )
    first = min_train + horizon - 1
    blocks = []
    for i in range(first, len(X), refit_every):
        train_end = i - horizon + 1              # rows < train_end have targets known by day i
        block = slice(i, min(i + refit_every, len(X)))
        Xb = X.iloc[block]
        last = prices.reindex(Xb.index).to_numpy(dtype=float)
        out = pd.DataFrame({"last_price": last, "actual": last * (1 + y.iloc[block].to_numpy())}, index=Xb.index)
        for name, make in models.items():
            model = make().fit(X.iloc[:train_end], y.iloc[:train_end])
            out[name] = last * (1 + np.asarray(model.predict(Xb), dtype=float))
        blocks.append(out)
    preds = pd.concat(blocks)
    preds.index.name = "date"
    return preds


def diebold_mariano(e_model: np.ndarray, e_base: np.ndarray, horizon: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test on squared errors. Returns (statistic, two-sided p-value).

    Negative statistic = the model has lower squared error than the baseline.
    Uses a Newey-West variance with horizon-1 lags because multi-day forecast
    errors overlap.
    """
    d = e_model**2 - e_base**2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return float("nan"), float("nan")
    d_c = d - d.mean()
    var = np.dot(d_c, d_c) / n
    for lag in range(1, horizon):
        var += 2 * (1 - lag / horizon) * np.dot(d_c[lag:], d_c[:-lag]) / n
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(var / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def metrics(actual: np.ndarray, pred: np.ndarray, last: np.ndarray) -> dict[str, float]:
    err = pred - actual
    move_true = np.sign(actual - last)
    move_pred = np.sign(pred - last)
    mask = (move_true != 0) & (move_pred != 0)
    return {
        "MAE (₹/kg)": float(np.mean(np.abs(err))),
        "RMSE (₹/kg)": float(np.sqrt(np.mean(err**2))),
        "MAPE (%)": float(np.mean(np.abs(err) / actual) * 100),
        "Direction hit-rate (%)": float(np.mean(move_true[mask] == move_pred[mask]) * 100) if mask.any() else np.nan,
    }


def summarise(preds: pd.DataFrame, model_names: list[str], horizon: int = 1) -> pd.DataFrame:
    actual = preds["actual"].to_numpy()
    last = preds["last_price"].to_numpy()
    e_naive = preds[NAIVE].to_numpy() - actual if NAIVE in preds else None
    table = {}
    for name in model_names:
        row = metrics(actual, preds[name].to_numpy(), last)
        if e_naive is not None and name != NAIVE:
            stat, p = diebold_mariano(preds[name].to_numpy() - actual, e_naive, horizon)
            row["DM stat vs naive"], row["DM p-value"] = stat, p
        else:
            row["DM stat vs naive"], row["DM p-value"] = np.nan, np.nan
        table[name] = row
    out = pd.DataFrame(table).T
    if NAIVE in out.index:
        naive = out.loc[NAIVE, "MAE (₹/kg)"]
        out["MAE vs naive (%)"] = (out["MAE (₹/kg)"] / naive - 1) * 100
    return out


def run(
    prices: pd.Series,
    sentiment: pd.DataFrame | None = None,
    min_train: int = 60,
    refit_every: int = 5,
    horizon: int = 1,
    models: dict[str, ModelFactory] | None = None,
) -> Backtest:
    X, y = features.build(prices, sentiment, horizon=horizon)
    models = models or default_models(with_news=sentiment is not None)
    preds = walk_forward(X, y, prices, models, min_train=min_train, refit_every=refit_every, horizon=horizon)
    coverage = None
    if sentiment is not None:
        coverage = float((sentiment.reindex(preds.index)["news_count"] > 0).mean())
    return Backtest(preds, summarise(preds, list(models), horizon), coverage, horizon)
