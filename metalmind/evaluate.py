"""Walk-forward backtesting.

For each test day t the models are trained only on rows strictly before t
(an expanding window), then asked for the return from t to t+1. Nothing from
the future is ever seen, which is what a random train/test split would leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from metalmind import features
from metalmind.models import DriftModel, ForestModel, NaiveModel

ModelFactory = Callable[[], object]


@dataclass
class Backtest:
    predictions: pd.DataFrame      # one row per test day; price columns per model
    summary: pd.DataFrame          # one row per model: MAE, RMSE, MAPE, direction
    news_coverage: float | None    # share of trading days with at least one article


def default_models(with_news: bool) -> dict[str, ModelFactory]:
    models: dict[str, ModelFactory] = {
        "Naive (no change)": NaiveModel,
        "Drift (mean return)": DriftModel,
        "Random Forest: price only": lambda: ForestModel(features.PRICE_FEATURES, "Random Forest: price only"),
    }
    if with_news:
        cols = features.PRICE_FEATURES + features.NEWS_FEATURES
        models["Random Forest: price + news"] = lambda: ForestModel(cols, "Random Forest: price + news")
    return models


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.Series,
    models: dict[str, ModelFactory],
    min_train: int = 60,
    refit_every: int = 5,
) -> pd.DataFrame:
    if len(X) <= min_train:
        raise ValueError(
            f"Need more than {min_train} usable days for a backtest, have {len(X)}. "
            "Supply a longer price history or lower --min-train."
        )
    fitted: dict[str, object] = {}
    rows = []
    for i in range(min_train, len(X)):
        if (i - min_train) % refit_every == 0:
            for name, make in models.items():
                fitted[name] = make().fit(X.iloc[:i], y.iloc[:i])
        day = X.index[i]
        last = float(prices.loc[day])
        row = {"date": day, "last_price": last, "actual": last * (1 + y.iloc[i])}
        for name, model in fitted.items():
            row[name] = last * (1 + float(model.predict(X.iloc[[i]])[0]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


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


def summarise(preds: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    table = {
        name: metrics(preds["actual"].to_numpy(), preds[name].to_numpy(), preds["last_price"].to_numpy())
        for name in model_names
    }
    out = pd.DataFrame(table).T
    naive = out.loc["Naive (no change)", "MAE (₹/kg)"]
    out["MAE vs naive (%)"] = (out["MAE (₹/kg)"] / naive - 1) * 100
    return out


def run(
    prices: pd.Series,
    sentiment: pd.DataFrame | None = None,
    min_train: int = 60,
    refit_every: int = 5,
) -> Backtest:
    X, y = features.build(prices, sentiment)
    models = default_models(with_news=sentiment is not None)
    preds = walk_forward(X, y, prices, models, min_train=min_train, refit_every=refit_every)
    coverage = None
    if sentiment is not None:
        coverage = float((sentiment.reindex(preds.index)["news_count"] > 0).mean())
    return Backtest(preds, summarise(preds, list(models)), coverage)
