"""Recursive multi-day forecast with an empirical error band."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metalmind import features
from metalmind.models import FeatureModel


def _next_dates(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Continue the calendar the history uses: business days unless the history includes weekends."""
    trades_weekends = bool((index.dayofweek >= 5).any())
    freq = "D" if trades_weekends else "B"
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]


def forecast(
    prices: pd.Series,
    sentiment: pd.DataFrame | None = None,
    horizon: int = 10,
    residual_std: float | None = None,
    kind: str = "ridge",
) -> pd.DataFrame:
    """Forecast `horizon` future prices.

    The model (Ridge regression by default, the most accurate learned model in
    the walk-forward experiment) is trained on all available history. Each step feeds the
    previous prediction back in as the latest price. News for future days is
    unknown, so the most recent day's sentiment is carried forward.

    `residual_std` is the standard deviation of next-day return errors from a
    walk-forward backtest; it gives an approximate 95% band that widens with
    the square root of the horizon.
    """
    cols = features.PRICE_FEATURES + (features.NEWS_FEATURES if sentiment is not None else [])
    X, y = features.build(prices, sentiment)
    model = FeatureModel(kind, cols).fit(X, y)

    future = _next_dates(prices.index, horizon)
    path = prices.astype(float).copy()
    sent = None
    if sentiment is not None:
        sent = sentiment.reindex(prices.index).fillna(0)
        last = sent.iloc[[-1]]
        sent = pd.concat([sent, pd.DataFrame(np.repeat(last.to_numpy(), horizon, axis=0), index=future, columns=sent.columns)])

    rows = []
    for k, day in enumerate(future, start=1):
        row = features.latest_row(path, sent.loc[: path.index[-1]] if sent is not None else None)
        ret = float(model.predict(row)[0])
        price = float(path.iloc[-1]) * (1 + ret)
        path.loc[day] = price
        band = 1.96 * residual_std * np.sqrt(k) if residual_std else np.nan
        rows.append({"date": day, "forecast": price, "lower": price * (1 - band), "upper": price * (1 + band)})
    out = pd.DataFrame(rows).set_index("date")
    return out
