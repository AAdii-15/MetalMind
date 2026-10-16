"""Leakage-free feature engineering.

The model predicts the next-day *return* rather than the next price level.
Tree models cannot extrapolate beyond the target values seen in training, so a
model trained on price levels can only ever predict averages of past prices;
returns avoid that and are comparable across metals.

Every feature in row t is computed from data available at the close of day t.
The target in row t is the return from day t to day t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["ret_1", "ret_2", "ret_3", "ret_5", "vol_5", "mom_5", "mom_10"]
NEWS_FEATURES = ["sentiment", "sentiment_3d", "log_news_count"]


def price_features(prices: pd.Series) -> pd.DataFrame:
    p = prices.astype(float)
    r = p.pct_change()
    f = pd.DataFrame(index=p.index)
    f["ret_1"] = r
    f["ret_2"] = r.shift(1)
    f["ret_3"] = r.shift(2)
    f["ret_5"] = p.pct_change(5)
    f["vol_5"] = r.rolling(5).std()
    f["mom_5"] = p / p.rolling(5).mean() - 1
    f["mom_10"] = p / p.rolling(10).mean() - 1
    return f


def news_features(sentiment: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=sentiment.index)
    f["sentiment"] = sentiment["sentiment"].astype(float)
    f["sentiment_3d"] = f["sentiment"].rolling(3, min_periods=1).mean()
    f["log_news_count"] = np.log1p(sentiment["news_count"].astype(float))
    return f


def build(prices: pd.Series, sentiment: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y is the next-day return. The last row (unknown target) is dropped."""
    X = price_features(prices)
    if sentiment is not None:
        X = X.join(news_features(sentiment.reindex(prices.index).fillna(0)))
    y = prices.shift(-1) / prices - 1
    y.name = "target_ret"
    data = X.join(y).dropna()
    return data[X.columns], data["target_ret"]


def latest_row(prices: pd.Series, sentiment: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature row for the most recent day (used to forecast the next day)."""
    X = price_features(prices)
    if sentiment is not None:
        X = X.join(news_features(sentiment.reindex(prices.index).fillna(0)))
    return X.iloc[[-1]]
