"""Forecasting models. All models predict the next-day return."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class NaiveModel:
    """Random-walk baseline: tomorrow's price equals today's (predicted return 0)."""

    name = "Naive (no change)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class DriftModel:
    """Baseline that predicts the average daily return seen in training."""

    name = "Drift (mean return)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DriftModel":
        self.mu_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu_)


class ForestModel:
    """Random Forest on the selected feature columns."""

    def __init__(self, columns: list[str], name: str, seed: int = 42):
        self.columns = list(columns)
        self.name = name
        self.model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.6, random_state=seed, n_jobs=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForestModel":
        self.model.fit(X[self.columns], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.columns])

    def importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.columns).sort_values(ascending=False)
