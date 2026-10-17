"""Forecasting models. All models predict the forward return over the chosen horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class NaiveModel:
    """Random-walk baseline: the price stays where it is (predicted return 0)."""

    name = "Naive (no change)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class DriftModel:
    """Baseline that predicts the average return seen in training."""

    name = "Drift (mean return)"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DriftModel":
        self.mu_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu_)


def _estimator(kind: str, seed: int):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    if kind == "forest":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.6, random_state=seed, n_jobs=-1
        )
    if kind == "boosting":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=30,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"Unknown model kind: {kind}")


MODEL_LABELS = {"ridge": "Ridge regression", "forest": "Random Forest", "boosting": "Gradient Boosting"}


class FeatureModel:
    """A scikit-learn regressor restricted to a set of feature columns."""

    def __init__(self, kind: str, columns: list[str], name: str | None = None, seed: int = 42):
        self.kind = kind
        self.columns = list(columns)
        self.name = name or MODEL_LABELS[kind]
        self.model = _estimator(kind, seed)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureModel":
        self.model.fit(X[self.columns], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.columns])

    def importances(self) -> pd.Series:
        est = self.model
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        elif hasattr(est, "steps"):  # ridge pipeline: absolute standardised coefficients
            vals = np.abs(est.steps[-1][1].coef_)
        else:
            return pd.Series(dtype=float)
        return pd.Series(vals, index=self.columns).sort_values(ascending=False)


def ForestModel(columns: list[str], name: str = "Random Forest", seed: int = 42) -> FeatureModel:
    """Backwards-compatible constructor for the Random Forest model."""
    return FeatureModel("forest", columns, name, seed)
