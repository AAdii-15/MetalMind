import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.Series:
    """Synthetic random-walk prices on business days (test fixture only, not market data)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2025-01-01", periods=160)
    rets = rng.normal(0.0005, 0.01, len(idx))
    s = pd.Series(800 * np.cumprod(1 + rets), index=idx, name="price")
    s.index.name = "date"
    return s


@pytest.fixture
def price_csv(tmp_path, prices):
    path = tmp_path / "copper.csv"
    prices.rename("Price").rename_axis("Date").to_frame().to_csv(path)
    return path
