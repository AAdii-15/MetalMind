import numpy as np
import pandas as pd
import pytest

from metalmind import evaluate, features, prices as prices_mod
from metalmind.cli import main
from metalmind.forecast import forecast


def test_target_is_next_day_return(prices):
    X, y = features.build(prices)
    day = X.index[20]
    nxt = prices.index[prices.index.get_loc(day) + 1]
    assert y.loc[day] == pytest.approx(prices.loc[nxt] / prices.loc[day] - 1)


def test_features_do_not_use_the_future(prices):
    X, _ = features.build(prices)
    cut = X.index[50]
    tampered = prices.copy()
    tampered.loc[tampered.index > cut] *= 3          # change every price after the cut
    X2, _ = features.build(tampered)
    pd.testing.assert_series_equal(X.loc[cut], X2.loc[cut])


def test_walk_forward_trains_only_on_the_past(prices):
    X, y = features.build(prices)
    seen = []

    class Spy:
        def fit(self, Xt, yt):
            seen.append(Xt.index.max())
            return self

        def predict(self, Xp):
            assert (Xp.index > seen[-1]).all()       # every predicted day is after the training data
            return np.zeros(len(Xp))

    evaluate.walk_forward(X, y, prices, {"spy": Spy}, min_train=40, refit_every=7)
    assert seen


def test_multi_day_horizon_leaves_a_gap(prices):
    """With a 5-day target, training rows must end 5 days before the prediction day."""
    h = 5
    X, y = features.build(prices, horizon=h)
    gaps = []

    class Spy:
        def fit(self, Xt, yt):
            self.end = Xt.index.max()
            return self

        def predict(self, Xp):
            pos = X.index.get_loc
            gaps.append(pos(Xp.index.min()) - pos(self.end))
            return np.zeros(len(Xp))

    evaluate.walk_forward(X, y, prices, {"spy": Spy}, min_train=40, refit_every=10, horizon=h)
    assert min(gaps) == h                    # target of the last training row is known on the first test day


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 2, 500)
    better = rng.normal(0, 1, 500)
    stat, p = evaluate.diebold_mariano(better, base)
    assert stat < 0 and p < 0.01
    stat, p = evaluate.diebold_mariano(base, base + rng.normal(0, 1e-9, 500))
    assert p > 0.05


def test_naive_metrics(prices):
    bt = evaluate.run(prices, min_train=60)
    p = bt.predictions
    expected = np.mean(np.abs(p["actual"] - p["last_price"]))
    assert bt.summary.loc["Naive (no change)", "MAE (₹/kg)"] == pytest.approx(expected)
    assert {"Random Forest: price only", "Drift (mean return)"} <= set(bt.summary.index)


def test_forecast_shape_and_band(prices):
    fc = forecast(prices, horizon=5, residual_std=0.01)
    assert len(fc) == 5
    assert (fc.index.dayofweek < 5).all()            # business-day history -> business-day forecast
    assert (fc["lower"] < fc["forecast"]).all() and (fc["forecast"] < fc["upper"]).all()
    widths = (fc["upper"] - fc["lower"]) / fc["forecast"]
    assert widths.is_monotonic_increasing


def test_bankbazaar_parser_and_csv_loader(tmp_path):
    html = """<table><tr><th>Date</th><th>Price</th></tr>
              <tr><td>14/07/2025</td><td>₹ 881.50</td></tr>
              <tr><td>15/07/2025</td><td>₹ 882.40</td></tr></table>"""
    s = prices_mod.parse_bankbazaar(html)
    assert list(s.values) == [881.5, 882.4] and s.index[0] == pd.Timestamp("2025-07-14")

    csv = tmp_path / "p.csv"
    csv.write_text("Date,Close\n2025-07-15,10\n2025-07-14,9\n2025-07-15,11\n")
    s = prices_mod.load_csv(csv)
    assert list(s.values) == [9.0, 11.0]              # sorted, duplicate date keeps the last value


def test_no_synthetic_fallback(tmp_path, monkeypatch):
    def fail(metal, timeout=20):
        raise prices_mod.PriceDataError("offline")

    monkeypatch.setattr(prices_mod, "fetch_bankbazaar", fail)
    with pytest.raises(prices_mod.PriceDataError, match="--prices-csv"):
        prices_mod.get_price_history("copper", data_dir=tmp_path, source="bankbazaar")


def test_cli_end_to_end(tmp_path, price_csv, capsys):
    common = ["--data-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out")]
    assert main(common + ["evaluate", "--metal", "copper", "--prices-csv", str(price_csv), "--no-news"]) == 0
    assert main(common + ["forecast", "--metal", "copper", "--prices-csv", str(price_csv), "--no-news",
                          "--horizon", "7"]) == 0
    text = capsys.readouterr().out
    assert "Walk-forward backtest" in text and "7-day forecast" in text
    assert len(list((tmp_path / "out").glob("*.png"))) == 2
