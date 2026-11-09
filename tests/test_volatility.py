import numpy as np
import pandas as pd

from metalmind import news, volatility


def _clustered_prices(n=700, seed=3):
    """Synthetic prices with volatility clustering (a fixture, not market data)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    sig = np.empty(n); sig[0] = 0.01
    shocks = rng.normal(size=n)
    for t in range(1, n):
        sig[t] = np.sqrt(0.00001 + 0.12 * (sig[t - 1] * shocks[t - 1]) ** 2 + 0.85 * sig[t - 1] ** 2)
    rets = sig * shocks
    return pd.Series(800 * np.cumprod(1 + rets), index=idx), pd.Series(sig, index=idx)


def test_target_is_forward_realised_volatility():
    p, _ = _clustered_prices(200)
    X, y = volatility.build(p, None, horizon=5)
    t = X.index[50]
    r = p.pct_change() * 100
    i = p.index.get_loc(t)
    expected = np.sqrt((r.iloc[i + 1:i + 6] ** 2).mean())
    assert abs(y.loc[t] - expected) < 1e-9


def test_features_ignore_the_future():
    p, _ = _clustered_prices(300)
    X, _ = volatility.build(p, None)
    cut = X.index[100]
    p2 = p.copy(); p2[p2.index > cut] *= 1.5
    X2, _ = volatility.build(p2, None)
    pd.testing.assert_series_equal(X.loc[cut], X2.loc[cut])


def test_walk_forward_coverage_and_report(tmp_path):
    p, sig = _clustered_prices()
    csv = tmp_path / "copper.csv"
    p.rename("price").rename_axis("date").to_frame().to_csv(csv)
    # news volume that rises with true volatility, so a news model has something to find
    rng = np.random.default_rng(1)
    rows = []
    for d, s in sig.items():
        for _ in range(rng.poisson(1 + 200 * s)):
            rows.append({"published_utc": pd.Timestamp(d, tz="UTC"), "date_ist": d, "title": "Copper strike disrupts supply",
                         "summary": "", "link": f"x{len(rows)}", "source": "t", "sentiment": -0.3})
    path = news.archive_path("copper", tmp_path / "data")
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows, columns=news.COLUMNS).to_csv(path, index=False)
    out = volatility.run(["copper"], data_dir=tmp_path / "data", results_dir=tmp_path / "res",
                         prices_csv={"copper": str(csv)}, progress=lambda *_: None)
    for f in ("report.md", "vol_summary.csv", "band_coverage.csv", "band_coverage.png", "vol_copper.png",
              "news_volatility_link.csv"):
        assert (out / f).exists(), f
    cov = pd.read_csv(out / "band_coverage.csv", index_col=0)
    assert ((cov["Coverage (%)"] > 50) & (cov["Coverage (%)"] <= 100)).all()
    assert "Best model, calibrated multiplier" in cov.index
    s = pd.read_csv(out / "vol_summary.csv", index_col=0)
    assert len(s) == 2 + 6                       # two baselines + 3 models x 2 feature sets
