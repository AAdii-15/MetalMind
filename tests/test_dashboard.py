import numpy as np
import pandas as pd
import pytest

from metalmind import news

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_dashboard_renders_offline(tmp_path, monkeypatch):
    """The dashboard runs end to end from saved data (offline mode), with no network."""
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2021-01-04", periods=420)
    prices = pd.Series(900 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))), idx)
    (tmp_path / "prices").mkdir()
    prices.rename("price").rename_axis("date").to_frame().to_csv(tmp_path / "prices" / "copper_yahoo.csv")
    rows = [{"published_utc": pd.Timestamp(d, tz="UTC"), "date_ist": d, "title": t, "summary": "",
             "link": f"https://example.com/{i}", "source": "t", "sentiment": news.score(t)}
            for i, (d, t) in enumerate(zip(idx[::3], ["Heavy rain floods copper mine", "Copper demand rises"] * 200))]
    path = news.archive_path("copper", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=news.COLUMNS).to_csv(path, index=False)

    monkeypatch.setenv("METALMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("METALMIND_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("METALMIND_OFFLINE", "1")
    import importlib.util

    app = importlib.util.find_spec("metalmind").submodule_search_locations[0] + "/dashboard.py"
    at = streamlit_testing.AppTest.from_file(app, default_timeout=180).run()
    assert not at.exception, at.exception
    assert "Copper" in at.title[0].value
    labels = [m.label for m in at.metric]
    assert any("Forecast in 10 trading days" in label for label in labels)
    assert len(at.tabs) == 5
