import numpy as np
import pandas as pd

from metalmind import events, experiment, news


def test_event_tagging():
    assert events.tag("Heavy rainfall floods copper mines in Jharkhand") == ["weather"]
    assert set(events.tag("Strike halts output at Chile copper mine")) == {"supply"}
    assert "policy" in events.tag("India raises import duty on aluminium")
    assert events.tag("Company reports quarterly results") == []


def _synthetic_archive(days: pd.DatetimeIndex, rng) -> pd.DataFrame:
    titles = ["Heavy rain disrupts copper mine output", "Copper demand rises on construction boom",
              "Copper flat as traders await Fed interest rate decision", "Copper prices steady"]
    rows = []
    for d in days[::2]:
        t = titles[rng.integers(len(titles))]
        rows.append({"published_utc": pd.Timestamp(d, tz="UTC"), "date_ist": d, "title": t, "summary": "",
                     "link": f"https://example.com/{d:%Y%m%d}", "source": "test",
                     "sentiment": news.score(t)})
    return pd.DataFrame(rows, columns=news.COLUMNS)


def test_experiment_end_to_end(tmp_path, prices, price_csv):
    rng = np.random.default_rng(0)
    archive = _synthetic_archive(prices.index, rng)
    path = news.archive_path("copper", tmp_path / "data")
    path.parent.mkdir(parents=True)
    archive.to_csv(path, index=False)

    out = experiment.run(["copper"], horizons=(1, 5), min_train=60, refit_every=20,
                         data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                         prices_csv={"copper": str(price_csv)}, progress=lambda *_: None)
    for f in ("report.md", "summary.csv", "event_study.csv", "data_summary.csv",
              "feature_importance.csv", "skill_vs_naive.png", "backtest_copper.png"):
        assert (out / f).exists(), f
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["horizon (days)"]) == {1, 5}
    assert len(summary) == 2 * 11                # 2 baselines + 3 models x 3 feature sets, per horizon
    study = pd.read_csv(out / "event_study.csv")
    assert {"weather", "supply", "any news"} <= set(study["event"])
    assert "Forecast accuracy" in (out / "report.md").read_text()
