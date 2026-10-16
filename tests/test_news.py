import feedparser
import pandas as pd
import pytest

from metalmind import news
from metalmind.config import get_metal

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test</title>
<item><title>Copper prices surge on supply disruption in Chile</title>
  <link>https://example.com/a</link><description>Mine strike cuts output.</description>
  <pubDate>Sat, 08 Mar 2025 10:00:00 GMT</pubDate></item>
<item><title>Copper prices surge on supply disruption in Chile</title>
  <link>https://example.org/syndicated</link><description>Same story elsewhere.</description>
  <pubDate>Sat, 08 Mar 2025 11:00:00 GMT</pubDate></item>
<item><title>Company announces leadership change</title>
  <link>https://example.com/b</link><description>New CEO to lead growth.</description>
  <pubDate>Mon, 10 Mar 2025 09:00:00 GMT</pubDate></item>
<item><title>No date on this copper story</title><link>https://example.com/c</link></item>
</channel></rss>"""


@pytest.mark.parametrize(
    "metal,text,expected",
    [
        ("lead", "Company announces leadership change", False),
        ("lead", "Hindustan Zinc raises lead prices", True),
        ("lead", "Demand for lead-acid batteries rises", True),
        ("steel", "Steps to protect the environment", False),
        ("steel", "Iron ore imports fall", True),
        ("aluminium", "Aluminum smelter restarts", True),
        ("copper", "Copperplate handwriting contest", False),
    ],
)
def test_keyword_matching(metal, text, expected):
    assert get_metal(metal).matches(text) is expected


def test_parse_feed_filters_scores_and_dedupes():
    df = news.deduplicate(news.parse_feed(feedparser.parse(RSS), "copper", source="test"))
    assert len(df) == 1                       # duplicate headline and undated item removed
    assert df.loc[0, "sentiment"] != 0
    assert df.loc[0, "date_ist"] == pd.Timestamp("2025-03-08")


def test_weekend_news_counts_towards_next_trading_day():
    days = pd.bdate_range("2025-03-06", "2025-03-11")  # Thu .. Tue
    articles = pd.DataFrame({
        "date_ist": pd.to_datetime(["2025-03-08", "2025-03-09", "2025-03-11"]),  # Sat, Sun, Tue
        "sentiment": [0.5, -0.1, 0.3],
    })
    out = news.daily_sentiment(articles, days)
    assert out.loc["2025-03-10", "news_count"] == 2           # Monday collects the weekend
    assert out.loc["2025-03-10", "sentiment"] == pytest.approx(0.2)
    assert out.loc["2025-03-07", "news_count"] == 0           # nothing leaks backwards
    assert out.loc["2025-03-11", "news_count"] == 1


def test_collect_appends_to_archive(tmp_path, monkeypatch):
    parsed = news.parse_feed(feedparser.parse(RSS), "copper", source="test")
    monkeypatch.setattr(news, "fetch", lambda metal, feeds=None: news.deduplicate(parsed))
    archive, new = news.collect("copper", tmp_path)
    assert new == 1 and len(archive) == 1
    archive, new = news.collect("copper", tmp_path)          # same items again: nothing new
    assert new == 0 and len(news.load_archive("copper", tmp_path)) == 1


def test_backfill_queries_each_window_and_archives(tmp_path, monkeypatch):
    from datetime import date

    urls = []
    parsed = feedparser.parse(RSS)

    def fake_parse(url, agent=None):
        urls.append(url)
        return parsed

    monkeypatch.setattr(news.feedparser, "parse", fake_parse)
    archive, new = news.backfill("copper", date(2025, 3, 1), date(2025, 3, 22), window_days=7,
                                 pause=0, data_dir=tmp_path, progress=None)
    assert len(urls) == 3                                   # three 7-day windows
    assert "after%3A2025-03-01" in urls[0] and "before%3A2025-03-08" in urls[0]
    assert new == 1 and len(news.load_archive("copper", tmp_path)) == 1
