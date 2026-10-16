"""News collection and sentiment scoring.

RSS feeds only expose their latest items (usually a day or two), so a single
run cannot recover months of news. `collect` therefore appends every relevant
headline to a local archive (data/news/<metal>.csv); run it daily and the
sentiment history builds up over time.
"""

from __future__ import annotations

import calendar
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from metalmind.config import DATA_DIR, feeds_for, get_metal, google_news_feed

log = logging.getLogger(__name__)

IST = "Asia/Kolkata"
COLUMNS = ["published_utc", "date_ist", "title", "summary", "link", "source", "sentiment"]

_analyzer = SentimentIntensityAnalyzer()
_TAG_RE = re.compile(r"<[^>]+>")


def score(text: str) -> float:
    """VADER compound sentiment in [-1, 1]."""
    return float(_analyzer.polarity_scores(text)["compound"])


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()


def _entry_time(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    # feedparser returns UTC struct_time; timegm keeps it in UTC
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def parse_feed(feed, metal: str, source: str = "") -> pd.DataFrame:
    """Turn a parsed feed into a DataFrame of relevant, scored articles."""
    spec = get_metal(metal)
    rows = []
    for entry in feed.entries:
        ts = _entry_time(entry)
        if ts is None:
            continue
        title = _clean_text(entry.get("title", ""))
        summary = _clean_text(entry.get("summary", ""))
        text = f"{title}. {summary}"
        if not spec.matches(text):
            continue
        rows.append({
            "published_utc": ts,
            "date_ist": pd.Timestamp(ts).tz_convert(IST).tz_localize(None).normalize(),
            "title": title,
            "summary": summary,
            "link": entry.get("link", ""),
            "source": source,
            "sentiment": score(text),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch(metal: str, feeds: list[str] | None = None) -> pd.DataFrame:
    """Download all feeds for a metal; a failing feed is logged, not fatal."""
    frames = []
    for url in feeds or feeds_for(metal):
        feed = feedparser.parse(url, agent="Mozilla/5.0 (MetalMind research tool)")
        if feed.get("bozo") and not feed.entries:
            log.warning("Feed unavailable: %s (%s)", url, feed.get("bozo_exception"))
            continue
        frames.append(parse_feed(feed, metal, source=url))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return deduplicate(pd.concat(frames, ignore_index=True))


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated stories (same link, or same headline syndicated across outlets)."""
    if df.empty:
        return df
    key = df["title"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    df = df.assign(_key=key).sort_values("published_utc")
    df = df.drop_duplicates("link", keep="first").drop_duplicates("_key", keep="first")
    return df.drop(columns="_key").reset_index(drop=True)


def _merge(old: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    parts = [df for df in (old, fresh) if not df.empty]
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return deduplicate(pd.concat(parts, ignore_index=True))


def archive_path(metal: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "news" / f"{get_metal(metal).name}.csv"


def load_archive(metal: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = archive_path(metal, data_dir)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    df["date_ist"] = pd.to_datetime(df["date_ist"])
    return df


def collect(metal: str, data_dir: Path = DATA_DIR, feeds: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Fetch fresh articles, merge them into the archive, return (archive, new_count)."""
    old = load_archive(metal, data_dir)
    fresh = fetch(metal, feeds)
    merged = _merge(old, fresh)
    path = archive_path(metal, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged, len(merged) - len(old)


def backfill(
    metal: str,
    since: date,
    until: date | None = None,
    window_days: int = 7,
    pause: float = 1.5,
    data_dir: Path = DATA_DIR,
    progress=print,
) -> tuple[pd.DataFrame, int]:
    """Rebuild historical headlines with date-bounded Google News searches.

    Each request asks for "<query> after:<start> before:<end>" for one window,
    so years of history can be collected week by week. Results are merged into
    the same archive used by `collect`.
    """
    spec = get_metal(metal)
    until = until or date.today()
    old = load_archive(metal, data_dir)
    frames, start, n_windows = [], since, 0
    while start < until:
        end = min(start + timedelta(days=window_days), until)
        query = f"{spec.news_query} after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}"
        feed = feedparser.parse(google_news_feed(query), agent="Mozilla/5.0 (MetalMind research tool)")
        if feed.get("bozo") and not feed.entries:
            log.warning("Window %s..%s failed: %s", start, end, feed.get("bozo_exception"))
        else:
            df = parse_feed(feed, metal, source="google-news-backfill")
            frames.append(df)
            n_windows += 1
            if progress and n_windows % 10 == 0:
                progress(f"  {metal}: up to {end:%d %b %Y}, {sum(len(f) for f in frames)} relevant articles so far")
        start = end
        time.sleep(pause)
    frames = [f for f in frames if not f.empty]
    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    merged = _merge(old, fresh)
    path = archive_path(metal, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged, len(merged) - len(old)


def daily_sentiment(articles: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate article sentiment onto trading days.

    Each article is assigned to the first trading day on or after its IST date,
    so weekend and holiday news counts towards the next session. Features for
    day t therefore only use news published up to day t.
    """
    out = pd.DataFrame(index=trading_days)
    out["sentiment"] = 0.0
    out["news_count"] = 0
    if articles.empty:
        return out
    days = pd.Series(trading_days.sort_values(), name="day")
    art = articles[["date_ist", "sentiment"]].sort_values("date_ist")
    art = art[(art["date_ist"] >= days.iloc[0]) & (art["date_ist"] <= days.iloc[-1])]
    if art.empty:
        return out
    mapped = pd.merge_asof(
        art, days.to_frame(), left_on="date_ist", right_on="day", direction="forward"
    ).dropna(subset=["day"])
    agg = mapped.groupby("day")["sentiment"].agg(["mean", "count"])
    out.loc[agg.index, "sentiment"] = agg["mean"].to_numpy()
    out.loc[agg.index, "news_count"] = agg["count"].to_numpy()
    return out
