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


def _fetch_with_retry(url: str, tries: int = 4, pause: float = 1.5):
    """Fetch a feed, backing off when Google rate-limits (empty/failed responses)."""
    feed = None
    for attempt in range(tries):
        feed = feedparser.parse(url, agent="Mozilla/5.0 (MetalMind research tool)")
        status = feed.get("status", 200)
        if feed.entries or (status == 200 and not feed.get("bozo")):
            return feed
        time.sleep(pause * (2 ** (attempt + 1)))
    return feed


def _progress_path(metal: str, data_dir: Path) -> Path:
    return data_dir / "news" / f"{get_metal(metal).name}_backfill_done.txt"


def backfill(
    metal: str,
    since: date,
    until: date | None = None,
    window_days: int = 7,
    pause: float = 1.0,
    data_dir: Path = DATA_DIR,
    progress=print,
) -> tuple[pd.DataFrame, int]:
    """Rebuild historical headlines with date-bounded Google News searches.

    For each window and each query, ask Google News for
    "<query> after:<start> before:<end>". Completed windows are recorded, so
    an interrupted run resumes where it stopped. Results are merged into the
    same archive used by `collect`, flushing every 20 windows.
    """
    spec = get_metal(metal)
    queries = [q for q in (spec.news_query, *spec.backfill_queries) if q]
    until = until or date.today()
    done_path = _progress_path(metal, data_dir)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    start_len = len(load_archive(metal, data_dir))

    pending, start = [], since
    while start < until:
        end = min(start + timedelta(days=window_days), until)
        for qi, q in enumerate(queries):
            key = f"{qi}:{start:%Y-%m-%d}:{end:%Y-%m-%d}"
            if key not in done:
                pending.append((key, q, start, end))
        start = end
    if progress:
        progress(f"  {metal}: {len(pending)} searches to run ({len(done)} already done)")

    buffer, failures = [], 0
    for n, (key, q, s0, s1) in enumerate(pending, start=1):
        feed = _fetch_with_retry(google_news_feed(f"{q} after:{s0:%Y-%m-%d} before:{s1:%Y-%m-%d}"))
        if feed is None or (feed.get("bozo") and not feed.entries):
            failures += 1
            log.warning("Search failed (%s): %s", key, feed.get("bozo_exception") if feed else "no response")
        else:
            buffer.append(parse_feed(feed, metal, source="google-news-backfill"))
            done.add(key)
        if n % 20 == 0 or n == len(pending):
            _flush(metal, buffer, data_dir)
            buffer = []
            done_path.write_text("\n".join(sorted(done)))
            if progress:
                total = len(load_archive(metal, data_dir))
                progress(f"  {metal}: {n}/{len(pending)} searches, reached {s1:%d %b %Y}, "
                         f"{total} articles in archive, {failures} failed")
        time.sleep(pause)
    archive = load_archive(metal, data_dir)
    return archive, len(archive) - start_len


def _flush(metal: str, frames: list[pd.DataFrame], data_dir: Path) -> None:
    fresh = [f for f in frames if not f.empty]
    if not fresh:
        return
    merged = _merge(load_archive(metal, data_dir), pd.concat(fresh, ignore_index=True))
    path = archive_path(metal, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)


def daily_signals(articles: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate articles onto trading days: mean sentiment, article count and event counts.

    Each article is assigned to the first trading day on or after its IST date,
    so weekend and holiday news counts towards the next session. Signals for
    day t therefore only use news published up to day t.
    """
    from metalmind.events import CATEGORIES, tag

    out = pd.DataFrame(index=trading_days)
    out["sentiment"] = 0.0
    out["news_count"] = 0
    for cat in CATEGORIES:
        out[f"ev_{cat}"] = 0
    if articles.empty:
        return out
    days = pd.Series(trading_days.sort_values(), name="day")
    art = articles.copy()
    art = art[(art["date_ist"] >= days.iloc[0]) & (art["date_ist"] <= days.iloc[-1])].sort_values("date_ist")
    if art.empty:
        return out
    text = art.get("title", pd.Series("", index=art.index)).fillna("") + ". " + \
        art.get("summary", pd.Series("", index=art.index)).fillna("")
    tags = text.map(tag)
    for cat in CATEGORIES:
        art[f"ev_{cat}"] = tags.map(lambda t, c=cat: int(c in t))
    mapped = pd.merge_asof(
        art, days.to_frame(), left_on="date_ist", right_on="day", direction="forward"
    ).dropna(subset=["day"])
    g = mapped.groupby("day")
    out.loc[g.size().index, "sentiment"] = g["sentiment"].mean().to_numpy()
    out.loc[g.size().index, "news_count"] = g.size().to_numpy()
    for cat in CATEGORIES:
        out.loc[g.size().index, f"ev_{cat}"] = g[f"ev_{cat}"].sum().to_numpy()
    return out


def daily_sentiment(articles: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily signals including mean sentiment and article count (alias kept for clarity)."""
    return daily_signals(articles, trading_days)
