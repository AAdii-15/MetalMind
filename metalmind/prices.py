"""Daily price history: load from CSV or scrape BankBazaar, and keep a local archive.

There is deliberately no synthetic fallback. If no real prices are available
the pipeline stops with a clear error instead of training on random numbers.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from metalmind.config import DATA_DIR, FX_TICKER, get_metal

log = logging.getLogger(__name__)

BANKBAZAAR_URL = "https://www.bankbazaar.com/commodity-price/{slug}.html"
_HEADERS = {"User-Agent": "Mozilla/5.0 (MetalMind research tool)"}


class PriceDataError(RuntimeError):
    """Raised when no usable price history can be obtained."""


def _parse_dates(col: pd.Series) -> pd.Series:
    """ISO dates (2025-07-15) as written; anything else (15/07/2025, 15 Jul 2025) day-first, as in India."""
    text = col.astype(str).str.strip()
    if text.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.to_datetime(text.str[:10], format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _clean(df: pd.DataFrame) -> pd.Series:
    """Normalise a raw table into a sorted, de-duplicated Series of float prices indexed by date."""
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)
    date_col = next((c for c in df.columns if "date" in c), None)
    price_col = next((c for c in df.columns if c in ("price", "close", "rate") or "price" in c), None)
    if date_col is None or price_col is None:
        raise PriceDataError(f"Expected a date and a price column, found {list(df.columns)}")

    prices = pd.to_numeric(df[price_col].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    dates = _parse_dates(df[date_col])
    s = pd.Series(prices.to_numpy(), index=pd.DatetimeIndex(dates).normalize(), name="price")
    s = s[s.index.notna() & s.notna() & (s > 0)]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index.name = "date"
    return s


def load_csv(path: str | Path) -> pd.Series:
    """Load a CSV with a date column and a price/close column (e.g. an MCX or LME export)."""
    path = Path(path)
    if not path.exists():
        raise PriceDataError(f"Price file not found: {path}")
    return _clean(pd.read_csv(path))


def parse_bankbazaar(html: str) -> pd.Series:
    """Extract the historical price table from a BankBazaar commodity page."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError as exc:  # no <table> on the page
        raise PriceDataError("No price table found on the BankBazaar page") from exc
    for table in tables:
        try:
            s = _clean(table)
        except (PriceDataError, ValueError):
            continue
        if len(s) >= 2:
            return s
    raise PriceDataError("BankBazaar page has no table with Date and Price columns")


def fetch_bankbazaar(metal: str, timeout: float = 20) -> pd.Series:
    """Scrape the recent daily price table (INR per kg) for a metal from BankBazaar."""
    slug = get_metal(metal).bankbazaar_slug
    if not slug:
        raise PriceDataError(f"BankBazaar has no page for {metal}")
    url = BANKBAZAAR_URL.format(slug=slug)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise PriceDataError(f"Could not download {url}: {exc}") from exc
    return parse_bankbazaar(resp.text)


def _yahoo_close(ticker: str, start: str) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise PriceDataError("yfinance is not installed: pip install yfinance") from exc
    try:
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    except Exception as exc:  # yfinance raises many network/parse error types
        raise PriceDataError(f"Yahoo Finance download failed for {ticker}: {exc}") from exc
    if hist is None or hist.empty or "Close" not in hist:
        raise PriceDataError(f"Yahoo Finance returned no data for {ticker}")
    close = hist["Close"].astype(float)
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[close > 0]
    return close[~close.index.duplicated(keep="last")].sort_index()


def fetch_yahoo(metal: str, start: str = "2020-01-01") -> pd.Series:
    """International benchmark futures converted to INR per kg using the daily USD/INR rate.

    Note: this is the global benchmark in rupees, not the Indian domestic
    (MCX / dealer) price, which adds import duty, freight and local premiums.
    """
    spec = get_metal(metal)
    if not spec.yahoo_ticker:
        raise PriceDataError(f"No Yahoo Finance benchmark configured for {metal}")
    usd = _yahoo_close(spec.yahoo_ticker, start)
    fx = _yahoo_close(FX_TICKER, start)
    fx = fx.reindex(usd.index.union(fx.index)).sort_index().ffill().reindex(usd.index)
    inr_per_kg = (usd * fx / spec.kg_per_unit).dropna()
    inr_per_kg.name = "price"
    inr_per_kg.index.name = "date"
    if len(inr_per_kg) < 2:
        raise PriceDataError(f"Not enough Yahoo Finance data for {metal}")
    return inr_per_kg


def archive_path(metal: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "prices" / f"{get_metal(metal).name}.csv"


def update_archive(metal: str, new: pd.Series, data_dir: Path = DATA_DIR) -> pd.Series:
    """Merge newly scraped prices into the local archive so history grows run by run."""
    path = archive_path(metal, data_dir)
    if path.exists():
        old = load_csv(path)
        merged = pd.concat([old, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new.sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.rename("price").to_frame().to_csv(path, date_format="%Y-%m-%d")
    return merged


def get_price_history(
    metal: str,
    csv: str | Path | None = None,
    data_dir: Path = DATA_DIR,
    source: str = "yahoo",
    start: str = "2020-01-01",
) -> tuple[pd.Series, str]:
    """Return (price series in INR/kg, source description).

    An explicit CSV always wins. Otherwise `source` selects:
      yahoo      - international benchmark futures x USD/INR (long history, default)
      bankbazaar - Indian retail rates scraped from BankBazaar (short recent window),
                   merged into a local archive that grows with each run
    """
    if csv:
        return load_csv(csv), f"CSV file {csv}"
    if source == "cache":  # offline mode: the last successful Yahoo download
        cache = data_dir / "prices" / f"{get_metal(metal).name}_yahoo.csv"
        if not cache.exists():
            raise PriceDataError(f"No saved prices at {cache}. Run once online first.")
        series = load_csv(cache)
        return series[series.index >= pd.Timestamp(start)], f"saved data (offline), last day {series.index[-1]:%d %b %Y}"
    if source == "yahoo":
        spec = get_metal(metal)
        cache = data_dir / "prices" / f"{spec.name}_yahoo.csv"
        try:
            series = fetch_yahoo(metal, start)
        except PriceDataError as exc:
            if cache.exists():  # offline: fall back to the last successful download
                series = load_csv(cache)
                series = series[series.index >= pd.Timestamp(start)]
                log.warning("%s; using cached prices from %s", exc, cache)
                return series, f"cached Yahoo Finance data (last day {series.index[-1]:%d %b %Y}, offline)"
            raise
        cache.parent.mkdir(parents=True, exist_ok=True)
        series.rename("price").to_frame().to_csv(cache, date_format="%Y-%m-%d")
        return series, f"Yahoo Finance: {spec.benchmark} x USD/INR"
    if source != "bankbazaar":
        raise ValueError(f"Unknown price source '{source}' (use yahoo, cache or bankbazaar)")
    try:
        scraped = fetch_bankbazaar(metal)
        merged = update_archive(metal, scraped, data_dir)
        return merged, f"BankBazaar (archive: {len(merged)} days)"
    except PriceDataError as exc:
        path = archive_path(metal, data_dir)
        if path.exists():
            log.warning("%s; using local archive %s", exc, path)
            return load_csv(path), f"local archive {path}"
        raise PriceDataError(
            f"{exc}. No local archive either. Pass --prices-csv with a daily price history."
        ) from exc
