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

from metalmind.config import DATA_DIR, get_metal

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
    url = BANKBAZAAR_URL.format(slug=get_metal(metal).bankbazaar_slug)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise PriceDataError(f"Could not download {url}: {exc}") from exc
    return parse_bankbazaar(resp.text)


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


def get_price_history(metal: str, csv: str | Path | None = None, data_dir: Path = DATA_DIR) -> tuple[pd.Series, str]:
    """Return (price series, source description).

    Priority: an explicit CSV, otherwise a fresh BankBazaar scrape merged into
    the local archive, otherwise the archive alone.
    """
    if csv:
        return load_csv(csv), f"CSV file {csv}"
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
