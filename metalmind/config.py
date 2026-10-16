"""Static configuration: supported metals, news keywords and feeds."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class MetalSpec:
    name: str
    bankbazaar_slug: str | None
    include: tuple[str, ...]          # regexes; any match marks an article relevant
    exclude: tuple[str, ...] = ()     # regexes; any match rejects the article
    news_query: str = ""              # search phrase for the Google News RSS feed
    yahoo_ticker: str | None = None   # international benchmark (USD) on Yahoo Finance
    kg_per_unit: float = 1.0          # kilograms in the ticker's price unit
    benchmark: str = ""               # human-readable benchmark name

    def matches(self, text: str) -> bool:
        text = text.lower()
        if any(re.search(p, text) for p in self.exclude):
            return False
        return any(re.search(p, text) for p in self.include)


# Word boundaries (\b) stop "lead" matching "leadership" and "iron" matching
# "environment". "lead" is only accepted in a metal context because the bare
# word is far more often a verb than the metal.
METALS: dict[str, MetalSpec] = {
    "copper": MetalSpec(
        "copper", "copper",
        include=(r"\bcopper\b",),
        news_query="copper price",
        yahoo_ticker="HG=F", kg_per_unit=0.45359237, benchmark="COMEX copper futures (USD/lb)",
    ),
    "aluminium": MetalSpec(
        "aluminium", "aluminium",
        include=(r"\balumini?um\b", r"\bbauxite\b", r"\balumina\b"),
        news_query="aluminium price",
        yahoo_ticker="ALI=F", kg_per_unit=1000.0, benchmark="COMEX aluminium futures (USD/tonne)",
    ),
    "steel": MetalSpec(
        "steel", "steel",
        include=(r"\bsteel\b", r"\biron ore\b", r"\bpig iron\b", r"\bsponge iron\b"),
        exclude=(r"\bstainless steel (bottle|utensil|cookware)s?\b",),
        news_query="steel price India",
        yahoo_ticker="HRC=F", kg_per_unit=907.18474, benchmark="US Midwest HRC steel futures (USD/short ton)",
    ),
    "iron_ore": MetalSpec(
        "iron_ore", None,
        include=(r"\biron[ -]ore\b", r"\bnmdc\b"),
        news_query="iron ore price",
        yahoo_ticker="TIO=F", kg_per_unit=1000.0, benchmark="SGX iron ore 62% Fe futures (USD/tonne)",
    ),
    "lead": MetalSpec(
        "lead", "lead",
        include=(r"\blead (price|prices|metal|ingots?|smelters?|ore|concentrate)\b",
                 r"\blead-acid\b", r"\bprices? of lead\b", r"\blead and zinc\b", r"\bzinc and lead\b"),
        news_query="lead metal price",
    ),
    "zinc": MetalSpec(
        "zinc", "zinc",
        include=(r"\bzinc\b",),
        news_query="zinc price",
    ),
    "nickel": MetalSpec(
        "nickel", "nickel",
        include=(r"\bnickel\b",),
        news_query="nickel price",
    ),
}

# General business / commodity feeds. Reuters retired its public RSS feeds in
# 2020, so it is no longer listed.
GENERAL_FEEDS: tuple[str, ...] = (
    "https://www.livemint.com/rss/industry",
    "https://economictimes.indiatimes.com/rssfeeds/13357270.cms",
)


def google_news_feed(query: str) -> str:
    """Topic-specific Google News RSS search feed (India edition)."""
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"


def feeds_for(metal: str) -> list[str]:
    spec = get_metal(metal)
    feeds = list(GENERAL_FEEDS)
    if spec.news_query:
        feeds.append(google_news_feed(spec.news_query))
    return feeds


FX_TICKER = "USDINR=X"


def get_metal(metal: str) -> MetalSpec:
    key = metal.strip().lower().replace("aluminum", "aluminium").replace(" ", "_").replace("-", "_")
    if key not in METALS:
        raise ValueError(f"Unsupported metal '{metal}'. Choose from: {', '.join(METALS)}")
    return METALS[key]
