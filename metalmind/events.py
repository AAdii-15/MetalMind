"""Rule-based event tagging of news headlines.

This is the systematic version of the 2025 prototype's idea ("heavy rain ->
mines shut -> price rises"): instead of a single hand-written rule, every
headline is tagged with the event types it mentions, daily counts become model
features, and an event study measures how prices actually behaved after each
event type.
"""

from __future__ import annotations

import re

EVENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "weather": (
        r"\brain(s|fall)?\b", r"\bmonsoon\b", r"\bflood(s|ing|ed)?\b", r"\bcyclone\b", r"\bstorm\b",
        r"\btyphoon\b", r"\blandslide\b", r"\bdrought\b", r"\bheat ?wave\b",
    ),
    "supply": (
        r"\bstrikes?\b", r"\bshut(down| down)?\b", r"\bhalt(s|ed)?\b", r"\bsuspend(s|ed)?\b", r"\boutage\b",
        r"\bclosure\b", r"\bforce majeure\b", r"\bdisrupt(ion|ed|s)?\b", r"\bcurtail(ment|s|ed)?\b",
        r"\b(output|production|supply) cuts?\b", r"\bshortage\b", r"\bdeficit\b", r"\bblockade\b",
        r"\baccident\b",
    ),
    "policy": (
        r"\btariffs?\b", r"\b(import|export|customs|anti-dumping|safeguard) dut(y|ies)\b", r"\bexport (ban|curbs?)\b",
        r"\bimport (ban|curbs?)\b", r"\bsanctions?\b", r"\bquotas?\b", r"\blevy\b", r"\banti-dumping\b",
        r"\bsubsid(y|ies)\b",
    ),
    "demand": (
        r"\bdemand\b", r"\bconsumption\b", r"\bconstruction\b", r"\binfrastructure\b", r"\belectric vehicles?\b",
        r"\bstimulus\b", r"\brecord imports\b",
    ),
    "macro": (
        r"\bfederal reserve\b", r"\bthe fed\b", r"\binterest rates?\b", r"\brate (cut|hike)s?\b", r"\binflation\b",
        r"\bgdp\b", r"\brecession\b", r"\bdollar\b", r"\brupee\b", r"\bpmi\b", r"\byuan\b",
    ),
    "inventory": (
        r"\binventor(y|ies)\b", r"\bstockpiles?\b", r"\bwarehouse\b", r"\b(lme|shfe|comex) stocks\b",
    ),
}

CATEGORIES = tuple(EVENT_PATTERNS)
_COMPILED = {cat: re.compile("|".join(pats)) for cat, pats in EVENT_PATTERNS.items()}


def tag(text: str) -> list[str]:
    """Event categories mentioned in a piece of text."""
    text = (text or "").lower()
    return [cat for cat, rx in _COMPILED.items() if rx.search(text)]
