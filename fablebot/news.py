"""Financial news headlines for each instrument, via Yahoo Finance.

Soft context only: headlines appear in the morning briefing and evening
recap so you know what's driving the market. Strategies trade on price;
news never blocks or forces a trade. High-impact keywords are flagged so
important items are easy to spot in the report.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

HIGH_IMPACT_KEYWORDS = {
    "fed", "fomc", "rate", "hike", "cut", "inflation", "cpi", "pce",
    "nfp", "jobs", "gdp", "recession", "crisis", "crash", "war", "tariff",
    "sanction", "default", "earnings", "guidance", "miss", "beat",
}


def _flag(title: str) -> bool:
    lower = title.lower()
    return any(kw in lower for kw in HIGH_IMPACT_KEYWORDS)


def fetch_news(symbol: str, max_items: int = 5) -> list[dict]:
    """Return up to ``max_items`` recent headlines for ``symbol``.
    Each item: {title, publisher, age_hours, high_impact}."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception:
        return []

    now = datetime.now(timezone.utc).timestamp()
    results = []
    for item in raw[:max_items]:
        published = item.get("providerPublishTime", 0)
        age_hours = (now - published) / 3600.0 if published else float("inf")
        results.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "age_hours": round(age_hours, 1),
            "high_impact": _flag(item.get("title", "")),
        })
    return results


def format_news_block(headlines: list[dict], indent: str = "  ") -> str:
    if not headlines:
        return f"{indent}no recent headlines"
    lines = []
    for h in headlines:
        flag = " *** HIGH IMPACT ***" if h["high_impact"] else ""
        age = f"{h['age_hours']:.0f}h ago" if h["age_hours"] < 999 else "unknown age"
        lines.append(f"{indent}[{age}] {h['title']}{flag}")
    return "\n".join(lines)
