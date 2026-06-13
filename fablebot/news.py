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


def _parse_published(content: dict, item: dict) -> float:
    """Epoch seconds for a news item, across yfinance schema versions.

    Old schema: a flat integer ``providerPublishTime``. New schema (≈0.2.40+):
    an ISO-8601 string under ``content.pubDate`` / ``content.displayTime``.
    Returns 0.0 when no usable timestamp is present.
    """
    epoch = item.get("providerPublishTime") or content.get("providerPublishTime")
    if isinstance(epoch, (int, float)) and epoch > 0:
        return float(epoch)
    for key in ("pubDate", "displayTime"):
        val = content.get(key) or item.get(key)
        if isinstance(val, str) and val:
            try:
                # 3.9's fromisoformat rejects a trailing 'Z'; normalise it.
                return datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
    return 0.0


def _parse_item(item: dict, now: float) -> dict:
    """Normalise one raw yfinance news item to {title, publisher, age_hours,
    high_impact}, tolerating both the old flat and new nested payloads."""
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (content.get("title") or item.get("title") or "").strip()

    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "")
    else:
        publisher = item.get("publisher", "") or (provider if isinstance(provider, str) else "")

    published = _parse_published(content, item)
    age_hours = (now - published) / 3600.0 if published else float("inf")
    return {
        "title": title,
        "publisher": publisher,
        "age_hours": round(age_hours, 1),
        "high_impact": _flag(title),
    }


def fetch_news(symbol: str, max_items: int = 5) -> list[dict]:
    """Return up to ``max_items`` recent headlines for ``symbol``.
    Each item: {title, publisher, age_hours, high_impact}. Items without a
    usable title are dropped so the report never shows blank lines."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception:
        return []

    now = datetime.now(timezone.utc).timestamp()
    parsed = (_parse_item(item, now) for item in raw)
    return [h for h in parsed if h["title"]][:max_items]


def format_news_block(headlines: list[dict], indent: str = "  ") -> str:
    if not headlines:
        return f"{indent}no recent headlines"
    lines = []
    for h in headlines:
        flag = " *** HIGH IMPACT ***" if h["high_impact"] else ""
        age = f"{h['age_hours']:.0f}h ago" if h["age_hours"] < 999 else "unknown age"
        src = f" ({h['publisher']})" if h.get("publisher") else ""
        lines.append(f"{indent}[{age}] {h['title']}{src}{flag}")
    return "\n".join(lines)
