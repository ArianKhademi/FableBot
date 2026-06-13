"""News parsing across yfinance schema versions.

yfinance changed its ``Ticker.news`` payload from a flat dict (``title``,
``providerPublishTime``) to a nested one (``content.title``,
``content.pubDate`` as ISO-8601, ``content.provider.displayName``). The parser
must handle both, or the report fills with blank ``[unknown age]`` lines.
"""

from datetime import datetime, timezone

from fablebot.news import _parse_item, fetch_news, format_news_block

NOW = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc).timestamp()


def test_new_nested_schema_is_parsed():
    item = {
        "id": "abc",
        "content": {
            "title": "Fed signals a rate cut",
            "pubDate": "2026-06-13T10:00:00Z",  # 2h before NOW
            "provider": {"displayName": "Reuters"},
        },
    }
    h = _parse_item(item, NOW)
    assert h["title"] == "Fed signals a rate cut"
    assert h["publisher"] == "Reuters"
    assert h["age_hours"] == 2.0
    assert h["high_impact"] is True  # "rate"/"cut"/"fed" keywords


def test_old_flat_schema_still_parsed():
    published = datetime(2026, 6, 13, 9, 0, tzinfo=timezone.utc).timestamp()  # 3h before
    item = {"title": "Gold steady ahead of data", "publisher": "Bloomberg",
            "providerPublishTime": int(published)}
    h = _parse_item(item, NOW)
    assert h["title"] == "Gold steady ahead of data"
    assert h["publisher"] == "Bloomberg"
    assert h["age_hours"] == 3.0
    assert h["high_impact"] is False


def test_missing_timestamp_is_unknown_age_not_zero():
    item = {"content": {"title": "Untimed headline"}}
    h = _parse_item(item, NOW)
    assert h["title"] == "Untimed headline"
    assert h["age_hours"] == float("inf")
    assert "unknown age" in format_news_block([h])


def test_blank_titles_are_dropped_by_fetch(monkeypatch):
    """An item with no title in either schema must not produce a blank line."""
    raw = [
        {"content": {"title": "", "pubDate": "2026-06-13T11:00:00Z"}},  # blank
        {"content": {"title": "Real news", "pubDate": "2026-06-13T11:00:00Z"}},
    ]

    class FakeTicker:
        def __init__(self, symbol): ...
        @property
        def news(self): return raw

    import fablebot.news as news_mod
    import sys, types
    fake_yf = types.SimpleNamespace(Ticker=FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    out = fetch_news("BTC-USD")
    assert len(out) == 1
    assert out[0]["title"] == "Real news"


def test_format_includes_publisher_and_flag():
    h = {"title": "CPI comes in hot", "publisher": "WSJ", "age_hours": 1.0, "high_impact": True}
    line = format_news_block([h])
    assert "CPI comes in hot" in line
    assert "(WSJ)" in line
    assert "HIGH IMPACT" in line
