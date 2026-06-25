"""Combined-overview renderer: one comparison page across several books."""

from fablebot import dashboard


def _book(name, realized, curve):
    return {
        "name": name,
        "account": dashboard.account_summary(100000.0, realized, {}, {}),
        "stats": dashboard.trade_stats([{"pnl": 1.0}] if realized > 0 else []),
        "equity_curve": curve,
    }


def test_render_overview_lists_every_book():
    books = [
        _book("FableBot — main", 500.0, [{"t": "a", "equity": 100000}, {"t": "b", "equity": 100500}]),
        _book("Experiment — 15m MR", -200.0, []),
    ]
    html = dashboard.render_overview(books, "2026-06-25 10:00")
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    assert "FableBot — all books" in html
    assert "FableBot — main" in html and "Experiment — 15m MR" in html
    assert "auto-refresh" in html


def test_render_overview_handles_no_books():
    html = dashboard.render_overview([], "now")
    assert "no books found" in html


def test_render_overview_links_books_with_href():
    """A book carrying an ``href`` renders its name as a clickable link to its own
    dashboard; a book without one stays plain text."""
    linked = _book("Adaptive", 100.0, [])
    linked["href"] = "../state_experiment_adaptive/dashboard.html"
    plain = _book("Fresh", 0.0, [])  # no href (e.g. never rendered its own page)
    html = dashboard.render_overview([linked, plain], "now")
    assert '<a href="../state_experiment_adaptive/dashboard.html"' in html
    assert ">Adaptive</a>" in html
    # The hrefless book is listed but not wrapped in a link.
    assert "Fresh" in html and ">Fresh</a>" not in html
