"""Per-book dashboard heading: each config carries a name and render_html uses it
as the title, so multiple bots (main, experiment, trailing) are distinguishable."""

from pathlib import Path

from fablebot import dashboard
from fablebot.config import load_config

ROOT = Path(__file__).resolve().parent.parent
EMPTY_ENGINE = {"genomes": [], "numeric_proposals": {}, "genome_qualified": 0, "last_runs": []}


def _snap():
    return {
        "account": dashboard.account_summary(100000.0, 0.0, {}, {}),
        "stats": dashboard.trade_stats([]), "equity_curve": [], "by_strategy": [],
        "by_instrument": [], "positions": [], "recent_trades": [],
    }


def test_render_html_uses_the_book_title():
    html = dashboard.render_html(_snap(), EMPTY_ENGINE, "now", title="Experiment — 15m MR")
    assert "<h1>Experiment — 15m MR</h1>" in html
    assert "<title>Experiment — 15m MR dashboard</title>" in html


def test_render_html_defaults_to_fablebot():
    assert "<h1>FableBot</h1>" in dashboard.render_html(_snap(), EMPTY_ENGINE, "now")


def test_configs_carry_distinct_names_and_trailing_variant():
    assert load_config(ROOT / "config.yaml").name == "FableBot — main"
    assert load_config(ROOT / "config_experiment.yaml").name == "Experiment — 15m mean-reversion"
    trail = load_config(ROOT / "config_experiment_trail.yaml")
    assert trail.name == "Experiment — 15m MR + trailing stop"
    # the only difference from the no-trail experiment: a trailing stop on each
    assert all(i.params.get("trail_atr_mult") for i in trail.instruments.values())
