from fablebot.analytics import analyze, render_digest
from fablebot.journal import Journal
from tests.test_journal import make_trade


def journal_with(trades=(), shadow_trades=(), decisions=()):
    journal = Journal()
    for action in decisions:
        journal.record("decision", instrument="X", action=action)
    for t in trades:
        journal.record_trade(t)
    for t in shadow_trades:
        journal.record_trade(t, shadow=True)
    return journal


def test_overall_and_grouped_stats():
    journal = journal_with(
        trades=[
            make_trade(pnl=500.0),
            make_trade(pnl=-1000.0, instrument="SPX", strategy="mean_reversion"),
            make_trade(pnl=300.0, instrument="SPX", strategy="mean_reversion"),
        ]
    )
    digest = analyze(journal.read())
    assert digest.overall.count == 3
    assert digest.overall.wins == 2
    assert digest.overall.total_pnl == -200.0
    assert digest.by_instrument["SPX"].count == 2
    assert digest.by_strategy["momentum_breakout"].total_pnl == 500.0


def test_near_miss_stop_detection():
    near_miss = make_trade(
        pnl=-1000.0,
        close_reason="hard stop hit at 61,138.45",
        max_favorable_pnl=600.0,  # was up 60% of the 1000 risk before dying
        risk_amount=1000.0,
    )
    clean_stop = make_trade(
        pnl=-1000.0,
        instrument="SPX",
        close_reason="hard stop hit at 7,224.53",
        max_favorable_pnl=100.0,  # never really worked
        risk_amount=1000.0,
    )
    digest = analyze(journal_with(trades=[near_miss, clean_stop]).read())
    assert digest.stop_outs == 2
    assert len(digest.near_miss_stop_outs) == 1
    assert digest.near_miss_stop_outs[0]["instrument"] == "BTC"


def test_reentry_after_stop_detection():
    stop = make_trade(
        pnl=-1000.0,
        close_reason="hard stop hit at 61,138.45",
        closed_at="2026-06-10T17:59:00+00:00",
    )
    reentry = make_trade(
        pnl=-400.0,
        opened_at="2026-06-11T02:22:00+00:00",  # 8.4h later — the 2am re-entry
        closed_at="2026-06-11T09:00:00+00:00",
    )
    unrelated = make_trade(
        instrument="GOLD",
        opened_at="2026-06-11T02:00:00+00:00",
        closed_at="2026-06-11T09:00:00+00:00",
    )
    digest = analyze(journal_with(trades=[stop, reentry, unrelated]).read())
    assert len(digest.reentries) == 1
    assert digest.reentries[0]["instrument"] == "BTC"
    assert 8.0 < digest.reentries[0]["hours_after_stop"] < 9.0


def test_shadow_counterfactual_kept_separate():
    journal = journal_with(
        trades=[make_trade(pnl=500.0)],
        shadow_trades=[make_trade(pnl=350.0, instrument="NDX")],
        decisions=["hold", "blocked"],
    )
    digest = analyze(journal.read())
    assert digest.overall.count == 1  # shadow trades never pollute real stats
    assert digest.shadow.count == 1
    assert digest.shadow.total_pnl == 350.0
    assert digest.blocked == 1
    assert digest.decisions == 2


def test_digest_renders_with_small_sample_warnings():
    journal = journal_with(trades=[make_trade(pnl=500.0)])
    text = render_digest(analyze(journal.read()))
    assert "SMALL SAMPLE" in text
    assert "## By strategy" in text
    assert "## Correlation filter counterfactual" in text


def test_empty_journal_renders():
    text = render_digest(analyze([]))
    assert "0 trades" in text
