import numpy as np
import pandas as pd
import pytest

from src.engine import extract_trades, run_backtest
from src.metrics import compute_metrics, summarize, trade_stats


def make_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D", name="date")
    return pd.Series(values, index=idx)


RETURNS = [0.0, 0.1, -0.1, 0.1]
EQUITY = [100, 110, 99, 108.9]


def test_total_return_and_max_drawdown():
    m = compute_metrics(make_series(RETURNS), make_series(EQUITY))
    assert m["total_return"] == pytest.approx(0.089)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)


def test_volatility_annualized():
    m = compute_metrics(make_series(RETURNS), make_series(EQUITY))
    expected = np.std([0.1, -0.1, 0.1], ddof=1) * np.sqrt(252)
    assert m["volatility"] == pytest.approx(expected)


def test_cagr_two_years():
    idx = pd.to_datetime(["2020-01-01", "2022-01-01"])
    equity = pd.Series([100.0, 121.0], index=idx)
    returns = pd.Series([0.0, 0.21], index=idx)
    m = compute_metrics(returns, equity)
    assert m["cagr"] == pytest.approx(0.10, abs=1e-3)


def test_sharpe_hand_computed():
    m = compute_metrics(make_series(RETURNS), make_series(EQUITY))
    # mean 0.03333 / std 0.11547 * sqrt(252)
    assert m["sharpe"] == pytest.approx(4.5826, rel=1e-3)


def test_flat_returns_give_zero_vol_and_nan_sharpe():
    m = compute_metrics(make_series([0.0] * 5), make_series([100.0] * 5))
    assert m["volatility"] == 0
    assert np.isnan(m["sharpe"])
    assert m["total_return"] == 0
    assert m["max_drawdown"] == 0


def test_trade_stats():
    trades = pd.DataFrame({"return_pct": [0.10, -0.05, 0.02, 0.03]})
    s = trade_stats(trades)
    assert s["num_trades"] == 4
    assert s["win_rate"] == pytest.approx(0.75)


def test_no_trades_stats():
    s = trade_stats(pd.DataFrame({"return_pct": []}))
    assert s["num_trades"] == 0
    assert np.isnan(s["win_rate"])


def test_summarize_matches_backtest():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="D", name="date")
    prices = pd.DataFrame(
        {"adj_close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))}, index=idx
    )
    bt = run_backtest(prices, 5, 20, fee_bps=5)
    trades = extract_trades(bt)
    out = summarize(bt, trades)

    strat_expected = bt["equity_strategy"].iloc[-1] / bt["equity_strategy"].iloc[0] - 1
    bh_expected = bt["equity_buyhold"].iloc[-1] / bt["equity_buyhold"].iloc[0] - 1
    assert out["strategy"]["total_return"] == pytest.approx(strat_expected)
    assert out["buy_and_hold"]["total_return"] == pytest.approx(bh_expected)
    assert out["strategy"]["num_trades"] == len(trades)
    assert out["buy_and_hold"]["num_trades"] == 1
    assert out["strategy"]["max_drawdown"] <= 0