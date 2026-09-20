import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest


def make_prices(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D", name="date")
    return pd.DataFrame({"adj_close": values}, index=idx)


def random_walk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return make_prices(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))


HAND_PRICES = [10, 9, 8, 7, 6, 7, 8, 9, 10, 11]


def test_position_is_signal_shifted_by_one():
    out = run_backtest(make_prices(HAND_PRICES), 2, 4, fee_bps=0)
    assert out["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert out["position"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    assert out["trade_flag"].sum() == 1


def test_strategy_return_and_costs_hand_computed():
    out = run_backtest(make_prices(HAND_PRICES), 2, 4, fee_bps=100)  # 1% fee
    # Flat until index 7, so no return and no cost before then.
    assert (out["strategy_return"].iloc[:7] == 0).all()
    # Index 7: market return 9/8 - 1 = 0.125, minus the 0.01 entry fee.
    assert out["strategy_return"].iloc[7] == pytest.approx(0.115)
    # Later days: no position change, so no fee.
    assert out["strategy_return"].iloc[8] == pytest.approx(10 / 9 - 1)
    assert out["strategy_return"].iloc[9] == pytest.approx(0.1)


def test_buy_and_hold_equity():
    out = run_backtest(make_prices(HAND_PRICES), 2, 4, initial_capital=1000)
    assert out["equity_buyhold"].iloc[0] == pytest.approx(1000)
    assert out["equity_buyhold"].iloc[-1] == pytest.approx(1000 * 11 / 10)


def test_no_lookahead_future_price_change():
    prices = random_walk()
    base = run_backtest(prices, 5, 20)

    altered_prices = prices.copy()
    altered_prices.iloc[-1, altered_prices.columns.get_loc("adj_close")] *= 3
    altered = run_backtest(altered_prices, 5, 20)

    # Changing only the final day's price must not change any earlier row,
    # and must not change the final day's position (it was decided the day before).
    pd.testing.assert_frame_equal(base.iloc[:-1], altered.iloc[:-1])
    assert base["position"].iloc[-1] == altered["position"].iloc[-1]


def test_higher_fee_lowers_final_equity():
    prices = random_walk()
    free = run_backtest(prices, 5, 20, fee_bps=0)
    costly = run_backtest(prices, 5, 20, fee_bps=100)
    assert free["trade_flag"].sum() > 0
    assert free["equity_strategy"].iloc[-1] > costly["equity_strategy"].iloc[-1]


def test_drawdown_and_starting_equity():
    out = run_backtest(random_walk(), 5, 20, initial_capital=5000)
    assert out["equity_strategy"].iloc[0] == pytest.approx(5000)
    assert out["drawdown"].iloc[0] == 0
    assert (out["drawdown"] <= 0).all()


def test_invalid_inputs_raise():
    prices = random_walk(50)
    with pytest.raises(ValueError):
        run_backtest(prices, 5, 100)  # not enough data for long window
    with pytest.raises(ValueError):
        run_backtest(random_walk(), 5, 20, fee_bps=-1)
    with pytest.raises(ValueError):
        run_backtest(random_walk(), 5, 20, initial_capital=0)