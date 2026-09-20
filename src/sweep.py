import numpy as np
import pandas as pd
from src.engine import rebase_backtest, run_backtest
from src.metrics import compute_metrics

DEFAULT_SHORTS = list(range(10, 101, 10))
DEFAULT_LONGS = list(range(100, 301, 25))


def run_sweep(
    prices: pd.DataFrame,
    short_windows=DEFAULT_SHORTS,
    long_windows=DEFAULT_LONGS,
    fee_bps: float = 5.0,
    initial_capital: float = 10_000.0,
    risk_free_rate: float = 0.0,
    eval_start=None,
) -> pd.DataFrame:
    """Sharpe ratio of the strategy for every (short, long) window pair.

    Rows are short windows, columns are long windows. Invalid pairs
    (short >= long) and pairs the data is too short for are left as NaN.
    If eval_start is given, the backtest runs on all of `prices` but Sharpe is
    scored only from eval_start onward (used for the out-of-sample grid).
    """
    grid = pd.DataFrame(
        np.nan,
        index=pd.Index(list(short_windows), name="short_window"),
        columns=pd.Index(list(long_windows), name="long_window"),
        dtype=float,
    )
    for short in grid.index:
        for long in grid.columns:
            if short >= long:
                continue
            try:
                bt = run_backtest(prices, short, long, fee_bps, initial_capital)
                if eval_start is not None:
                    bt = rebase_backtest(bt, eval_start, initial_capital)
            except ValueError:
                continue
            grid.loc[short, long] = compute_metrics(
                bt["strategy_return"], bt["equity_strategy"], risk_free_rate
            )["sharpe"]
    return grid

def best_params(grid: pd.DataFrame) -> tuple[int, int, float]:
    """Return (short_window, long_window, sharpe) for the highest Sharpe in the grid."""
    values = grid.to_numpy(dtype=float)
    if np.isnan(values).all():
        raise ValueError("Sweep grid has no valid results.")
    i, j = np.unravel_index(np.nanargmax(values), values.shape)
    return int(grid.index[i]), int(grid.columns[j]), float(values[i, j])

def train_test_analysis(
    prices: pd.DataFrame,
    split_date,
    short_windows=DEFAULT_SHORTS,
    long_windows=DEFAULT_LONGS,
    fee_bps: float = 5.0,
    initial_capital: float = 10_000.0,
    risk_free_rate: float = 0.0,
) -> dict:
    """Optimize windows on data before split_date, then evaluate them on data from split_date on."""
    split = pd.Timestamp(split_date)
    train = prices.loc[prices.index < split]
    n_test = int((prices.index >= split).sum())
    if len(train) == 0 or n_test < 2:
        raise ValueError(
            "split_date must fall inside the price data, with data on both sides."
        )

    # In-sample: the optimizer only ever sees the training data.
    train_grid = run_sweep(
        train, short_windows, long_windows, fee_bps, initial_capital, risk_free_rate
    )
    best_short, best_long, train_sharpe = best_params(train_grid)

    # Out-of-sample grid: every pair scored on the test period only.
    test_grid = run_sweep(
        prices,
        short_windows,
        long_windows,
        fee_bps,
        initial_capital,
        risk_free_rate,
        eval_start=split,
    )

    # The pair chosen on the training data, evaluated on the test period.
    bt = run_backtest(prices, best_short, best_long, fee_bps, initial_capital)
    test_bt = rebase_backtest(bt, split, initial_capital)
    test_strategy = compute_metrics(
        test_bt["strategy_return"], test_bt["equity_strategy"], risk_free_rate
    )
    test_buy_and_hold = compute_metrics(
        test_bt["market_return"], test_bt["equity_buyhold"], risk_free_rate
    )
    test_sharpe = test_strategy["sharpe"]

    valid = test_grid.to_numpy(dtype=float)
    valid = valid[~np.isnan(valid)]
    test_rank = None if np.isnan(test_sharpe) else int(1 + (valid > test_sharpe).sum())

    return {
        "best_short": best_short,
        "best_long": best_long,
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "test_rank": test_rank,
        "test_cells": int(len(valid)),
        "train_grid": train_grid,
        "test_grid": test_grid,
        "test_strategy": test_strategy,
        "test_buy_and_hold": test_buy_and_hold,
    }