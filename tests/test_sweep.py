import numpy as np
import pandas as pd
import pytest

from src.engine import run_backtest
from src.metrics import compute_metrics
from src.sweep import best_params, run_sweep


def random_walk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame(
        {"adj_close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n))}, index=idx
    )


def test_grid_shape_and_labels():
    grid = run_sweep(random_walk(), [5, 10], [20, 50, 100])
    assert grid.shape == (2, 3)
    assert grid.index.name == "short_window"
    assert grid.columns.name == "long_window"
    assert grid.index.tolist() == [5, 10]
    assert grid.columns.tolist() == [20, 50, 100]


def test_invalid_combinations_are_nan():
    grid = run_sweep(random_walk(), [10, 50], [20, 50])
    assert np.isnan(grid.loc[50, 20])
    assert np.isnan(grid.loc[50, 50])
    assert not np.isnan(grid.loc[10, 20])
    assert not np.isnan(grid.loc[10, 50])


def test_cell_matches_direct_backtest():
    prices = random_walk()
    grid = run_sweep(prices, [5], [20], fee_bps=5)
    bt = run_backtest(prices, 5, 20, fee_bps=5)
    expected = compute_metrics(bt["strategy_return"], bt["equity_strategy"])["sharpe"]
    assert grid.loc[5, 20] == pytest.approx(expected)


def test_insufficient_data_gives_nan():
    grid = run_sweep(random_walk(150), [5], [20, 200])
    assert not np.isnan(grid.loc[5, 20])
    assert np.isnan(grid.loc[5, 200])


def test_best_params_picks_max():
    grid = pd.DataFrame(
        [[0.1, 0.5, np.nan], [0.9, 0.2, 0.3]],
        index=pd.Index([10, 20], name="short_window"),
        columns=pd.Index([50, 100, 200], name="long_window"),
    )
    short, long, sharpe = best_params(grid)
    assert (short, long) == (20, 50)
    assert sharpe == pytest.approx(0.9)


def test_best_params_all_nan_raises():
    grid = pd.DataFrame(
        [[np.nan, np.nan]],
        index=pd.Index([10], name="short_window"),
        columns=pd.Index([50, 100], name="long_window"),
    )
    with pytest.raises(ValueError):
        best_params(grid)