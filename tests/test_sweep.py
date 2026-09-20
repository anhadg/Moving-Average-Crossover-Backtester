import numpy as np
import pandas as pd
import pytest
from src.engine import rebase_backtest, run_backtest
from src.metrics import compute_metrics
from src.sweep import best_params, run_sweep, train_test_analysis

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

def test_eval_start_scores_only_the_out_of_sample_window():
    prices = random_walk(400)
    split = prices.index[250]
    grid = run_sweep(prices, [5], [20], fee_bps=5, eval_start=split)
    bt = rebase_backtest(run_backtest(prices, 5, 20, fee_bps=5), split, 10_000.0)
    expected = compute_metrics(bt["strategy_return"], bt["equity_strategy"])["sharpe"]
    assert grid.loc[5, 20] == pytest.approx(expected)


def test_train_test_no_leakage_from_test_period():
    prices = random_walk(500)
    split = prices.index[300]
    altered = prices.copy()
    altered.iloc[300:, altered.columns.get_loc("adj_close")] *= 1.5
    a = train_test_analysis(prices, split, [5, 10], [20, 50])
    b = train_test_analysis(altered, split, [5, 10], [20, 50])
    # Changing only test-period prices must not change what the optimizer sees or picks.
    pd.testing.assert_frame_equal(a["train_grid"], b["train_grid"])
    assert (a["best_short"], a["best_long"]) == (b["best_short"], b["best_long"])


def test_train_test_results_are_consistent():
    prices = random_walk(500)
    res = train_test_analysis(prices, prices.index[300], [5, 10], [20, 50])
    s, l = res["best_short"], res["best_long"]
    assert res["train_sharpe"] == pytest.approx(res["train_grid"].loc[s, l])
    assert res["train_sharpe"] == pytest.approx(res["train_grid"].max().max())
    assert res["test_sharpe"] == pytest.approx(res["test_grid"].loc[s, l])
    assert res["test_cells"] == 4
    assert 1 <= res["test_rank"] <= res["test_cells"]


def test_train_test_invalid_split_raises():
    prices = random_walk(300)
    with pytest.raises(ValueError):
        train_test_analysis(prices, "2000-01-01", [5], [20])  # before the data starts
    with pytest.raises(ValueError):
        train_test_analysis(prices, "2100-01-01", [5], [20])  # after the data ends