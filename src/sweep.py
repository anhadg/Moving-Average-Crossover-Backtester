import numpy as np
import pandas as pd

from src.engine import run_backtest
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
) -> pd.DataFrame:
    """Sharpe ratio of the strategy for every (short, long) window pair.

    Rows are short windows, columns are long windows. Invalid pairs
    (short >= long) and pairs the data is too short for are left as NaN.
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