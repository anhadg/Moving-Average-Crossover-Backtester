import numpy as np
import pandas as pd

from src.engine import run_backtest
from src.plots import (
    interactive_drawdown,
    interactive_equity,
    interactive_signals,
    interactive_sweep_heatmap,
)
from src.sweep import run_sweep


def make_prices(n=300):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame(
        {"adj_close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n))}, index=idx
    )


def make_bt():
    return run_backtest(make_prices(), 5, 20)


def test_interactive_charts_build():
    bt = make_bt()
    assert len(interactive_equity(bt, "SPY").data) == 2
    assert len(interactive_signals(bt, "SPY", 5, 20).data) == 5
    assert len(interactive_drawdown(bt, "SPY").data) == 2


def test_sweep_heatmap_builds_with_highlight():
    grid = run_sweep(make_prices(), [5, 10], [20, 50])
    fig = interactive_sweep_heatmap(grid, "test", (5, 20))
    assert len(fig.data) == 1
    assert len(fig.layout.shapes) == 1