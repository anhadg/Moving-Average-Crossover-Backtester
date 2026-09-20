import numpy as np
import pandas as pd

from src.engine import run_backtest
from src.plots import interactive_drawdown, interactive_equity, interactive_signals


def make_bt():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="D", name="date")
    prices = pd.DataFrame(
        {"adj_close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))}, index=idx
    )
    return run_backtest(prices, 5, 20)


def test_interactive_charts_build():
    bt = make_bt()
    assert len(interactive_equity(bt, "SPY").data) == 2
    assert len(interactive_signals(bt, "SPY", 5, 20).data) == 5
    assert len(interactive_drawdown(bt, "SPY").data) == 2