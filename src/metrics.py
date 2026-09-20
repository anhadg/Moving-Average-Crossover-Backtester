import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """Core performance stats from a daily return series and its equity curve.

    The first row is the starting point (its return is 0 by construction), so it
    is excluded from volatility and Sharpe. risk_free_rate is an annual rate.
    """
    if len(equity) < 2:
        raise ValueError("Need at least 2 data points to compute metrics.")

    r = returns.iloc[1:]
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])

    total_return = end / start - 1

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years > 0 and start > 0 and end > 0:
        cagr = (end / start) ** (1 / years) - 1
    else:
        cagr = np.nan

    daily_vol = float(r.std(ddof=1))
    volatility = daily_vol * np.sqrt(TRADING_DAYS)

    if daily_vol > 1e-12:
        excess = r.mean() - risk_free_rate / TRADING_DAYS
        sharpe = float(excess / daily_vol * np.sqrt(TRADING_DAYS))
    else:
        sharpe = np.nan

    max_drawdown = float((equity / equity.cummax() - 1).min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def trade_stats(trades: pd.DataFrame) -> dict:
    """Number of trades and win rate (share of trades with a positive gross return)."""
    n = len(trades)
    win_rate = float((trades["return_pct"] > 0).mean()) if n > 0 else np.nan
    return {"num_trades": n, "win_rate": win_rate}


def summarize(
    bt: pd.DataFrame,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """Metrics for the strategy and for buy-and-hold, keyed for side-by-side display."""
    strategy = compute_metrics(bt["strategy_return"], bt["equity_strategy"], risk_free_rate)
    strategy.update(trade_stats(trades))

    buy_hold = compute_metrics(bt["market_return"], bt["equity_buyhold"], risk_free_rate)
    buy_hold.update({"num_trades": 1, "win_rate": np.nan})

    return {"strategy": strategy, "buy_and_hold": buy_hold}