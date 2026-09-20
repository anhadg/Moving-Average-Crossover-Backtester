import numpy as np
import pandas as pd

from src.signals import add_signals

TRADE_COLUMNS = [
    "trade_id",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "return_pct",
    "holding_days",
]
BACKTEST_COLUMNS = [
    "adj_close",
    "ma_short",
    "ma_long",
    "signal",
    "position",
    "market_return",
    "strategy_return",
    "trade_flag",
    "equity_strategy",
    "equity_buyhold",
    "drawdown",
]


def run_backtest(
    prices: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    fee_bps: float = 5.0,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Simulate a long/flat MA-crossover strategy and a buy-and-hold benchmark.

    The position held on day t is the signal computed at the close of day t-1
    (shift by 1), so the strategy never trades on information it could not have had.
    A fee of fee_bps basis points is charged on every day the position changes.
    """
    if fee_bps < 0:
        raise ValueError("fee_bps cannot be negative.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if len(prices) <= long_window:
        raise ValueError(
            f"Need more than {long_window} trading days of data for a "
            f"{long_window}-day moving average (got {len(prices)}). "
            "Choose a longer date range or a shorter long window."
        )

    df = add_signals(prices, short_window, long_window)

    df["position"] = df["signal"].shift(1).fillna(0).astype(int)
    df["market_return"] = df["adj_close"].pct_change().fillna(0.0)
    df["trade_flag"] = df["position"].diff().abs().fillna(0).astype(int)

    cost = df["trade_flag"] * (fee_bps / 10_000)
    df["strategy_return"] = df["position"] * df["market_return"] - cost

    df["equity_strategy"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["equity_buyhold"] = initial_capital * (1 + df["market_return"]).cumprod()
    df["drawdown"] = df["equity_strategy"] / df["equity_strategy"].cummax() - 1

    return df[BACKTEST_COLUMNS]

def extract_trades(bt: pd.DataFrame) -> pd.DataFrame:
    """Build the trades table from a backtest DataFrame.

    Because the engine credits day t's return to the position decided at the
    close of day t-1, a trade is entered at the close of the signal day (the
    day before the first held day) and exited at the close of the last held day.
    return_pct is the gross price return (fees excluded). holding_days counts
    calendar days. A trade still open on the final row is marked to that
    day's close.
    """
    pos = bt["position"].to_numpy()
    prev_pos = np.concatenate(([0], pos[:-1]))
    next_pos = np.concatenate((pos[1:], [0]))

    first_held = np.flatnonzero((pos == 1) & (prev_pos == 0))
    last_held = np.flatnonzero((pos == 1) & (next_pos == 0))

    rows = []
    for trade_id, (first, last) in enumerate(zip(first_held, last_held), start=1):
        entry_i = first - 1  # the signal day; position is 0 on row 0, so first >= 1
        entry_date = bt.index[entry_i]
        exit_date = bt.index[last]
        entry_price = float(bt["adj_close"].iloc[entry_i])
        exit_price = float(bt["adj_close"].iloc[last])
        rows.append(
            {
                "trade_id": trade_id,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": exit_price / entry_price - 1,
                "holding_days": int((exit_date - entry_date).days),
            }
        )

    return pd.DataFrame(rows, columns=TRADE_COLUMNS)