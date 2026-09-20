import pandas as pd


def add_signals(
    prices: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Return a copy of prices with ma_short, ma_long, and signal columns.

    signal = 1 when the short SMA is above the long SMA, else 0.
    Rows before the long SMA exists (warm-up) get signal 0.
    """
    if short_window < 1 or long_window < 1:
        raise ValueError("Moving-average windows must be at least 1.")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if price_col not in prices.columns:
        raise ValueError(f"Column '{price_col}' not found in prices.")

    df = prices.copy()
    df["ma_short"] = df[price_col].rolling(window=short_window).mean()
    df["ma_long"] = df[price_col].rolling(window=long_window).mean()

    # Comparisons with NaN evaluate to False, so warm-up rows become 0 automatically.
    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)
    return df