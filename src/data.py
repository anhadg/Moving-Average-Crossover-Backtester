from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance and clean it into our schema."""
    df = yf.download(
        ticker, start=start, end=end, auto_adjust=False, progress=False
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. Check the symbol and date range."
        )

    # Newer yfinance versions return MultiIndex columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    df = df[COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index().dropna()


def load_prices(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool = True,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Return daily prices for [start, end). Reuses data/{ticker}.csv when it covers the range."""
    ticker = ticker.strip().upper()
    start_ts = pd.Timestamp(start)
    end_ts = min(pd.Timestamp(end), pd.Timestamp.today().normalize())
    cache_path = Path(data_dir) / f"{ticker}.csv"

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, index_col="date", parse_dates=True)
        tolerance = pd.Timedelta(days=7)  # weekends and holidays
        if (
            not cached.empty
            and cached.index.min() <= start_ts + tolerance
            and cached.index.max() >= end_ts - tolerance
        ):
            return cached.loc[(cached.index >= start_ts) & (cached.index < end_ts)]

    df = _download(ticker, start, end)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df