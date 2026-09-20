import pandas as pd
import pytest

from src.signals import add_signals


def make_prices(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D", name="date")
    return pd.DataFrame({"adj_close": values}, index=idx)


def test_signal_matches_hand_computed_values():
    # Prices fall, then rise. With windows 2 and 4, the short MA crosses
    # above the long MA at index 6.
    prices = make_prices([10, 9, 8, 7, 6, 7, 8, 9, 10, 11])
    out = add_signals(prices, short_window=2, long_window=4)
    assert out["signal"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_warmup_rows_are_flat():
    prices = make_prices(range(1, 21))
    out = add_signals(prices, short_window=3, long_window=10)
    assert out["ma_long"].iloc[:9].isna().all()
    assert (out["signal"].iloc[:9] == 0).all()


def test_signal_is_binary_int():
    prices = make_prices([5, 6, 7, 6, 5, 6, 7, 8, 7, 6, 5, 6])
    out = add_signals(prices, short_window=2, long_window=5)
    assert set(out["signal"].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(out["signal"])


def test_invalid_windows_raise():
    prices = make_prices(range(1, 21))
    with pytest.raises(ValueError):
        add_signals(prices, short_window=200, long_window=50)
    with pytest.raises(ValueError):
        add_signals(prices, short_window=0, long_window=10)