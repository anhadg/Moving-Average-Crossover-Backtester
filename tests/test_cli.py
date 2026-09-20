import numpy as np
import pandas as pd

import backtest


def fake_prices(*args, **kwargs):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", periods=600, freq="B", name="date")
    return pd.DataFrame(
        {"adj_close": 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 600))}, index=idx
    )


def base_args(tmp_path, *extra):
    return [
        "--ticker", "SPY", "--short", "10", "--long", "50",
        "--start", "2018-01-01", "--end", "2020-12-31",
        "--out-dir", str(tmp_path), *extra,
    ]


def test_default_run_prints_summary_and_saves_charts(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(backtest, "load_prices", fake_prices)
    assert backtest.main(base_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "Sharpe ratio" in out
    assert "Buy & Hold" in out
    for name in ("equity_curve.png", "drawdown.png", "price_signals.png"):
        assert (tmp_path / name).exists()


def test_no_charts_flag_saves_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(backtest, "load_prices", fake_prices)
    assert backtest.main(base_args(tmp_path, "--no-charts")) == 0
    assert list(tmp_path.glob("*.png")) == []


def test_invalid_windows_return_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(backtest, "load_prices", fake_prices)
    args = ["--short", "200", "--long", "50", "--out-dir", str(tmp_path)]
    assert backtest.main(args) == 1
    assert "smaller" in capsys.readouterr().err


def test_start_after_end_returns_error(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(backtest, "load_prices", fake_prices)
    args = ["--start", "2025-01-01", "--end", "2020-01-01", "--out-dir", str(tmp_path)]
    assert backtest.main(args) == 1
    assert "--start" in capsys.readouterr().err


def test_data_error_returns_error(monkeypatch, capsys, tmp_path):
    def bad_loader(*args, **kwargs):
        raise ValueError("No data found for ticker 'ZZZZ'.")

    monkeypatch.setattr(backtest, "load_prices", bad_loader)
    assert backtest.main(base_args(tmp_path)) == 1
    assert "No data found" in capsys.readouterr().err


def test_sweep_flag_prints_best_pair_and_saves_heatmap(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(backtest, "load_prices", fake_prices)
    assert backtest.main(base_args(tmp_path, "--sweep")) == 0
    assert "Best pair" in capsys.readouterr().out
    assert (tmp_path / "sweep_heatmap.png").exists()


def test_split_date_runs_train_test_analysis(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(backtest, "load_prices", fake_prices)
    assert backtest.main(base_args(tmp_path, "--split-date", "2019-06-01")) == 0
    out = capsys.readouterr().out
    assert "Test Sharpe (out-of-sample)" in out
    assert (tmp_path / "sweep_train.png").exists()
    assert (tmp_path / "sweep_test.png").exists()