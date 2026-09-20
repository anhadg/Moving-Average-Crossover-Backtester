"""Command-line entry point for the moving-average crossover backtester."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import load_prices
from src.engine import extract_trades, run_backtest
from src.metrics import summarize
from src.plots import make_static_charts, plot_sweep_heatmap
from src.sweep import best_params, run_sweep, train_test_analysis

METRIC_LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "volatility": "Volatility (ann.)",
    "sharpe": "Sharpe ratio",
    "max_drawdown": "Max drawdown",
    "num_trades": "Trades",
    "win_rate": "Win rate",
}
PERCENT_METRICS = {"total_return", "cagr", "volatility", "max_drawdown", "win_rate"}


def _fmt(key: str, value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if key in PERCENT_METRICS:
        return f"{value * 100:.2f}%"
    if key == "num_trades":
        return str(int(value))
    return f"{value:.2f}"


def format_summary(summary: dict) -> str:
    """Side-by-side table of strategy vs buy-and-hold metrics."""
    rows = {
        label: [
            _fmt(key, summary["strategy"][key]),
            _fmt(key, summary["buy_and_hold"][key]),
        ]
        for key, label in METRIC_LABELS.items()
    }
    return pd.DataFrame(rows, index=["Strategy", "Buy & Hold"]).T.to_string()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest a moving-average crossover strategy against buy-and-hold.",
        epilog="Educational project, not financial advice.",
        allow_abbrev=False,
    )
    p.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    p.add_argument("--short", type=int, default=50, help="Short MA window (default: 50)")
    p.add_argument("--long", type=int, default=200, help="Long MA window (default: 200)")
    p.add_argument("--fee-bps", type=float, default=5.0, help="Fee per position change, in basis points (default: 5)")
    p.add_argument("--start", default="2015-01-01", help="Start date, YYYY-MM-DD")
    p.add_argument("--end", default="2025-01-01", help="End date, YYYY-MM-DD")
    p.add_argument("--capital", type=float, default=10_000.0, help="Initial capital (default: 10000)")
    p.add_argument("--sweep", action="store_true", help="Also run the short/long window sweep and save a Sharpe heatmap")
    p.add_argument("--split-date", default=None, help="Also run the train/test analysis, splitting at this date (YYYY-MM-DD)")
    p.add_argument("--out-dir", default="assets", help="Folder for saved charts (default: assets)")
    p.add_argument("--no-charts", action="store_true", help="Skip saving charts")
    p.add_argument("--no-cache", action="store_true", help="Ignore the CSV price cache and re-download")
    return p


def _report_sweep(prices, args, ticker, out_dir) -> None:
    grid = run_sweep(prices, fee_bps=args.fee_bps, initial_capital=args.capital)
    short, long, sharpe = best_params(grid)
    print("\nParameter sweep (in-sample, full period)")
    print(f"Best pair: short={short}, long={long} (Sharpe {sharpe:.2f})")
    if not args.no_charts:
        path = plot_sweep_heatmap(grid, ticker, out_dir / "sweep_heatmap.png", " (full period, in-sample)")
        print(f"Saved {path}")


def _report_split(prices, args, ticker, out_dir) -> None:
    res = train_test_analysis(
        prices, args.split_date, fee_bps=args.fee_bps, initial_capital=args.capital
    )
    print(f"\nTrain/test analysis (split at {args.split_date})")
    print(f"Pair chosen on training data: short={res['best_short']}, long={res['best_long']}")
    print(f"Train Sharpe: {res['train_sharpe']:.2f}")
    print(f"Test Sharpe (out-of-sample): {res['test_sharpe']:.2f}")
    print(f"Buy & Hold Sharpe over the test period: {res['test_buy_and_hold']['sharpe']:.2f}")
    if res["test_rank"] is not None:
        print(f"Out-of-sample rank of the chosen pair: {res['test_rank']} of {res['test_cells']}")
    if not args.no_charts:
        train_path = plot_sweep_heatmap(
            res["train_grid"], ticker, out_dir / "sweep_train.png",
            f" (in-sample, before {args.split_date})",
        )
        test_path = plot_sweep_heatmap(
            res["test_grid"], ticker, out_dir / "sweep_test.png",
            f" (out-of-sample, from {args.split_date})",
            highlight=(res["best_short"], res["best_long"]),
        )
        print(f"Saved {train_path}")
        print(f"Saved {test_path}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ticker = args.ticker.strip().upper()
    out_dir = Path(args.out_dir)

    try:
        if pd.Timestamp(args.start) >= pd.Timestamp(args.end):
            raise ValueError("--start must be earlier than --end.")

        prices = load_prices(ticker, args.start, args.end, use_cache=not args.no_cache)
        bt = run_backtest(prices, args.short, args.long, args.fee_bps, args.capital)
        trades = extract_trades(bt)
        summary = summarize(bt, trades)

        print(f"{ticker}: {prices.index[0].date()} to {prices.index[-1].date()} ({len(prices)} trading days)")
        print(f"Strategy: {args.short}/{args.long} SMA crossover, long/flat, {args.fee_bps:g} bps per position change")
        print(f"Initial capital: ${args.capital:,.0f}\n")
        print(format_summary(summary))

        if not args.no_charts:
            print()
            for path in make_static_charts(bt, ticker, args.short, args.long, out_dir):
                print(f"Saved {path}")

        if args.sweep:
            _report_sweep(prices, args, ticker, out_dir)
        if args.split_date:
            _report_split(prices, args, ticker, out_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())