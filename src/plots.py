from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

STRATEGY_COLOR = "#00c2a8"
BENCH_COLOR = "#8a8f98"
BUY_COLOR = "#2ecc71"
SELL_COLOR = "#e74c3c"
BG = "#0e1117"
FG = "#e6e6e6"
GRID = "#2a2f3a"


def _new_axes(title: str, ylabel: str):
    """Create a dark-themed figure and axes. Uses Figure directly, so no GUI window opens."""
    fig = Figure(figsize=(10, 5), facecolor=BG)
    ax = fig.subplots()
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, loc="left")
    ax.set_xlabel("Date", color=FG)
    ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    return fig, ax


def _legend(ax) -> None:
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="upper left")


def _save(fig: Figure, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    return path


def plot_equity(bt: pd.DataFrame, ticker: str, path) -> Path:
    fig, ax = _new_axes(f"{ticker}: Strategy vs Buy & Hold", "Portfolio value ($)")
    ax.plot(bt.index, bt["equity_buyhold"], color=BENCH_COLOR, linewidth=1.6, label="Buy & Hold")
    ax.plot(bt.index, bt["equity_strategy"], color=STRATEGY_COLOR, linewidth=1.8, label="MA Crossover")
    _legend(ax)
    return _save(fig, path)


def plot_drawdown(bt: pd.DataFrame, ticker: str, path) -> Path:
    bh_drawdown = bt["equity_buyhold"] / bt["equity_buyhold"].cummax() - 1
    fig, ax = _new_axes(f"{ticker}: Drawdown", "Drawdown from peak (%)")
    ax.fill_between(bt.index, bh_drawdown * 100, 0, color=BENCH_COLOR, alpha=0.35, label="Buy & Hold")
    ax.fill_between(bt.index, bt["drawdown"] * 100, 0, color=STRATEGY_COLOR, alpha=0.55, label="MA Crossover")
    _legend(ax)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower left")
    return _save(fig, path)


def plot_signals(
    bt: pd.DataFrame, ticker: str, short_window: int, long_window: int, path
) -> Path:
    fig, ax = _new_axes(f"{ticker}: Price, Moving Averages, and Signals", "Adjusted close ($)")
    ax.plot(bt.index, bt["adj_close"], color=BENCH_COLOR, linewidth=1.0, alpha=0.9, label="Price")
    ax.plot(bt.index, bt["ma_short"], color="#f5a623", linewidth=1.3, label=f"{short_window}-day MA")
    ax.plot(bt.index, bt["ma_long"], color="#4a90e2", linewidth=1.3, label=f"{long_window}-day MA")

    change = bt["signal"].diff()
    buys = bt[change == 1]
    sells = bt[change == -1]
    ax.scatter(buys.index, buys["adj_close"], marker="^", s=70, color=BUY_COLOR, zorder=5, label="Buy signal")
    ax.scatter(sells.index, sells["adj_close"], marker="v", s=70, color=SELL_COLOR, zorder=5, label="Sell signal")
    _legend(ax)
    return _save(fig, path)


def make_static_charts(
    bt: pd.DataFrame,
    ticker: str,
    short_window: int,
    long_window: int,
    out_dir: Path = ASSETS_DIR,
) -> list[Path]:
    """Save all three README charts to out_dir and return their paths."""
    out_dir = Path(out_dir)
    return [
        plot_equity(bt, ticker, out_dir / "equity_curve.png"),
        plot_drawdown(bt, ticker, out_dir / "drawdown.png"),
        plot_signals(bt, ticker, short_window, long_window, out_dir / "price_signals.png"),
    ]