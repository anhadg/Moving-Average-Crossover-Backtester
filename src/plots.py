from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.sweep import best_params
from src.theme import (
    BENCH,
    BUY,
    FONT_BODY,
    HEAT_MID,
    INK,
    MA_LONG,
    MA_SHORT,
    PAPER,
    RULE,
    SELL,
    STRATEGY,
)

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

def plot_sweep_heatmap(
    grid: pd.DataFrame, ticker: str, path, title_suffix: str = "", highlight=None
) -> Path:
    """Heatmap of Sharpe by (short, long) window, with the best cell outlined."""
    best_short, best_long, _ = best_params(grid)  # raises if the grid is all NaN
    if highlight is not None:
        best_short, best_long = highlight
    data = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    mask = np.ma.getmaskarray(data)

    fig = Figure(figsize=(10, 6), facecolor=BG)
    ax = fig.subplots()
    ax.set_facecolor(BG)
    im = ax.imshow(data, cmap="viridis", aspect="auto", origin="lower")

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns.tolist())
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index.tolist())
    ax.set_xlabel("Long MA window (days)", color=FG)
    ax.set_ylabel("Short MA window (days)", color=FG)
    ax.set_title(
        f"{ticker}: Sharpe ratio by MA windows{title_suffix}",
        color=FG,
        fontsize=13,
        loc="left",
    )
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    midpoint = (data.min() + data.max()) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not mask[i, j]:
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if data[i, j] > midpoint else "white",
                )

    bi = grid.index.get_loc(best_short)
    bj = grid.columns.get_loc(best_long)
    ax.add_patch(
        Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="#ff4d4d", linewidth=2.5)
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sharpe ratio", color=FG)
    cbar.ax.tick_params(colors=FG)
    cbar.outline.set_edgecolor(GRID)
    return _save(fig, path)

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _style_interactive(fig: go.Figure, title: str, ylabel: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16, "color": INK}},
        height=540,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_BODY, "color": INK, "size": 13},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(title_text="Date", gridcolor=RULE, linecolor=RULE, zeroline=False)
    fig.update_yaxes(title_text=ylabel, gridcolor=RULE, linecolor=RULE, zeroline=False)
    return fig


def interactive_equity(bt: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["equity_buyhold"],
            name="Buy & Hold",
            line={"color": BENCH, "width": 2},
            hovertemplate="%{y:$,.0f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["equity_strategy"],
            name="MA crossover",
            line={"color": STRATEGY, "width": 2.6},
            hovertemplate="%{y:$,.0f}",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _style_interactive(fig, f"{ticker}: growth of starting capital", "Portfolio value ($)")


def interactive_signals(
    bt: pd.DataFrame, ticker: str, short_window: int, long_window: int
) -> go.Figure:
    change = bt["signal"].diff()
    buys = bt[change == 1]
    sells = bt[change == -1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["adj_close"],
            name="Price",
            line={"color": BENCH, "width": 1.3},
            hovertemplate="%{y:$,.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["ma_short"],
            name=f"{short_window}-day average",
            line={"color": MA_SHORT, "width": 1.8},
            hovertemplate="%{y:$,.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bt["ma_long"],
            name=f"{long_window}-day average",
            line={"color": MA_LONG, "width": 1.8},
            hovertemplate="%{y:$,.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["adj_close"],
            mode="markers",
            name="Buy signal",
            marker={"symbol": "triangle-up", "size": 12, "color": BUY, "line": {"color": PAPER, "width": 1}},
            hovertemplate="%{y:$,.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["adj_close"],
            mode="markers",
            name="Sell signal",
            marker={"symbol": "triangle-down", "size": 12, "color": SELL, "line": {"color": PAPER, "width": 1}},
            hovertemplate="%{y:$,.2f}",
        )
    )
    fig.update_yaxes(tickprefix="$")
    return _style_interactive(fig, f"{ticker}: price, averages, and signals", "Adjusted close ($)")


def interactive_drawdown(bt: pd.DataFrame, ticker: str) -> go.Figure:
    bench_dd = (bt["equity_buyhold"] / bt["equity_buyhold"].cummax() - 1) * 100
    strat_dd = bt["drawdown"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=bench_dd,
            name="Buy & Hold",
            fill="tozeroy",
            line={"color": BENCH, "width": 1.2},
            fillcolor=_rgba(BENCH, 0.35),
            hovertemplate="%{y:.1f}%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt.index,
            y=strat_dd,
            name="MA crossover",
            fill="tozeroy",
            line={"color": STRATEGY, "width": 1.4},
            fillcolor=_rgba(STRATEGY, 0.4),
            hovertemplate="%{y:.1f}%",
        )
    )
    fig.update_yaxes(ticksuffix="%")
    return _style_interactive(fig, f"{ticker}: drawdown from previous peak", "Drawdown (%)")

def interactive_sweep_heatmap(
    grid: pd.DataFrame, title: str, highlight=None
) -> go.Figure:
    """Sharpe heatmap (rows = short window, columns = long window).

    Red is negative, blue is positive, and 0 sits at the neutral middle.
    highlight is an optional (short, long) pair to outline.
    """
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(dtype=float),
            x=[str(c) for c in grid.columns],
            y=[str(r) for r in grid.index],
            colorscale=[[0.0, SELL], [0.5, HEAT_MID], [1.0, STRATEGY]],
            zmid=0,
            xgap=2,
            ygap=2,
            texttemplate="%{z:.2f}",
            hoverongaps=False,
            hovertemplate="Short %{y}, long %{x}<br>Sharpe %{z:.2f}<extra></extra>",
            colorbar={"title": {"text": "Sharpe"}, "thickness": 14},
        )
    )
    if highlight is not None:
        short, long = highlight
        xi = list(grid.columns).index(long)
        yi = list(grid.index).index(short)
        fig.add_shape(
            type="rect",
            x0=xi - 0.5,
            x1=xi + 0.5,
            y0=yi - 0.5,
            y1=yi + 0.5,
            line={"color": INK, "width": 3},
        )
    _style_interactive(fig, title, "Short average (days)")
    fig.update_xaxes(title_text="Long average (days)", type="category", showgrid=False)
    fig.update_yaxes(type="category", showgrid=False)
    fig.update_layout(hovermode="closest", height=460, showlegend=False)
    return fig