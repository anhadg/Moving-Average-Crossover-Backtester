import datetime as dt
import html

import pandas as pd
import streamlit as st

from src.data import load_prices
from src.engine import extract_trades, run_backtest
from src.metrics import summarize
from src.plots import (
    interactive_drawdown,
    interactive_equity,
    interactive_signals,
    interactive_sweep_heatmap,
)
from src.sweep import train_test_analysis
from src.theme import BENCH, INK, MUTED, PAPER, RULE, SIDEBAR, STRATEGY

PRESETS = ["SPY", "QQQ", "AAPL", "Other"]
EARLIEST = dt.date(1990, 1, 1)

ROWS = [
    ("Total return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Volatility (annualized)", "volatility", "pct"),
    ("Sharpe ratio", "sharpe", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Trades", "num_trades", "int"),
    ("Win rate", "win_rate", "pct"),
]

FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Atkinson+Hyperlegible:wght@400;700&family=Bitter:wght@600;700&display=swap');"
)
ROOT = (
    f":root{{--paper:{PAPER};--sidebar:{SIDEBAR};--ink:{INK};--muted:{MUTED};"
    f"--rule:{RULE};--strategy:{STRATEGY};--bench:{BENCH};}}"
)
CSS = (
    FONTS
    + ROOT
    + """
.stApp{background:var(--paper);color:var(--ink);}
html,body,.stApp,.stApp p,.stApp label,.stApp li,.stApp input,.stApp textarea,.stApp button,.stApp [data-baseweb="select"] div{font-family:'Atkinson Hyperlegible',system-ui,sans-serif;}
[data-testid="stIconMaterial"]{font-family:'Material Symbols Rounded' !important;}
[data-testid="stHeader"]{background:transparent;height:2rem;}
.block-container{padding:0.6rem 1.4rem 1rem;max-width:100%;}
[data-testid="stSidebar"]{background:var(--sidebar);border-right:1px solid var(--rule);}
[data-testid="stSidebarUserContent"]{padding-top:0.6rem;}
[data-testid="stVerticalBlock"]{gap:0.6rem;}
[data-testid="stSidebar"] label p{font-size:0.86rem;}
[data-baseweb="tab-list"]{gap:1.4rem;border-bottom:1px solid var(--rule);}
button[data-baseweb="tab"]{padding:0.35rem 0;background:transparent;}
button[data-baseweb="tab"] p{font-weight:700;font-size:0.98rem;}
.titlebar{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;border-bottom:1px solid var(--rule);padding-bottom:0.35rem;}
.titlebar .title{font-family:'Bitter',Georgia,serif;font-weight:700;font-size:1.55rem;letter-spacing:-0.01em;}
.titlebar .sub{color:var(--muted);font-size:0.9rem;}
.ctx{color:var(--muted);font-size:0.92rem;}
.note{color:var(--muted);font-size:0.85rem;margin-top:0.5rem;line-height:1.35;}
.cmp{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;}
.cmp th{font-family:'Bitter',Georgia,serif;font-weight:600;font-size:0.95rem;text-align:right;padding:0.3rem 0.4rem 0.35rem;}
.cmp th.s{border-bottom:3px solid var(--strategy);}
.cmp th.b{border-bottom:3px solid var(--bench);}
.cmp td{padding:0.42rem 0.4rem;border-bottom:1px solid var(--rule);text-align:right;font-size:1.02rem;}
.cmp td:first-child{text-align:left;color:var(--muted);font-size:0.92rem;}
.cmp td.s{font-weight:700;color:var(--strategy);}
.cmp td.b{color:var(--ink);}
"""
)

st.set_page_config(page_title="MA crossover backtester", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def fmt(value, kind: str) -> str:
    if pd.isna(value):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "int":
        return str(int(value))
    return f"{value:.2f}"


def comparison_table(summary: dict) -> str:
    body = "".join(
        f"<tr><td>{label}</td>"
        f"<td class='s'>{fmt(summary['strategy'][key], kind)}</td>"
        f"<td class='b'>{fmt(summary['buy_and_hold'][key], kind)}</td></tr>"
        for label, key, kind in ROWS
    )
    return (
        "<table class='cmp'><thead><tr><th></th>"
        "<th class='s'>Strategy</th><th class='b'>Buy &amp; Hold</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def show_chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

ABOUT_TEXT = """
**What the strategy does.** Every day it compares two simple moving averages of the closing price, a short one and a long one (50 and 200 days by default). While the short average sits above the long average, the strategy holds the asset. Otherwise it holds cash. It never sells short and never borrows.

**How trades and costs work.** A signal is read at the close and acted on the next trading day. Each time the position changes, a fee in basis points is deducted (1 basis point is 0.01 percent). Buy and hold is the benchmark: the same asset over the same dates with no trading.

**What lookahead bias is.** Lookahead bias means using information that was not yet available when a decision was made. If a rule can see today's close and also earns today's return, the backtest looks smarter than any real trader could be. This app delays every signal by one day, and the project's tests check that changing the final day's price cannot change any earlier result.

**Why the sweep uses a train/test split.** Trying many window pairs and keeping the best one mostly rewards luck in that particular stretch of history. The Sweep tab picks the best pair on the earlier period, then scores that same pair on a later period it never saw. A big drop between the two shows how much the first number was flattered.

**Limits to keep in mind.** Prices are adjusted daily closes from Yahoo Finance, which can contain errors. The model ignores taxes and any trading costs beyond the fee you set. Results describe the past only.

**Disclaimer.** This is an educational project and not financial advice.
"""


@st.cache_data(show_spinner=False)
def cached_analysis(ticker, start, end, fee_bps, capital, split):
    prices = load_prices(ticker, start, end)
    return train_test_analysis(
        prices, split, fee_bps=fee_bps, initial_capital=capital
    )


def trades_for_display(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["entry_date"] = out["entry_date"].dt.strftime("%Y-%m-%d")
    out["exit_date"] = out["exit_date"].dt.strftime("%Y-%m-%d")
    out["return_pct"] = out["return_pct"] * 100
    return out.rename(
        columns={
            "trade_id": "Trade",
            "entry_date": "Entry date",
            "exit_date": "Exit date",
            "entry_price": "Entry price",
            "exit_price": "Exit price",
            "return_pct": "Return (%)",
            "holding_days": "Days held",
        }
    )


def render_sweep(bt: pd.DataFrame, p: dict) -> None:
    c1, c2 = st.columns(2, vertical_alignment="bottom")
    split = c1.date_input(
        "Train/test split",
        value=bt.index[len(bt) // 2].date(),
        min_value=bt.index[0].date(),
        max_value=bt.index[-1].date(),
        format="YYYY-MM-DD",
        help="Window pairs are chosen using data before this date, then scored on data from this date on.",
    )
    run_clicked = c2.button("Run sweep", width="stretch")

    key = (
        p["ticker"],
        p["start"],
        p["end"],
        float(p["fee_bps"]),
        float(p["capital"]),
        str(split),
    )
    if run_clicked:
        try:
            with st.spinner("Testing every pair of averages..."):
                analysis = cached_analysis(*key)
        except ValueError as exc:
            st.error(
                f"Could not run the sweep. {exc} Try a later split date or a longer date range."
            )
        else:
            st.session_state["sweep_results"] = {"key": key, "analysis": analysis}

    saved = st.session_state.get("sweep_results")
    if saved is None or saved["key"] != key:
        st.info(
            "Choose a split date, then run the sweep. It tests about 90 pairs of averages and takes a few seconds."
        )
        return

    res = saved["analysis"]
    view = st.radio(
        "Period shown",
        ["In-sample (train)", "Out-of-sample (test)"],
        horizontal=True,
    )
    if view.startswith("In-sample"):
        grid = res["train_grid"]
        suffix = f"in-sample, before {split}"
    else:
        grid = res["test_grid"]
        suffix = f"out-of-sample, from {split}"

    show_chart(
        interactive_sweep_heatmap(
            grid,
            f"{p['ticker']}: Sharpe ratio by pair of averages ({suffix})",
            highlight=(res["best_short"], res["best_long"]),
        )
    )
    rank_text = (
        f", ranking {res['test_rank']} of {res['test_cells']} pairs"
        if res["test_rank"]
        else ""
    )
    st.markdown(
        f"Best pair on the training data: **{res['best_short']}/{res['best_long']}** "
        f"(Sharpe {fmt(res['train_sharpe'], 'num')}). On the test period the same pair scores "
        f"**{fmt(res['test_sharpe'], 'num')}**{rank_text}. Buy and hold scores "
        f"{fmt(res['test_buy_and_hold']['sharpe'], 'num')} over the test period."
    )
    st.caption(
        "Each cell is the Sharpe ratio for one pair of averages. Blue is positive and red is negative. "
        "The outlined cell is the pair that scored best on the training data, so the test view shows "
        "how it held up on data it never saw."
    )


def render_trades(trades: pd.DataFrame, bt: pd.DataFrame, p: dict) -> None:
    if trades.empty:
        st.info(
            "The strategy made no trades in this period. Try a wider date range or shorter averages."
        )
        return
    table = trades_for_display(trades)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Entry price": st.column_config.NumberColumn(format="%.2f"),
            "Exit price": st.column_config.NumberColumn(format="%.2f"),
            "Return (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    d1, d2 = st.columns([1, 3], vertical_alignment="center")
    d1.download_button(
        "Download CSV",
        table.to_csv(index=False),
        file_name=f"{p['ticker']}_trades.csv",
        mime="text/csv",
    )
    note = "Click a column header to sort. Returns are gross price returns before fees."
    if bt["position"].iloc[-1] == 1:
        note += " The last trade is still open and is valued at the final close."
    d2.caption(note)
    
# ---------- Sidebar controls ----------
with st.sidebar:
    preset = st.selectbox("Ticker", PRESETS)
    if preset == "Other":
        ticker = st.text_input("Ticker symbol", value="MSFT").strip().upper()
    else:
        ticker = preset

    today = dt.date.today()
    c1, c2 = st.columns(2)
    start = c1.date_input(
        "Start",
        value=dt.date(2015, 1, 1),
        min_value=EARLIEST,
        max_value=today,
        format="YYYY-MM-DD",
    )
    end = c2.date_input(
        "End",
        value=dt.date(2025, 1, 1),
        min_value=EARLIEST,
        max_value=today,
        format="YYYY-MM-DD",
    )
    short_window = st.slider("Short average (days)", 5, 100, 50)
    long_window = st.slider("Long average (days)", 50, 300, 200)
    f1, f2 = st.columns(2)
    fee_bps = f1.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    capital = f2.number_input("Capital ($)", min_value=100.0, value=10_000.0, step=1_000.0)
    run_clicked = st.button("Run backtest", type="primary", width="stretch")

# ---------- Header ----------
st.markdown(
    "<div class='titlebar'><span class='title'>MA crossover backtester</span>"
    "<span class='sub'>Educational project, not financial advice.</span></div>",
    unsafe_allow_html=True,
)

# ---------- Run the backtest ----------
# Run once automatically on first load so a visitor sees results immediately.
should_run = run_clicked or not st.session_state.get("auto_ran", False)
st.session_state["auto_ran"] = True

if should_run:
    params = {
        "ticker": ticker,
        "start": str(start),
        "end": str(end),
        "short": short_window,
        "long": long_window,
        "fee_bps": fee_bps,
        "capital": capital,
    }

    error = None
    if not ticker:
        error = "Enter a ticker symbol."
    elif start >= end:
        error = "The start date must be earlier than the end date."
    elif short_window >= long_window:
        error = "The short average must be smaller than the long average."

    if error:
        st.error(error)
    else:
        try:
            with st.spinner(f"Downloading {ticker} prices and running the backtest..."):
                prices = load_prices(ticker, params["start"], params["end"])
                bt = run_backtest(prices, short_window, long_window, fee_bps, capital)
                trades = extract_trades(bt)
                metrics = summarize(bt, trades)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "Could not download price data right now. Check your connection and try again."
            )
        else:
            st.session_state["params"] = params
            st.session_state["results"] = {
                "backtest": bt,
                "trades": trades,
                "metrics": metrics,
            }

# ---------- Main area ----------
results = st.session_state.get("results")
if results is None:
    st.info("Pick your settings in the sidebar, then choose Run backtest.")
else:
    p = st.session_state["params"]
    bt = results["backtest"]
    st.markdown(
        f"<div class='ctx'>{html.escape(p['ticker'])} from {bt.index[0].date()} to "
        f"{bt.index[-1].date()}. {p['short']}/{p['long']}-day average crossover, "
        f"{p['fee_bps']:g} bps per position change, starting capital ${p['capital']:,.0f}.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2.4], gap="medium")
    with left:
        st.markdown(comparison_table(results["metrics"]), unsafe_allow_html=True)
        st.markdown(
            "<div class='note'>Fees are charged on every position change. Win rate counts "
            "trades with a positive gross return. Buy and hold is a single trade.</div>",
            unsafe_allow_html=True,
        )
    with right:
        tab_equity, tab_signals, tab_drawdown, tab_sweep, tab_trades, tab_about = st.tabs(
            ["Equity", "Signals", "Drawdown", "Sweep", "Trades", "About"]
        )
        with tab_equity:
            show_chart(interactive_equity(bt, p["ticker"]))
            st.caption(
                "How the starting capital would have grown under the crossover rule "
                "compared with simply holding. Fees are included in the strategy line."
            )
        with tab_signals:
            show_chart(interactive_signals(bt, p["ticker"], p["short"], p["long"]))
            st.caption(
                "A green triangle marks the day the short average crosses above the long one, "
                "and a red triangle marks the cross back below. The position changes the next trading day."
            )
        with tab_drawdown:
            show_chart(interactive_drawdown(bt, p["ticker"]))
            st.caption("How far each portfolio sat below its previous peak. Shallower is better.")
        with tab_sweep:
            render_sweep(bt, p)
        with tab_trades:
            render_trades(results["trades"], bt, p)
        with tab_about:
            st.markdown(ABOUT_TEXT)