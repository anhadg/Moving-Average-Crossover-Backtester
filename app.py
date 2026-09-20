import datetime as dt
import html

import pandas as pd
import streamlit as st

from src.data import load_prices
from src.engine import extract_trades, run_backtest
from src.metrics import summarize
from src.plots import interactive_drawdown, interactive_equity, interactive_signals
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
        tab_equity, tab_signals, tab_drawdown = st.tabs(["Equity", "Signals", "Drawdown"])
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