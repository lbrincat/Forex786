# --- My Signal Pro (Full Strategy + Golden Cross Scanner, no autorefresh) ---
import os
import tempfile
import sqlite3
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from contextlib import closing
from datetime import datetime
from pytz import timezone
from dateutil import parser as date_parser
import xml.etree.ElementTree as ET

# --- PAGE SETUP ---
st.set_page_config(page_title="My Signal Pro", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 My Signal Pro</h1>", unsafe_allow_html=True)

# ==============================
# Sidebar controls
# ==============================
with st.sidebar:
    st.subheader("⚙️ Settings")

    mode = st.radio(
        "Scanner Mode",
        ["Full Strategy Signals", "Golden Cross Scanner"],
        index=0,
        help="Full Strategy = RSI/MACD/ADX/ATR model with basic strength.\n"
             "Golden Cross = EMA13 vs EMA49 crossover scanner (BUY/SELL)."
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["5m", "15m", "30m", "1h", "4h", "1d"],
        index=0,
        help="All calculations are done on this timeframe."
    )

    # keeping this slider for future use, but no autorefresh call anymore
    auto_refresh_ms = st.slider(
        "Auto-refresh (seconds, manual only here)",
        60, 600, 180, step=30
    ) * 1000

    trade_strength = st.multiselect(
        "Trade strength filter (used in Full Strategy mode)",
        ["Strong", "Medium"],
        default=["Strong"]
    )

    sl_mult = st.number_input("SL = ATR ×", 0.5, 10.0, 1.2, 0.1)
    tp_mult = st.number_input("TP = ATR ×", 0.5, 10.0, 2.5, 0.1)

    safe_mode = st.checkbox(
        "Safe Mode (skip network calls)",
        value=True,
        help="Leave this ON in Streamlit Cloud if you just want the UI to load.\n"
             "Turn OFF to fetch live data from Yahoo Finance."
    )

    st.caption("Golden Cross mode ignores RSI/MACD/etc. and only reports EMA13/EMA49 crosses.")

# Manual refresh button
if st.button("🔄 Refresh now"):
    st.rerun()

# ==============================
# Helper functions
# ==============================

def play_beep_once():
    """Browser beep (used for divergence or alerts if you want)."""
    components.html(
        """
        <audio autoplay>
            <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
        </audio>
        """,
        height=0
    )

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yf_data_resilient(label, ticker, interval):
    """
    Download price candles (Open/High/Low/Close/Volume) from Yahoo Finance,
    using a short period so Cloud isn't hammered.
    """
    try:
        # yfinance tickers for FX pairs often end with =X, e.g. EURUSD=X
        df = yf.download(
            tickers=ticker,
            period="5d",
            interval=interval,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower).reset_index()
        # standardize timestamp column
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "time"}, inplace=True)
        elif "Date" in df.columns:
            df.rename(columns={"Date": "time"}, inplace=True)
        elif "datetime" in df.columns:
            df.rename(columns={"datetime": "time"}, inplace=True)
        else:
            df.rename(columns={"index": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").set_index("time")
        # keep expected OHLCV cols
        keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep_cols]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_forex_factory_news_today():
    """
    Get today's scheduled news from Forex Factory RSS.
    We don't yet surface this in UI, but you can integrate later.
    """
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
    except Exception:
        return []

    today_utc = datetime.utcnow().date()
    events = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text)
            currency_el = item.find("{http://www.forexfactory.com/rss}currency")
            currency = currency_el.text.strip().upper() if currency_el is not None else ""
            if pub_time.date() == today_utc:
                events.append({"title": title, "time": pub_time, "currency": currency})
        except Exception:
            continue
    return events

@st.cache_data(ttl=300, show_spinner=False)
def fetch_dxy_data():
    """
    Fetch a proxy for the US Dollar Index (DXY). If it fails, return (None, None).
    """
    for ticker in ["DX-Y.NYB", "^DXY"]:
        try:
            hist = yf.download(
                tickers=ticker,
                period="1d",
                interval="5m",
                progress=False,
                threads=False
            )
            if hist is None or hist.empty:
                continue
            current = float(hist["Close"].iloc[-1])
            start = float(hist["Close"].iloc[0])
            pct_change = ((current - start) / start * 100.0) if start else 0.0
            return current, pct_change
        except Exception:
            continue
    return None, None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    # Average True Range
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    # Simplified ADX calc
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

def detect_ma_cross(df, fast_col="EMA13", slow_col="EMA49"):
    """
    Golden cross / death cross detector:
    - BUY  if EMA13 crosses above EMA49
    - SELL if EMA13 crosses below EMA49
    We only care about the *latest* bar.
    """
    if len(df) < 2:
        return ""
    prev_fast = df[fast_col].iloc[-2]
    prev_slow = df[slow_col].iloc[-2]
    now_fast = df[fast_col].iloc[-1]
    now_slow = df[slow_col].iloc[-1]

    # Golden cross
    if prev_fast < prev_slow and now_fast > now_slow:
        return "BUY"
    # Death cross
    if prev_fast > prev_slow and now_fast < now_slow:
        return "SELL"

    return ""

# (Optional) simple strength labeler for the full strategy section
def classify_strength(adx_value):
    if adx_value >= 25:
        return "Strong"
    elif adx_value >= 18:
        return "Medium"
    else:
        return "Weak"

# ==============================
# Market symbols
# ==============================
symbols = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "XAU/USD": "GC=F",   # Gold futures proxy
    "XAG/USD": "SI=F",   # Silver futures
    "WTI/USD": "CL=F",   # Crude oil futures
}

# ==============================
# Fetch context data
# ==============================
if safe_mode:
    dxy_price, dxy_change = (None, None)
else:
    dxy_price, dxy_change = fetch_dxy_data()

# ==============================
# Scan all symbols
# ==============================
rows = []

if not safe_mode:
    for pair_label, yf_symbol in symbols.items():
        df = fetch_yf_data_resilient(pair_label, yf_symbol, timeframe)
        if df.empty or not {"open", "high", "low", "close"}.issubset(df.columns):
            continue

        # common EMAs
        df["EMA13"] = calculate_ema(df["close"], 13)
        df["EMA49"] = calculate_ema(df["close"], 49)

        if mode == "Golden Cross Scanner":
            df = df.dropna()
            if df.empty:
                continue
            last_price = float(df["close"].iloc[-1])
            cross_sig = detect_ma_cross(df, "EMA13", "EMA49")
            rows.append({
                "Pair": pair_label,
                "Price": round(last_price, 5),
                "EMA13": round(df["EMA13"].iloc[-1], 5),
                "EMA49": round(df["EMA49"].iloc[-1], 5),
                "Cross Signal": cross_sig if cross_sig else "—",
            })
            continue

        # --- Full Strategy path ---
        df["RSI"] = calculate_rsi(df["close"])
        macd_line, macd_sig = calculate_macd(df["close"])
        df["MACD"] = macd_line
        df["MACD_Signal"] = macd_sig
        df["EMA9"] = calculate_ema(df["close"], 9)
        df["EMA20"] = calculate_ema(df["close"], 20)
        df["ATR"] = calculate_atr(df)
        df["ADX"] = calculate_adx(df)

        df = df.dropna()
        if df.empty:
            continue

        last_close = float(df["close"].iloc[-1])
        last_rsi = float(df["RSI"].iloc[-1])
        last_atr = float(df["ATR"].iloc[-1])
        last_adx = float(df["ADX"].iloc[-1])
        last_ema9 = float(df["EMA9"].iloc[-1])
        last_ema20 = float(df["EMA20"].iloc[-1])

        # simple trend read
        trend = "Bullish" if last_ema9 > last_ema20 else "Bearish"

        # simple signal type from RSI midpoint
        signal_type = "Bullish" if last_rsi >= 50 else "Bearish"

        # classify "strength" using ADX
        strength = classify_strength(last_adx)

        # generate basic suggestion
        suggestion = (
            f"{strength} {signal_type} Signal @ {last_close:.5f} | "
            f"ATR: {last_atr:.5f}"
        )

        # dollar index impact for USD pairs
        if "USD" in pair_label and dxy_price is not None and dxy_change is not None:
            dxy_disp = f"{dxy_price:.2f} ({dxy_change:+.2f}%)"
        else:
            dxy_disp = "—"

        rows.append({
            "Pair": pair_label,
            "Price": round(last_close, 5),
            "RSI": round(last_rsi, 2),
            "ATR": round(last_atr, 5),
            "Trend": trend,
            "Signal Type": signal_type,
            "Strength": strength,
            "AI Suggestion": suggestion,
            "DXY Impact": dxy_disp,
        })

# ==============================
# Render results
# ==============================
if not rows:
    st.warning("No data to display. Either Safe Mode is ON (no network), or Yahoo Finance returned nothing.")
else:
    if mode == "Golden Cross Scanner":
        st.subheader("📈 Golden Cross Scanner (EMA13 vs EMA49)")
        df_rows = pd.DataFrame(rows)

        # Sort: BUY first, SELL next, others last
        df_rows["sort_key"] = np.select(
            [
                df_rows["Cross Signal"] == "BUY",
                df_rows["Cross Signal"] == "SELL",
            ],
            [2, 1],
            default=0
        )
        df_rows = df_rows.sort_values(by="sort_key", ascending=False).drop(columns=["sort_key"])

        # Pretty render
        for _, r in df_rows.iterrows():
            color = "#d4edda" if r["Cross Signal"] == "BUY" else \
                    "#f8d7da" if r["Cross Signal"] == "SELL" else ""
            st.markdown(
                f"<div style='background:{color};padding:8px;border-radius:8px;"
                f"margin-bottom:6px;border:1px solid #ccc;'>"
                f"<b>{r['Pair']}</b><br>"
                f"Price: {r['Price']}<br>"
                f"EMA13: {r['EMA13']} | EMA49: {r['EMA49']}<br>"
                f"Signal: <b>{r['Cross Signal']}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

    else:
        st.subheader("📊 Full Strategy Signals")
        df_rows = pd.DataFrame(rows)

        # highlight rows with Strong
        def highlight_row(row):
            if row["Strength"] == "Strong":
                return ['background-color: #d4edda'] * len(row)
            elif row["Strength"] == "Medium":
                return ['background-color: #d1ecf1'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(
            df_rows.style.apply(highlight_row, axis=1),
            use_container_width=True
        )

tz = timezone("Europe/Malta")
st.caption(
    f"Mode: {mode} | Timeframe: {timeframe} | Last updated: "
    f"{datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')}"
)
