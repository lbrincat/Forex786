# --- My Signal Pro (Full Strategy + Golden Cross Scanner) ---
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
# Sidebar
# ==============================
with st.sidebar:
    st.subheader("⚙️ Settings")

    mode = st.radio(
        "Scanner Mode",
        ["Full Strategy Signals", "Golden Cross Scanner"],
        index=0,
        help="Full Strategy = RSI/MACD/etc. Golden Cross = EMA13 vs EMA49 cross alerts."
    )

    timeframe = st.selectbox(
        "Timeframe",
        ["5m", "15m", "30m", "1h", "4h", "1d"],
        index=0,
        help="All signals are calculated on this interval."
    )

    auto_refresh_ms = st.slider("Auto-refresh (seconds)", 60, 600, 180, step=30) * 1000

    trade_strength = st.multiselect(
        "Trade signal strengths (for paper trades)",
        ["Strong", "Medium"],
        default=["Strong"]
    )

    sl_mult = st.number_input("SL = ATR ×", 0.5, 10.0, 1.2, 0.1)
    tp_mult = st.number_input("TP = ATR ×", 0.5, 10.0, 2.5, 0.1)

    safe_mode = st.checkbox(
        "Safe Mode (skip network calls)",
        value=True,
        help="Turn this ON in Streamlit Cloud for fast first load. Turn OFF to fetch data."
    )

    st.caption("Tip: 'Golden Cross Scanner' ignores RSI/MACD and only reports EMA13/EMA49 crosses.")

# Manual refresh button
if st.button("🔄 Refresh now"):
    st.rerun()

# ==============================
# Helper Functions
# ==============================
def play_beep_once():
    """Play a short beep in the browser"""
    components.html("""
    <audio autoplay>
        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
    </audio>
    """, height=0)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yf_data_resilient(label, ticker, interval):
    """Get candle data from Yahoo Finance"""
    try:
        df = yf.download(
            tickers=ticker.replace("/", ""),
            period="5d",
            interval=interval,
            progress=False,
            threads=False
        )
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        df.reset_index(inplace=True)
        df.rename(columns={"datetime": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_forex_factory_news_today():
    """Fetch Forex Factory XML and filter for today"""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        news = []
        today = datetime.utcnow().date()
        for item in root.findall("./channel/item"):
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text)
            currency = item.find("{http://www.forexfactory.com/rss}currency").text.strip().upper()
            if pub_time.date() == today:
                news.append({"title": title, "time": pub_time, "currency": currency})
        return news
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_dxy_data():
    """Fetch Dollar Index"""
    try:
        hist = yf.download("DX-Y.NYB", period="1d", interval="5m", progress=False, threads=False)
        if hist.empty:
            return None, None
        cur = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[0]
        change = (cur - prev) / prev * 100
        return cur, change
    except Exception:
        return None, None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['+DM'] = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
                         np.maximum(df['high'] - df['high'].shift(), 0), 0)
    df['-DM'] = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
                         np.maximum(df['low'].shift() - df['low'], 0), 0)
    tr14 = df['TR'].rolling(window=period).mean()
    plus_dm14 = df['+DM'].rolling(window=period).mean()
    minus_dm14 = df['-DM'].rolling(window=period).mean()
    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    return dx.rolling(window=period).mean()

def detect_ma_cross(df, fast_col="EMA13", slow_col="EMA49"):
    """Detect EMA13/EMA49 golden/death cross"""
    if len(df) < 2:
        return ""
    f_prev, s_prev = df[fast_col].iloc[-2], df[slow_col].iloc[-2]
    f_now, s_now = df[fast_col].iloc[-1], df[slow_col].iloc[-1]
    if f_prev < s_prev and f_now > s_now:
        return "BUY"
    if f_prev > s_prev and f_now < s_now:
        return "SELL"
    return ""

# ==============================
# Symbol list
# ==============================
symbols = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "CAD=X", "USD/CHF": "CHF=X",
    "XAU/USD": "GC=F", "XAG/USD": "SI=F", "WTI/USD": "CL=F",
}

# ==============================
# Data fetching
# ==============================
rows = []
if not safe_mode:
    dxy_price, dxy_change = fetch_dxy_data()
else:
    dxy_price, dxy_change = (None, None)

for label, yf_symbol in symbols.items():
    if safe_mode:
        continue

    df = fetch_yf_data_resilient(label, yf_symbol, timeframe)
    if df.empty:
        continue

    df["EMA13"] = calculate_ema(df["close"], 13)
    df["EMA49"] = calculate_ema(df["close"], 49)
    df.dropna(inplace=True)

    price = float(df["close"].iloc[-1])

    if mode == "Golden Cross Scanner":
        cross = detect_ma_cross(df)
        rows.append({
            "Pair": label,
            "Price": round(price, 5),
            "EMA13": round(df["EMA13"].iloc[-1], 5),
            "EMA49": round(df["EMA49"].iloc[-1], 5),
            "Cross Signal": cross or "—"
        })
        continue

    # --- Full Strategy below ---
    df["RSI"] = calculate_rsi(df["close"])
    df["MACD"], df["MACD_Signal"] = calculate_macd(df["close"])
    df["EMA9"] = calculate_ema(df["close"], 9)
    df["EMA20"] = calculate_ema(df["close"], 20)
    df["ATR"] = calculate_atr(df)
    df["ADX"] = calculate_adx(df)
    df.dropna(inplace=True)

    atr = df["ATR"].iloc[-1]
    rsi_val = df["RSI"].iloc[-1]
    trend = "Bullish" if df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1] else "Bearish"

    signal_type = "Bullish" if rsi_val > 50 else "Bearish"
    indicators = ["RSI", "EMA", "MACD", "ADX"]
    strength = "Strong" if df["ADX"].iloc[-1] > 25 else "Medium"
    suggestion = f"{strength} {signal_type} @ {price:.5f}"

    rows.append({
        "Pair": label,
        "Price": round(price, 5),
        "RSI": round(rsi_val, 2),
        "ATR": round(atr, 5),
        "Trend": trend,
        "Signal Type": signal_type,
        "AI Suggestion": suggestion,
        "DXY Impact": f"{dxy_price:.2f} ({dxy_change:+.2f}%)" if dxy_price else "—",
    })

# ==============================
# Render table
# ==============================
if not rows:
    st.warning("No data to display yet. Turn OFF Safe Mode to fetch data.")
else:
    df = pd.DataFrame(rows)
    if mode == "Golden Cross Scanner":
        st.subheader("📈 Golden Cross Scanner Results")
        for _, row in df.iterrows():
            color = "#d4edda" if row["Cross Signal"] == "BUY" else "#f8d7da" if row["Cross Signal"] == "SELL" else ""
            st.markdown(
                f"<div style='background:{color};padding:5px;border-radius:8px;'>"
                f"<b>{row['Pair']}</b> — Price: {row['Price']} — "
                f"<b>{row['Cross Signal']}</b>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.subheader("📊 Full Strategy Signals")
        st.dataframe(df, use_container_width=True)

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
