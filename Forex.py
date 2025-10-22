# --- My Signal Pro (yfinance + timeframe selector + DB + paper trading + backtest) ---
import os
import sqlite3
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from contextlib import closing
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from pytz import timezone
from dateutil import parser as date_parser
import xml.etree.ElementTree as ET

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="Signals", layout="wide")
st.markdown("<h1 style='text-align:center; color:#007acc;'>📊 My Signal Pro</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ Settings")
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], index=0)
    auto_refresh_ms = st.slider("Auto-refresh (seconds)", 60, 600, 180, step=30) * 1000
    trade_strength = st.multiselect("Trade signal strengths", ["Strong", "Medium"], default=["Strong"])
    sl_mult = st.number_input("SL = ATR ×", 0.5, 10.0, 1.2, 0.1)
    tp_mult = st.number_input("TP = ATR ×", 0.5, 10.0, 2.5, 0.1)
    st.caption("Tip: Longer intervals reduce Yahoo throttling.")

st_autorefresh(interval=auto_refresh_ms, key="ai_refresh")
if st.button("🔄 Refresh now"):
    st.rerun()

MTZ = timezone("Europe/Malta")

# =========================
# SYMBOL MAP (App label -> Yahoo Finance ticker)
# =========================
symbols = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X", "NZD/USD": "NZDUSD=X", "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X", "EUR/CAD": "EURCAD=X", "GBP/JPY": "GBPJPY=X", "EUR/AUD": "EURAUD=X",
    "AUD/JPY": "AUDJPY=X", "GBP/NZD": "GBPNZD=X", "EUR/NZD": "EURNZD=X",
    "XAU/USD": "GC=F",     # Gold futures (robust)
    "XAG/USD": "SI=F",     # Silver futures
    "WTI/USD": "CL=F",     # WTI crude futures
}

# =========================
# DB SETUP
# =========================
DB_PATH = "signals.db"

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            pair TEXT NOT NULL,
            price REAL NOT NULL,
            rsi REAL,
            atr REAL,
            atr_status TEXT,
            trend TEXT,
            reversal_signal TEXT,
            signal_type TEXT,
            indicators TEXT,
            candle_pattern TEXT,
            divergence TEXT,
            strength TEXT,
            suggestion_html TEXT,
            dxy_impact TEXT,
            src_note TEXT
        )
        """)
        conn.commit()

def init_trades_table():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            direction TEXT NOT NULL,       -- 'Bullish' or 'Bearish'
            timeframe TEXT NOT NULL,
            entry_time TEXT NOT NULL,      -- UTC ISO
            entry_price REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            atr_entry REAL,
            strength TEXT NOT NULL,        -- 'Strong'/'Medium'
            exit_time TEXT,                -- UTC ISO
            exit_price REAL,
            exit_reason TEXT,              -- 'TP'/'SL'/'MANUAL'/'TIME'
            pnl REAL                       -- signed by direction
        )
        """)
        conn.commit()

@st.cache_data(ttl=5)
def _db_exists():
    return os.path.exists(DB_PATH)

def log_signals(rows, diag_lines):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    src = " | ".join(diag_lines[:1]) if diag_lines else ""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.cursor()
        for r in rows:
            strength = ("Strong" if "Strong" in (r.get("AI Suggestion") or "")
                        else "Medium" if "Medium" in (r.get("AI Suggestion") or "")
                        else "None")
            cur.execute("""
            INSERT INTO signals
            (ts_utc, pair, price, rsi, atr, atr_status, trend, reversal_signal, signal_type,
             indicators, candle_pattern, divergence, strength, suggestion_html, dxy_impact, src_note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ts, r.get("Pair"), float(r.get("Price", 0)), float(r.get("RSI", 0)), float(r.get("ATR", 0)),
                r.get("ATR Status"), r.get("Trend"), r.get("Reversal Signal"), r.get("Signal Type"),
                r.get("Confirmed Indicators"), r.get("Candle Pattern"), r.get("Divergence"),
                strength, r.get("AI Suggestion"), r.get("DXY Impact"), src
            ))
        conn.commit()

def _now_utc():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def get_open_trade(pair: str):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM trades WHERE pair=? AND exit_time IS NULL ORDER BY id DESC LIMIT 1", (pair,))
        row = cur.fetchone()
        cols = [d[0] for d in cur.description] if cur.description else []
        return dict(zip(cols, row)) if row else None

def open_trade(pair: str, direction: str, timeframe: str, entry_price: float,
               sl: float, tp: float, atr_entry: float, strength: str):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
        INSERT INTO trades (pair,direction,timeframe,entry_time,entry_price,sl,tp,atr_entry,strength)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, (pair, direction, timeframe, _now_utc(), entry_price, sl, tp, atr_entry, strength))
        conn.commit()

def close_trade(trade_id: int, exit_price: float, reason: str, direction: str):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT entry_price FROM trades WHERE id=?", (trade_id,))
        row = cur.fetchone()
        entry = float(row[0]) if row else 0.0
        pnl = (exit_price - entry) if direction == "Bullish" else (entry - exit_price)
        conn.execute("""
        UPDATE trades
        SET exit_time=?, exit_price=?, exit_reason=?, pnl=?
        WHERE id=?
        """, (_now_utc(), exit_price, reason, pnl, trade_id))
        conn.commit()

def maybe_enter_or_exit(pair_label: str, timeframe: str, price: float, atr: float,
                        signal_type: str, strength: str,
                        use_strength=("Strong", "Medium"), sl_mult=1.2, tp_mult=2.5):
    """Simple paper-trading: one open trade per pair. SL/TP on refresh."""
    # Check open trade
    ot = get_open_trade(pair_label)
    if ot:
        if ot["direction"] == "Bullish":
            if price <= ot["sl"]:
                close_trade(ot["id"], price, "SL", ot["direction"])
                return "closed:SL"
            if price >= ot["tp"]:
                close_trade(ot["id"], price, "TP", ot["direction"])
                return "closed:TP"
        else:  # Bearish
            if price >= ot["sl"]:
                close_trade(ot["id"], price, "SL", ot["direction"])
                return "closed:SL"
            if price <= ot["tp"]:
                close_trade(ot["id"], price, "TP", ot["direction"])
                return "closed:TP"
        return "holding"

    # Consider a new entry
    if signal_type in ("Bullish", "Bearish") and strength in use_strength:
        sl = price - sl_mult * atr if signal_type == "Bullish" else price + sl_mult * atr
        tp = price + tp_mult * atr if signal_type == "Bullish" else price - tp_mult * atr
        open_trade(pair_label, signal_type, timeframe, price, sl, tp, atr, strength)
        return f"opened:{signal_type}"

    return "flat"

# Initialize DBs
init_db()
init_trades_table()

# =========================
# UI / ALERT UTILS
# =========================
def play_beep_once_per_window(seconds=30):
    now = datetime.utcnow()
    last = st.session_state.get("last_beep", datetime.min)
    if (now - last).total_seconds() > seconds:
        components.html(
            """
            <audio autoplay>
                <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
            </audio>
            """,
            height=0,
        )
        st.session_state["last_beep"] = now

# =========================
# NEWS (optional helper)
# =========================
@st.cache_data(ttl=180, show_spinner=False)
def fetch_forex_factory_news_today():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    today_malta = datetime.now(MTZ).date()
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return []
    out = []
    for item in root.findall("./channel/item"):
        try:
            title = item.find("title").text
            pub_time = date_parser.parse(item.find("pubDate").text).astimezone(MTZ)
            currency = item.find("{http://www.forexfactory.com/rss}currency").text.strip().upper()
            if pub_time.date() == today_malta:
                out.append({"title": title, "time": pub_time, "currency": currency})
        except Exception:
            continue
    return out

# =========================
# DATA SOURCE (yfinance resilient)
# =========================
PERIOD_OPTIONS_BY_INTERVAL = {
    "5m":  ["7d", "10d", "30d", "60d"],
    "15m": ["30d", "60d", "90d"],
    "30m": ["60d", "90d", "180d"],
    "1h":  ["90d", "180d", "1y"],
    "4h":  ["2y", "5y"],
    "1d":  ["1y", "2y", "5y", "10y"],
}
ALT_TICKERS = {
    "XAU/USD": ["GC=F", "XAUUSD=X"],
    "XAG/USD": ["SI=F", "XAGUSD=X"],
    "WTI/USD": ["CL=F"],
}

def _try_download(ticker, interval, period):
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval,
                         auto_adjust=False, progress=False, group_by="ticker")
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            first = df.columns.levels[0][0]
            df = df[first]
        df = df.rename(columns=str.lower).reset_index()
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})
        if "datetime" not in df.columns:
            return pd.DataFrame()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").set_index("datetime")
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]
    except Exception:
        return pd.DataFrame()

def _try_history(ticker, interval, period):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower).reset_index()
        if "Date" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        if "Datetime" in df.columns and "datetime" not in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        if "datetime" not in df.columns:
            return pd.DataFrame()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").set_index("datetime")
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=180, show_spinner=False)
def fetch_yf_data_resilient(label: str, yf_symbol: str, interval: str) -> pd.DataFrame:
    periods = PERIOD_OPTIONS_BY_INTERVAL.get(interval, ["60d"])
    candidates = ALT_TICKERS.get(label, [yf_symbol])
    for ticker in candidates:
        for period in periods:
            df = _try_download(ticker, interval, period)
            if not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
                df.attrs["src"] = f"download {ticker} {interval}/{period}"
                return df
            df = _try_history(ticker, interval, period)
            if not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
                df.attrs["src"] = f"history {ticker} {interval}/{period}"
                return df
    return pd.DataFrame()

def fetch_dxy_data():
    for ticker in ["^DXY", "DX-Y.NYB"]:
        try:
            hist = yf.download(tickers=ticker, period="1d", interval="5m", progress=False)
            if hist is None or hist.empty:
                continue
            current = float(hist["Close"].iloc[-1])
            previous = float(hist["Close"].iloc[0])
            pct = ((current - previous) / previous) * 100 if previous else 0.0
            return current, pct
        except Exception:
            continue
    return None, None

# =========================
# INDICATORS
# =========================
def calculate_rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_ema(series: pd.Series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_atr(df: pd.DataFrame, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period=14):
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def detect_candle_pattern(df: pd.DataFrame):
    if len(df) < 2:
        return ""
    o, c, h, l = df["open"].iloc[-2:], df["close"].iloc[-2:], df["high"].iloc[-2:], df["low"].iloc[-2:]
    co, cc, ch, cl = o.iloc[-1], c.iloc[-1], h.iloc[-1], l.iloc[-1]
    po, pc = o.iloc[-2], c.iloc[-2]
    body, rng = abs(cc - co), ch - cl
    if rng == 0: return ""
    if body < rng * 0.1: return "Doji"
    if pc < po and cc > co and cc > po and co < pc: return "Bullish Engulfing"
    if pc > po and cc < co and cc < po and co > pc: return "Bearish Engulfing"
    if body < rng * 0.3 and cl < co and cl < cc and (ch - max(co, cc)) < body: return "Hammer"
    if body < rng * 0.3 and ch > co and ch > cc and (min(co, cc) - cl) < body: return "Shooting Star"
    return ""

def detect_trend_reversal(df: pd.DataFrame):
    if len(df) < 3:
        return ""
    e9, e20 = df["EMA9"].iloc[-3:], df["EMA20"].iloc[-3:]
    if e9.iloc[0] < e20.iloc[0] and e9.iloc[1] > e20.iloc[1] and e9.iloc[2] > e20.iloc[2]:
        return "Reversal Confirmed Bullish"
    if e9.iloc[0] > e20.iloc[0] and e9.iloc[1] < e20.iloc[1] and e9.iloc[2] < e20.iloc[2]:
        return "Reversal Confirmed Bearish"
    if e9.iloc[-2] < e20.iloc[-2] and e9.iloc[-1] > e20.iloc[-1]:
        return "Reversal Forming Bullish"
    if e9.iloc[-2] > e20.iloc[-2] and e9.iloc[-1] < e20.iloc[-1]:
        return "Reversal Forming Bearish"
    return ""

def detect_divergence(df: pd.DataFrame):
    closes = df["close"]; rsis = df["RSI"]
    if len(closes) < 10 or len(rsis) < 10: return ""
    window = 5
    pr_low_idx = closes.iloc[-window:].idxmin()
    pr_high_idx = closes.iloc[-window:].idxmax()
    rsi_low_idx = rsis.iloc[-window:].idxmin()
    rsi_high_idx = rsis.iloc[-window:].idxmax()
    last_close = closes.iloc[-1]; last_rsi = rsis.iloc[-1]
    pr_low_close = closes.loc[pr_low_idx]; pr_high_close = closes.loc[pr_high_idx]
    rsi_at_low = rsis.loc[rsi_low_idx]; rsi_at_high = rsis.loc[rsi_high_idx]
    if (pr_low_idx != rsi_low_idx) and (pr_low_close < last_close) and (rsi_at_low > last_rsi): return "Bullish Divergence"
    if (pr_high_idx != rsi_high_idx) and (pr_high_close > last_close) and (rsi_at_high < last_rsi): return "Bearish Divergence"
    return ""

def score_signal(indicators, signal_type):
    score = 0
    if "EMA" in indicators: score += 2
    if "MACD" in indicators: score += 1
    if "RSI" in indicators: score += 1
    if "ADX" in indicators: score += 1
    if "Candle" in indicators: score += 1
    if "Divergence" in indicators: score += 2
    if signal_type in ("Bullish", "Bearish"): score += 1
    if score >= 6: return "Strong"
    if score >= 4: return "Medium"
    return "None"

def generate_ai_suggestion(price, atr, signal_type, strength):
    if strength == "None" or not signal_type: return ""
    sl = price - (atr * 1.2) if signal_type == "Bullish" else price + (atr * 1.2)
    tp = price + (atr * 2.5) if signal_type == "Bullish" else price - (atr * 2.5)
    color = "green" if signal_type == "Bullish" else "red"
    return f"{strength} <span style='color:{color}'>{signal_type}</span> Signal @ {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"

# =========================
# MAIN RUN
# =========================
news_events = fetch_forex_factory_news_today()
dxy_price, dxy_change = fetch_dxy_data()

rows, diag = [], []

for label, yf_symbol in symbols.items():
    df = fetch_yf_data_resilient(label, yf_symbol, timeframe)
    if df.empty or not set(["open", "high", "low", "close"]).issubset(df.columns):
        diag.append(f"❌ {label}: no data for {yf_symbol} at {timeframe}")
        continue
    else:
        diag.append(f"✅ {label}: {df.attrs.get('src','yfinance')}  rows={len(df)}")

    # Indicators
    df["RSI"] = calculate_rsi(df["close"])
    macd, macd_sig = calculate_macd(df["close"])
    df["MACD"], df["MACD_Signal"] = macd, macd_sig
    df["EMA9"] = calculate_ema(df["close"], 9)
    df["EMA20"] = calculate_ema(df["close"], 20)
    df["ATR"] = calculate_atr(df)
    df["ADX"] = calculate_adx(df)

    # Clean / warm-up cut
    df = df.dropna().copy()
    min_required = 50 if timeframe in ["5m", "15m", "30m", "1h"] else 30
    if len(df) < min_required:
        diag.append(f"ℹ️  {label}: only {len(df)} bars after indicators; need >= {min_required}")
        continue

    price = float(df["close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])

    # Trend
    if (df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1]) and (price > df["EMA9"].iloc[-1]):
        trend = "Bullish"
    elif (df["EMA9"].iloc[-1] < df["EMA20"].iloc[-1]) and (price < df["EMA9"].iloc[-1]):
        trend = "Bearish"
    else:
        trend = "Sideways"

    # Signals
    rsi_val = float(df["RSI"].iloc[-1])
    indicators = []
    signal_type = ""
    if rsi_val >= 55:
        indicators.append("RSI"); signal_type = "Bullish"
    elif rsi_val <= 45:
        indicators.append("RSI"); signal_type = "Bearish"

    if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
        indicators.append("MACD")
    if (df["EMA9"].iloc[-1] > df["EMA20"].iloc[-1]) and (price > df["EMA9"].iloc[-1]):
        indicators.append("EMA")
    if df["ADX"].iloc[-1] > 20:
        indicators.append("ADX")

    pattern = detect_candle_pattern(df)
    if pattern:
        indicators.append("Candle")

    divergence = detect_divergence(df)
    if divergence:
        indicators.append("Divergence")
        play_beep_once_per_window(30)

    strength = score_signal(indicators, signal_type)
    suggestion = generate_ai_suggestion(price, atr, signal_type, strength)

    # Paper trading: entries/exits
    paper_status = maybe_enter_or_exit(
        pair_label=label,
        timeframe=timeframe,
        price=price,
        atr=atr,
        signal_type=signal_type,
        strength=strength,
        use_strength=tuple(trade_strength) if trade_strength else ("Strong",),
        sl_mult=float(sl_mult),
        tp_mult=float(tp_mult)
    )

    # DXY impact for USD pairs
    dxy_txt = "—"
    if "USD" in label and (dxy_price is not None) and (dxy_change is not None):
        sign = "+" if dxy_change >= 0 else ""
        dxy_txt = f"{dxy_price:.2f} ({sign}{dxy_change:.2f}%)"

    # ATR status
    atr_status = "🔴 Low" if atr < 0.0004 else "🟡 Normal" if atr < 0.0009 else "🟢 High"

    rows.append({
        "Pair": label,
        "Price": round(price, 5),
        "RSI": round(rsi_val, 2),
        "ATR": round(atr, 5),
        "ATR Status": atr_status,
        "Trend": trend,
        "Reversal Signal": detect_trend_reversal(df) or "—",
        "Signal Type": signal_type or "—",
        "Confirmed Indicators": ", ".join(indicators) if indicators else "—",
        "Candle Pattern": pattern or "—",
        "Divergence": divergence or "—",
        "AI Suggestion": suggestion or "",
        "DXY Impact": dxy_txt,
        "Paper Status": paper_status,
    })

# =========================
# RENDER + LOGGING CONTROLS
# =========================
colA, colB, colC = st.columns([1,1,3])
with colA:
    enable_log = st.toggle("💾 Auto-log this scan", value=False,
                           help="When ON, each refresh appends the current table to the DB.")
with colB:
    if st.button("Append to DB now") and rows:
        log_signals(rows, diag)
        st.success(f"Logged {len(rows)} rows to {DB_PATH}")
with colC:
    if _db_exists() and st.button("🔍 Preview last 10 logs"):
        with closing(sqlite3.connect(DB_PATH)) as conn:
            df_prev = pd.read_sql_query(
                "SELECT ts_utc,pair,price,signal_type,strength,trend,indicators FROM signals ORDER BY id DESC LIMIT 10", conn)
        st.dataframe(df_prev, use_container_width=True)

if enable_log and rows:
    log_signals(rows, diag)

if not rows:
    st.error("No rows produced. See diagnostics below.")
    if diag:
        with st.expander("Diagnostics"):
            for line in diag:
                st.write(line)
    st.stop()

df_result = pd.DataFrame(rows)

# Strong/Medium sorting
score_series = np.select(
    [
        df_result["AI Suggestion"].fillna("").str.contains("Strong"),
        df_result["AI Suggestion"].fillna("").str.contains("Medium"),
    ],
    [3, 2],
    default=0,
)
df_result["Score"] = score_series
df_sorted = df_result.sort_values(by="Score", ascending=False).drop(columns=["Score"])

column_order = [
    "Pair", "Price", "RSI", "ATR", "ATR Status", "Trend", "Reversal Signal",
    "Signal Type", "Confirmed Indicators", "Candle Pattern", "Divergence",
    "AI Suggestion", "DXY Impact", "Paper Status"
]

styled_html = "<div style='overflow-x:auto'>"
styled_html += "<table style='width:100%; border-collapse: collapse;'>"
styled_html += "<tr>" + "".join([
    f"<th style='border:1px solid #ccc; padding:6px; background:#e0e0e0; position:sticky; top:0'>{col}</th>"
    for col in column_order
]) + "</tr>"

for _, row in df_sorted.iterrows():
    style = 'background-color: #d4edda;' if "Strong" in row["AI Suggestion"] else \
            'background-color: #d1ecf1;' if "Medium" in row["AI Suggestion"] else ''
    styled_html += f"<tr style='{style}'>"
    for col in column_order:
        val = row.get(col, "—")
        if col == "Pair":
            val = f"<strong style='font-size: 18px;'>{val}</strong>"
        elif col == "Trend":
            color = 'green' if row['Trend'] == 'Bullish' else 'red' if row['Trend'] == 'Bearish' else 'gray'
            val = f"<span style='color:{color}; font-weight:bold;'>{row['Trend']}</span>"
        elif col == "Signal Type":
            color = 'green' if row['Signal Type'] == 'Bullish' else 'red' if row['Signal Type'] == 'Bearish' else 'gray'
            val = f"<span style='color:{color}; font-weight:bold;'>{row['Signal Type']}</span>"
        elif col == "RSI":
            color = "red" if row["RSI"] > 75 else "green" if row["RSI"] < 20 else "black"
            val = f"<span style='color:{color}; font-weight:bold;'>{val}</span>"
        elif col == "DXY Impact" and row["DXY Impact"] != "—":
            dxy_color = "green" if "+" in str(row["DXY Impact"]) else "red"
            val = f"<span style='color:{dxy_color}; font-weight:bold;'>{row['DXY Impact']}</span>"
        elif col == "Divergence" and row["Divergence"] != "—":
            div_color = "green" if "Bullish" in row["Divergence"] else "red"
            val = f"<span style='color:{div_color}; font-weight:bold;'>{row['Divergence']}</span>"
        styled_html += f"<td style='border:1px solid #ccc; padding:6px; white-space:pre-wrap;'>{val}</td>"
    styled_html += "</tr>"

styled_html += "</table></div>"
st.markdown(styled_html, unsafe_allow_html=True)

tz_label = datetime.now(MTZ).strftime('%Y-%m-%d %H:%M:%S %Z')
st.caption(f"Timeframe: {timeframe} | Last updated: {tz_label}")
st.text(f"Scanned Pairs: {len(rows)}")
st.text(f"Strong Signals Found: {sum('Strong' in (r.get('AI Suggestion') or '') for r in rows)}")

with st.expander("Data diagnostics"):
    for line in diag:
        st.write(line)

# =========================
# PAPER TRADES VIEW + METRICS
# =========================
with st.expander("📒 Paper trades (DB)"):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        df_open = pd.read_sql_query(
            "SELECT id,pair,direction,timeframe,entry_time,entry_price,sl,tp,strength FROM trades WHERE exit_time IS NULL ORDER BY id DESC",
            conn
        )
        df_closed = pd.read_sql_query(
            "SELECT id,pair,direction,timeframe,entry_time,entry_price,exit_time,exit_price,exit_reason,pnl,strength FROM trades WHERE exit_time IS NOT NULL ORDER BY id DESC",
            conn
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Open positions", len(df_open))
    with c2:
        wins = (df_closed["pnl"] > 0).sum() if not df_closed.empty else 0
        total = len(df_closed)
        winrate = round(100*wins/total, 2) if total else 0.0
        st.metric("Win rate", f"{winrate}%")
    with c3:
        total_pnl = round(df_closed["pnl"].sum(), 5) if not df_closed.empty else 0.0
        st.metric("Total PnL", total_pnl)
    st.write("**Open trades**")
    st.dataframe(df_open, use_container_width=True)
    st.write("**Closed trades**")
    st.dataframe(df_closed, use_container_width=True)

# =========================
# BACKTEST (same rules on selected timeframe)
# =========================
def generate_signals_over_history(df):
    out = df.copy()
    out["RSI"] = calculate_rsi(out["close"])
    macd, macd_sig = calculate_macd(out["close"])
    out["MACD"], out["MACD_Signal"] = macd, macd_sig
    out["EMA9"] = calculate_ema(out["close"], 9)
    out["EMA20"] = calculate_ema(out["close"], 20)
    out["ATR"] = calculate_atr(out)
    out["ADX"] = calculate_adx(out)
    out = out.dropna().copy()

    cond_trend_bull = (out["EMA9"] > out["EMA20"]) & (out["close"] > out["EMA9"])
    cond_trend_bear = (out["EMA9"] < out["EMA20"]) & (out["close"] < out["EMA9"])
    rsi_bull = out["RSI"] >= 55
    rsi_bear = out["RSI"] <= 45
    macd_bull = out["MACD"] > out["MACD_Signal"]
    adx_trend = out["ADX"] > 20

    out["ind_EMA"] = cond_trend_bull | cond_trend_bear
    out["ind_MACD"] = macd_bull
    out["ind_RSI"] = rsi_bull | rsi_bear
    out["ind_ADX"] = adx_trend

    out["score"] = (
        out["ind_EMA"].astype(int)*2 +
        out["ind_MACD"].astype(int) +
        out["ind_RSI"].astype(int) +
        out["ind_ADX"].astype(int)
    )
    out["signal_type"] = np.where(rsi_bull, "Bullish", np.where(rsi_bear, "Bearish", ""))
    out["trend"] = np.where(cond_trend_bull, "Bullish", np.where(cond_trend_bear, "Bearish", "Sideways"))
    out["strength"] = np.where(out["score"] >= 6, "Strong", np.where(out["score"] >= 4, "Medium", "None"))
    return out

def backtest_pair(yf_symbol, label, interval, sl_mult=1.2, tp_mult=2.5, max_hold_bars=96, use_strength=("Strong","Medium")):
    df = fetch_yf_data_resilient(label, yf_symbol, interval)
    if df.empty:
        return pd.DataFrame(), {"error": f"No data for {label} at {interval}"}
    sig = generate_signals_over_history(df)
    if sig.empty:
        return pd.DataFrame(), {"error": "No usable bars after indicators."}

    trades = []
    in_trade = False
    entry_price = entry_time = direction = None
    atr_entry = None
    prev_dir = ""
    bars_held = 0

    for t, row in sig.iterrows():
        dir_now = row["signal_type"]
        if not in_trade:
            if dir_now and row["strength"] in use_strength and dir_now != prev_dir:
                in_trade = True
                entry_price = row["close"]; entry_time = t; direction = dir_now; atr_entry = row["ATR"]
                sl = entry_price - sl_mult*atr_entry if direction == "Bullish" else entry_price + sl_mult*atr_entry
                tp = entry_price + tp_mult*atr_entry if direction == "Bullish" else entry_price - tp_mult*atr_entry
                bars_held = 0
        else:
            bars_held += 1
            price = row["close"]
            hit_tp = (price >= tp) if direction == "Bullish" else (price <= tp)
            hit_sl = (price <= sl) if direction == "Bullish" else (price >= sl)
            exit_now = hit_tp or hit_sl or (bars_held >= max_hold_bars)
            if exit_now:
                pnl = (price - entry_price) if direction == "Bullish" else (entry_price - price)
                trades.append({
                    "pair": label, "entry_time": entry_time, "exit_time": t,
                    "direction": direction, "entry": entry_price, "exit": price,
                    "pnl": pnl, "bars": bars_held, "hit": "TP" if hit_tp else ("SL" if hit_sl else "TIME")
                })
                in_trade = False
                entry_price = entry_time = direction = atr_entry = None
        prev_dir = dir_now

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"trades": 0, "winrate": 0, "avg_pnl": 0, "total_pnl": 0}
    total_pnl = trades_df["pnl"].sum()
    winrate = (trades_df["pnl"] > 0).mean()*100
    stats = {
        "trades": len(trades_df),
        "winrate": round(winrate, 2),
        "avg_pnl": round(trades_df["pnl"].mean(), 5),
        "median_pnl": round(trades_df["pnl"].median(), 5),
        "total_pnl": round(total_pnl, 5),
        "tp_rate": round((trades_df["hit"]=="TP").mean()*100, 2),
        "sl_rate": round((trades_df["hit"]=="SL").mean()*100, 2),
    }
    return trades_df, stats

with st.expander("🧪 Backtest (replays current rules on selected timeframe)"):
    bt_pair = st.selectbox("Pair to backtest", list(symbols.keys()))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bt_sl_mult = st.number_input("BT SL = ATR ×", min_value=0.5, max_value=5.0, value=float(sl_mult), step=0.1)
    with c2:
        bt_tp_mult = st.number_input("BT TP = ATR ×", min_value=0.5, max_value=10.0, value=float(tp_mult), step=0.1)
    with c3:
        max_hold = st.number_input("Max bars per trade", min_value=12, max_value=500, value=96, step=6)
    with c4:
        use_str = st.multiselect("Use signals", ["Strong","Medium"], default=trade_strength or ["Strong"])

    if st.button("Run backtest"):
        trades_df, stats = backtest_pair(
            symbols[bt_pair], bt_pair, timeframe,
            sl_mult=float(bt_sl_mult), tp_mult=float(bt_tp_mult),
            max_hold_bars=int(max_hold),
            use_strength=tuple(use_str) if use_str else ("Strong","Medium")
        )
        if "error" in stats:
            st.warning(stats["error"])
        else:
            st.write(f"**Trades:** {stats['trades']}  |  **Winrate:** {stats['winrate']}%  |  "
                     f"**TP rate:** {stats['tp_rate']}%  |  **SL rate:** {stats['sl_rate']}%  |  "
                     f"**Total PnL:** {stats['total_pnl']}")
            st.dataframe(trades_df, use_container_width=True)
