#!/usr/bin/env python3
import sqlite3, numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
from contextlib import closing

DB_PATH = "signals.db"

SYMBOLS = {
    "EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"USDJPY=X","AUD/USD":"AUDUSD=X",
    "USD/CAD":"USDCAD=X","USD/CHF":"USDCHF=X","NZD/USD":"NZDUSD=X","EUR/JPY":"EURJPY=X",
    "EUR/GBP":"EURGBP=X","EUR/CAD":"EURCAD=X","GBP/JPY":"GBPJPY=X","EUR/AUD":"EURAUD=X",
    "AUD/JPY":"AUDJPY=X","GBP/NZD":"GBPNZD=X","EUR/NZD":"EURNZD=X",
    "XAU/USD":"GC=F","XAG/USD":"SI=F","WTI/USD":"CL=F",
}
ALT_TICKERS = {"XAU/USD":["GC=F","XAUUSD=X"], "XAG/USD":["SI=F","XAGUSD=X"], "WTI/USD":["CL=F"]}
PERIODS_BY_INTERVAL = {"5m":["7d","10d","30d","60d"]}
INTERVAL = "5m"  # keep it light for cron

def _try_download(t, interval, period):
    try:
        df = yf.download(tickers=t, period=period, interval=interval, progress=False, auto_adjust=False, group_by="ticker")
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df = df[df.columns.levels[0][0]]
        df = df.rename(columns=str.lower).reset_index().rename(columns={"Datetime":"datetime","Date":"datetime"})
        if "datetime" not in df.columns: return pd.DataFrame()
        df["datetime"]=pd.to_datetime(df["datetime"], utc=True); df=df.sort_values("datetime").set_index("datetime")
        keep=[c for c in ["open","high","low","close","volume"] if c in df.columns]; return df[keep]
    except Exception: return pd.DataFrame()

def _try_history(t, interval, period):
    try:
        h = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=False)
        if h is None or h.empty: return pd.DataFrame()
        df=h.rename(columns=str.lower).reset_index().rename(columns={"Datetime":"datetime","Date":"datetime"})
        if "datetime" not in df.columns: return pd.DataFrame()
        df["datetime"]=pd.to_datetime(df["datetime"], utc=True); df=df.sort_values("datetime").set_index("datetime")
        keep=[c for c in ["open","high","low","close","volume"] if c in df.columns]; return df[keep]
    except Exception: return pd.DataFrame()

def fetch_resilient(label, yf_symbol, interval):
    periods = PERIODS_BY_INTERVAL.get(interval, ["60d"])
    cands = ALT_TICKERS.get(label,[yf_symbol])
    for t in cands:
        for per in periods:
            for fn in (_try_download,_try_history):
                df = fn(t, interval, per)
                if not df.empty and {"open","high","low","close"}.issubset(df.columns):
                    return df
    return pd.DataFrame()

# Indicators
def rsi(s, period=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / dn.ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100/(1+rs))
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def macd(s): m = ema(s,12)-ema(s,26); sig = m.ewm(span=9, adjust=False).mean(); return m, sig
def atr(df, n=14):
    tr = pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def adx(df, n=14):
    up = df.high.diff(); dn = -df.low.diff()
    pdm = np.where((up>dn)&(up>0), up, 0.0); mdm = np.where((dn>up)&(dn>0), dn, 0.0)
    tr = pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/atrv
    mdi = 100*pd.Series(mdm, index=df.index).ewm(alpha=1/n, adjust=False).mean()/atrv
    dx = (100*(pdi-mdi).abs()/(pdi+mdi)).replace([np.inf,-np.inf], np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

# DB
def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_utc TEXT NOT NULL, pair TEXT NOT NULL, price REAL NOT NULL,
          rsi REAL, atr REAL, atr_status TEXT, trend TEXT, reversal_signal TEXT,
          signal_type TEXT, indicators TEXT, candle_pattern TEXT, divergence TEXT,
          strength TEXT, suggestion_html TEXT, dxy_impact TEXT, src_note TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          pair TEXT NOT NULL, direction TEXT NOT NULL, timeframe TEXT NOT NULL,
          entry_time TEXT NOT NULL, entry_price REAL NOT NULL, sl REAL NOT NULL, tp REAL NOT NULL,
          atr_entry REAL, strength TEXT NOT NULL,
          exit_time TEXT, exit_price REAL, exit_reason TEXT, pnl REAL
        )""")
        conn.commit()

def log_rows(ts, rows, src_note="collector"):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur=conn.cursor()
        for r in rows:
            strength = r["Strength"]
            cur.execute("""INSERT INTO signals
            (ts_utc,pair,price,rsi,atr,atr_status,trend,reversal_signal,signal_type,
             indicators,candle_pattern,divergence,strength,suggestion_html,dxy_impact,src_note)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(
                ts, r["Pair"], r["Price"], r["RSI"], r["ATR"], r["ATR Status"], r["Trend"],
                r["Reversal Signal"], r["Signal Type"], r["Confirmed Indicators"], r["Candle Pattern"],
                r["Divergence"], strength, r["AI Suggestion"], r["DXY Impact"], src_note
            ))
        conn.commit()

def _now_utc(): return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
def get_open_trade(pair):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur=conn.cursor(); cur.execute("SELECT * FROM trades WHERE pair=? AND exit_time IS NULL ORDER BY id DESC LIMIT 1",(pair,))
        row=cur.fetchone(); cols=[d[0] for d in cur.description] if cur.description else []; return dict(zip(cols,row)) if row else None
def open_trade(pair, direction, timeframe, price, sl, tp, atr_entry, strength):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""INSERT INTO trades (pair,direction,timeframe,entry_time,entry_price,sl,tp,atr_entry,strength)
                        VALUES (?,?,?,?,?,?,?,?,?)""", (pair,direction,INTERVAL,_now_utc(),price,sl,tp,atr_entry,strength)); conn.commit()
def close_trade(tid, exit_price, reason, direction):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur=conn.cursor(); cur.execute("SELECT entry_price FROM trades WHERE id=?", (tid,)); row=cur.fetchone()
        entry=float(row[0]) if row else 0.0
        pnl=(exit_price-entry) if direction=="Bullish" else (entry-exit_price)
        conn.execute("""UPDATE trades SET exit_time=?, exit_price=?, exit_reason=?, pnl=? WHERE id=?""",
                     (_now_utc(), exit_price, reason, pnl, tid)); conn.commit()

def maybe_enter_or_exit(pair, price, atr, signal_type, strength, use_strength=("Strong","Medium"), sl_mult=1.2, tp_mult=2.5):
    ot=get_open_trade(pair)
    if ot:
        if ot["direction"]=="Bullish":
            if price<=ot["sl"]: close_trade(ot["id"], price, "SL", ot["direction"]); return "closed:SL"
            if price>=ot["tp"]: close_trade(ot["id"], price, "TP", ot["direction"]); return "closed:TP"
        else:
            if price>=ot["sl"]: close_trade(ot["id"], price, "SL", ot["direction"]); return "closed:SL"
            if price<=ot["tp"]: close_trade(ot["id"], price, "TP", ot["direction"]); return "closed:TP"
        return "holding"
    if signal_type in ("Bullish","Bearish") and strength in use_strength:
        sl = price - sl_mult*atr if signal_type=="Bullish" else price + sl_mult*atr
        tp = price + tp_mult*atr if signal_type=="Bullish" else price - tp_mult*atr
        open_trade(pair, signal_type, INTERVAL, price, sl, tp, atr, strength)
        return f"opened:{signal_type}"
    return "flat"

def run_once():
    init_db()
    rows=[]
    for label, yfs in SYMBOLS.items():
        df = fetch_resilient(label, yfs, INTERVAL)
        if df.empty: continue
        df["RSI"]=rsi(df.close); m,ms = macd(df.close); df["MACD"]=m; df["MACD_Signal"]=ms
        df["EMA9"]=ema(df.close,9); df["EMA20"]=ema(df.close,20)
        df["ATR"]=atr(df); df["ADX"]=adx(df); df=df.dropna()
        if len(df)<50: continue

        price=float(df.close.iloc[-1]); rsi_v=float(df.RSI.iloc[-1]); atr_v=float(df.ATR.iloc[-1])
        trend = "Bullish" if (df.EMA9.iloc[-1]>df.EMA20.iloc[-1] and price>df.EMA9.iloc[-1]) else \
                "Bearish" if (df.EMA9.iloc[-1]<df.EMA20.iloc[-1] and price<df.EMA9.iloc[-1]) else "Sideways"
        indicators=[]; signal=""
        if rsi_v>=55: indicators.append("RSI"); signal="Bullish"
        elif rsi_v<=45: indicators.append("RSI"); signal="Bearish"
        if df.MACD.iloc[-1]>df.MACD_Signal.iloc[-1]: indicators.append("MACD")
        if (df.EMA9.iloc[-1]>df.EMA20.iloc[-1]) and (price>df.EMA9.iloc[-1]): indicators.append("EMA")
        if df.ADX.iloc[-1]>20: indicators.append("ADX")
        score = (2*("EMA" in indicators)) + ("MACD" in indicators) + ("RSI" in indicators) + ("ADX" in indicators) + 0
        strength = "Strong" if score>=6 else "Medium" if score>=4 else "None"
        sugg = "" if strength=="None" else (
            f"{strength} <span style='color:{'green' if signal=='Bullish' else 'red'}'>{signal}</span> "
            f"@ {price:.5f} | SL: {(price-(1.2*atr_v) if signal=='Bullish' else price+(1.2*atr_v)):.5f} "
            f"| TP: {(price+(2.5*atr_v) if signal=='Bullish' else price-(2.5*atr_v)):.5f}"
        )
        atr_status = "🔴 Low" if atr_v<0.0004 else "🟡 Normal" if atr_v<0.0009 else "🟢 High"

        # update paper trades
        _ = maybe_enter_or_exit(label, price, atr_v, signal, strength, use_strength=("Strong",))

        rows.append({
            "Pair":label,"Price":round(price,5),"RSI":round(rsi_v,2),
            "ATR":round(atr_v,5),"ATR Status":atr_status,"Trend":trend,
            "Reversal Signal":"—","Signal Type":signal or "—",
            "Confirmed Indicators":", ".join(indicators) if indicators else "—",
            "Candle Pattern":"—","Divergence":"—",
            "AI Suggestion":sugg,"DXY Impact":"—","Strength":strength
        })

    if rows:
        ts=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_rows(ts, rows, src_note="collector.py")
        print(f"[{ts}] Logged {len(rows)} rows and updated paper trades.")
    else:
        print("No rows to log.")

if __name__ == "__main__":
    run_once()
