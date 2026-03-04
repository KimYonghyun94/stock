import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

# -----------------------------
# Optional dependencies
# -----------------------------
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import FinanceDataReader as fdr  # pip name: finance-datareader
    FDR_OK = True
except Exception:
    FDR_OK = False

try:
    from pykrx import stock as krx_stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

try:
    from pandas_datareader import data as pdr  # pip name: pandas-datareader
    PDR_OK = True
except Exception:
    PDR_OK = False


# ============================================================
# Indicator helpers
# ============================================================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.rolling(n).mean()
    al = loss.rolling(n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    upper = mid + k * sd
    lower = mid - k * sd
    return mid, upper, lower

def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    up = high.rolling(n).max()
    dn = low.rolling(n).min()
    return up, dn

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h

def annualized_vol(close: pd.Series, window: int = 20, ann: int = 252) -> pd.Series:
    return close.pct_change().rolling(window).std() * np.sqrt(ann)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ============================================================
# Data fetch (with caching + fallback)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def krx_listing():
    """KRX listing for search by name. Uses FinanceDataReader if available."""
    if not FDR_OK:
        return None
    try:
        df = fdr.StockListing("KRX")
        return df
    except Exception:
        return None

def infer_yahoo_suffix(code6: str, listing_df: pd.DataFrame | None):
    """Heuristic: KOSPI .KS, KOSDAQ .KQ."""
    if listing_df is None:
        return ".KS"
    hit = listing_df[listing_df["Symbol"].astype(str) == str(code6)]
    if len(hit) == 0:
        return ".KS"
    market = str(hit.iloc[0].get("Market", "")).upper()
    if "KOSDAQ" in market:
        return ".KQ"
    return ".KS"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_daily(symbol: str, start: date, end: date, auto_adjust: bool = True) -> pd.DataFrame:
    """Daily OHLCV via yfinance. Returns empty df on failure."""
    if not YF_OK:
        return pd.DataFrame()
    try:
        df = yf.download(
            symbol,
            start=str(start),
            end=str(end + timedelta(days=1)),
            interval="1d",
            auto_adjust=auto_adjust,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename_axis("Date")
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[keep].dropna()
        return out
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq_us_daily(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Stooq fallback for US tickers. Usually uses AAPL.US format.
    Returns empty df on failure.
    """
    if not PDR_OK:
        return pd.DataFrame()
    try:
        sym = ticker if ticker.upper().endswith(".US") else f"{ticker.upper()}.US"
        df = pdr.DataReader(sym, "stooq", start, end).sort_index()
        if df is None or df.empty:
            return pd.DataFrame()
        df.index.name = "Date"
        # stooq columns: Open, High, Low, Close, Volume (often already title case)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fdr_korea_daily(code6: str, start: date, end: date) -> pd.DataFrame:
    if not FDR_OK:
        return pd.DataFrame()
    try:
        df = fdr.DataReader(code6, str(start), str(end))
        if df is None or df.empty:
            return pd.DataFrame()
        df.index.name = "Date"
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pykrx_daily(code6: str, start: date, end: date, adjusted: bool = True) -> pd.DataFrame:
    if not PYKRX_OK:
        return pd.DataFrame()
    try:
        df = krx_stock.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            code6,
            adjusted=adjusted
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


def fetch_us(ticker: str, start: date, end: date, auto_adjust: bool, prefer: str = "yfinance") -> tuple[pd.DataFrame, str]:
    """
    prefer: 'yfinance' or 'stooq'
    returns (df, source_used)
    """
    ticker = ticker.strip().upper()
    if prefer == "stooq":
        df = fetch_stooq_us_daily(ticker, start, end)
        if not df.empty:
            return df, "stooq"
        df = fetch_yfinance_daily(ticker, start, end, auto_adjust=auto_adjust)
        if not df.empty:
            return df, "yfinance"
        return pd.DataFrame(), "none"

    # prefer yfinance
    df = fetch_yfinance_daily(ticker, start, end, auto_adjust=auto_adjust)
    if not df.empty:
        return df, "yfinance"

    # fallback
    df2 = fetch_stooq_us_daily(ticker, start, end)
    if not df2.empty:
        return df2, "stooq"

    return pd.DataFrame(), "none"


def fetch_kr(code6: str, start: date, end: date, adjusted: bool, source: str, listing_df: pd.DataFrame | None):
    """
    source: 'pykrx' | 'fdr' | 'yfinance'
    returns (df, source_used, symbol_hint)
    """
    code6 = code6.strip()
    if source == "pykrx":
        df = fetch_pykrx_daily(code6, start, end, adjusted=adjusted)
        return df, "pykrx", code6
    if source == "fdr":
        df = fetch_fdr_korea_daily(code6, start, end)
        return df, "fdr", code6
    # yfinance
    suffix = infer_yahoo_suffix(code6, listing_df)
    sym = f"{code6}{suffix}"
    df = fetch_yfinance_daily(sym, start, end, auto_adjust=adjusted)
    return df, "yfinance", sym


# ============================================================
# Backtest engine (long/flat, close-to-close, simplified)
# ============================================================
def positions_to_trades(close: pd.Series, pos: pd.Series) -> pd.DataFrame:
    """
    Create a trades table from a 0/1 position series.
    Assumption: entry/exit at close (simplified).
    """
    pos = pos.fillna(0).astype(float)
    d = pos.diff().fillna(pos)

    entries = d[d > 0].index
    exits = d[d < 0].index

    trades = []
    j = 0
    for entry_dt in entries:
        # first exit after entry
        while j < len(exits) and exits[j] <= entry_dt:
            j += 1
        exit_dt = exits[j] if j < len(exits) else close.index[-1]
        if j < len(exits):
            j += 1

        entry_px = safe_float(close.loc[entry_dt])
        exit_px = safe_float(close.loc[exit_dt])
        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px == 0:
            continue
        ret = exit_px / entry_px - 1.0
        trades.append([entry_dt, exit_dt, entry_px, exit_px, ret])

    if not trades:
        return pd.DataFrame(columns=["Entry", "Exit", "EntryPx", "ExitPx", "Return", "Days"])

    tdf = pd.DataFrame(trades, columns=["Entry", "Exit", "EntryPx", "ExitPx", "Return"])
    tdf["Days"] = (tdf["Exit"] - tdf["Entry"]).dt.days
    return tdf


def compute_metrics(equity: pd.Series, strat_ret: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    strat_ret = strat_ret.dropna()

    if len(equity) < 2:
        return {}

    ann = 252.0
    n = len(strat_ret)
    years = n / ann if n > 0 else np.nan

    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if np.isfinite(years) and years > 0 else np.nan

    mu = strat_ret.mean()
    sd = strat_ret.std(ddof=0)
    sharpe = float(np.sqrt(ann) * mu / sd) if sd and np.isfinite(sd) else np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min()) if len(dd) else np.nan

    if trades is not None and len(trades) > 0:
        win_rate = float((trades["Return"] > 0).mean())
        avg_trade = float(trades["Return"].mean())
        n_trades = int(len(trades))
    else:
        win_rate = np.nan
        avg_trade = np.nan
        n_trades = 0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": mdd,
        "Trades": n_trades,
        "Win Rate": win_rate,
        "Avg Trade Return": avg_trade,
    }


def run_backtest(df: pd.DataFrame, pos: pd.Series, fee_bps: float = 5.0, slippage_bps: float = 0.0):
    """
    Simplified backtest:
    - Use pos.shift(1) to avoid look-ahead (signal at t-1 close => hold during t)
    - Costs charged on abs(position change)
    """
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos = pos.reindex(df.index).fillna(0.0).astype(float)
    held = pos.shift(1).fillna(0.0)

    gross = held * ret

    dpos = pos.diff().fillna(pos)
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = cost_rate * dpos.abs()

    strat_ret = gross - costs
    equity = (1.0 + strat_ret).cumprod()

    trades = positions_to_trades(close, pos)
    metrics = compute_metrics(equity, strat_ret, trades)

    return equity, strat_ret, trades, metrics


# ============================================================
# Strategies (all long/flat)
# ============================================================
def strat_buyhold(df, **params):
    return pd.Series(1.0, index=df.index)

def strat_sma_cross(df, fast=20, slow=60, **params):
    c = df["Close"].astype(float)
    f = sma(c, fast)
    s = sma(c, slow)
    return (f > s).astype(float)

def strat_ema_cross(df, fast=12, slow=26, **params):
    c = df["Close"].astype(float)
    f = ema(c, fast)
    s = ema(c, slow)
    return (f > s).astype(float)

def strat_macd_trend(df, fast=12, slow=26, signal=9, **params):
    c = df["Close"].astype(float)
    m, s, _ = macd(c, fast, slow, signal)
    return (m > s).astype(float)

def strat_rsi_reversion(df, n=14, low=30, high=70, **params):
    c = df["Close"].astype(float)
    rr = rsi(c, n)
    entry = (rr < low).astype(int)
    exit_ = (rr > high).astype(int)

    pos = pd.Series(0, index=df.index, dtype=float)
    in_pos = 0
    for t in df.index:
        if in_pos == 0 and entry.loc[t] == 1:
            in_pos = 1
        elif in_pos == 1 and exit_.loc[t] == 1:
            in_pos = 0
        pos.loc[t] = float(in_pos)
    return pos

def strat_bollinger_reversion(df, n=20, k=2.0, **params):
    c = df["Close"].astype(float)
    mid, up, lo = bollinger(c, n, k)
    entry = (c < lo).astype(int)
    exit_ = (c > mid).astype(int)

    pos = pd.Series(0, index=df.index, dtype=float)
    in_pos = 0
    for t in df.index:
        if in_pos == 0 and entry.loc[t] == 1:
            in_pos = 1
        elif in_pos == 1 and exit_.loc[t] == 1:
            in_pos = 0
        pos.loc[t] = float(in_pos)
    return pos

def strat_donchian_breakout(df, n=20, **params):
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    up, dn = donchian(h, l, n)

    entry = (c > up.shift(1)).astype(int)
    exit_ = (c < dn.shift(1)).astype(int)

    pos = pd.Series(0, index=df.index, dtype=float)
    in_pos = 0
    for t in df.index:
        if in_pos == 0 and entry.loc[t] == 1:
            in_pos = 1
        elif in_pos == 1 and exit_.loc[t] == 1:
            in_pos = 0
        pos.loc[t] = float(in_pos)
    return pos

def strat_momentum_filter(df, lookback=120, sma_filter=200, **params):
    """
    Simple time-series momentum:
    - long if lookback return > 0 AND close > SMA(sma_filter)
    """
    c = df["Close"].astype(float)
    mom = c / c.shift(lookback) - 1.0
    filt = c > sma(c, sma_filter)
    return ((mom > 0) & filt).astype(float)

def strat_vol_target_trend(df, fast=20, slow=60, vol_window=20, vol_cap=0.30, **params):
    """
    Trend signal (SMA cross) + volatility cap:
    - pos=1 if fast>slo
    - reduce to 0 if annualized vol > vol_cap (risk-off)
    """
    c = df["Close"].astype(float)
    signal = (sma(c, fast) > sma(c, slow))
    vol = annualized_vol(c, window=vol_window)
    risk_ok = vol <= vol_cap
    return (signal & risk_ok).astype(float)

STRATEGIES = {
    "Buy & Hold": {
        "fn": strat_buyhold,
        "desc": "항상 보유(기준선).",
        "params": [],
        "defaults": {},
    },
    "SMA Crossover": {
        "fn": strat_sma_cross,
        "desc": "FAST SMA > SLOW SMA일 때 보유(추세).",
        "params": [
            dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
            dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
        ],
        "defaults": {"fast": 20, "slow": 60},
    },
    "EMA Crossover": {
        "fn": strat_ema_cross,
        "desc": "FAST EMA > SLOW EMA일 때 보유(추세).",
        "params": [
            dict(name="fast", label="FAST EMA", kind="int", min=3, max=100, step=1, default=12),
            dict(name="slow", label="SLOW EMA", kind="int", min=5, max=200, step=1, default=26),
        ],
        "defaults": {"fast": 12, "slow": 26},
    },
    "MACD Trend": {
        "fn": strat_macd_trend,
        "desc": "MACD > Signal일 때 보유(추세).",
        "params": [
            dict(name="fast", label="MACD fast EMA", kind="int", min=3, max=50, step=1, default=12),
            dict(name="slow", label="MACD slow EMA", kind="int", min=10, max=120, step=1, default=26),
            dict(name="signal", label="Signal EMA", kind="int", min=3, max=30, step=1, default=9),
        ],
        "defaults": {"fast": 12, "slow": 26, "signal": 9},
    },
    "RSI Mean Reversion": {
        "fn": strat_rsi_reversion,
        "desc": "RSI가 LOW 아래면 진입, HIGH 위면 청산(역추세).",
        "params": [
            dict(name="n", label="RSI period", kind="int", min=5, max=50, step=1, default=14),
            dict(name="low", label="Entry (oversold)", kind="int", min=5, max=45, step=1, default=30),
            dict(name="high", label="Exit (overbought)", kind="int", min=55, max=95, step=1, default=70),
        ],
        "defaults": {"n": 14, "low": 30, "high": 70},
    },
    "Bollinger Mean Reversion": {
        "fn": strat_bollinger_reversion,
        "desc": "하단 밴드 이탈 시 진입, 중단선 회귀 시 청산.",
        "params": [
            dict(name="n", label="BB period", kind="int", min=5, max=60, step=1, default=20),
            dict(name="k", label="Std multiplier (k)", kind="float", min=1.0, max=4.0, step=0.1, default=2.0),
        ],
        "defaults": {"n": 20, "k": 2.0},
    },
    "Donchian Breakout": {
        "fn": strat_donchian_breakout,
        "desc": "n일 상단 돌파 시 진입, 하단 이탈 시 청산(추세추종).",
        "params": [
            dict(name="n", label="Donchian window", kind="int", min=5, max=120, step=1, default=20),
        ],
        "defaults": {"n": 20},
    },
    "Momentum + SMA Filter": {
        "fn": strat_momentum_filter,
        "desc": "lookback 수익률>0 AND 가격>SMA(filter)일 때 보유.",
        "params": [
            dict(name="lookback", label="Momentum lookback (days)", kind="int", min=20, max=300, step=5, default=120),
            dict(name="sma_filter", label="SMA filter (days)", kind="int", min=50, max=300, step=10, default=200),
        ],
        "defaults": {"lookback": 120, "sma_filter": 200},
    },
    "Trend + Vol Cap": {
        "fn": strat_vol_target_trend,
        "desc": "SMA 추세 신호 + 변동성(연환산) 상한 넘으면 risk-off.",
        "params": [
            dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
            dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
            dict(name="vol_window", label="Vol window (days)", kind="int", min=10, max=120, step=5, default=20),
            dict(name="vol_cap", label="Vol cap (ann.)", kind="float", min=0.10, max=1.00, step=0.01, default=0.30),
        ],
        "defaults": {"fast": 20, "slow": 60, "vol_window": 20, "vol_cap": 0.30},
    },
}


# ============================================================
# Plotting
# ============================================================
def plot_price_with_signals(df: pd.DataFrame, pos: pd.Series, title: str):
    c = df["Close"].astype(float)
    pos = pos.reindex(df.index).fillna(0.0)
    d = pos.diff().fillna(pos)

    entries = d[d > 0].index
    exits = d[d < 0].index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=c, mode="lines", name="Close"))

    if len(entries) > 0:
        fig.add_trace(go.Scatter(
            x=entries, y=c.loc[entries],
            mode="markers", name="Entry",
            marker=dict(symbol="triangle-up", size=10)
        ))
    if len(exits) > 0:
        fig.add_trace(go.Scatter(
            x=exits, y=c.loc[exits],
            mode="markers", name="Exit",
            marker=dict(symbol="triangle-down", size=10)
        ))

    fig.update_layout(title=title, height=520, xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_equity(equity: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(title=title, height=460, xaxis_title="Date", yaxis_title="Equity (start=1.0)")
    return fig

def plot_drawdown(equity: pd.Series, title: str):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(title=title, height=320, xaxis_title="Date", yaxis_title="Drawdown")
    return fig


# ============================================================
# Strategy param UI
# ============================================================
def build_params_ui(strategy_name: str):
    info = STRATEGIES[strategy_name]
    params = {}
    presets = {
        "Conservative": info.get("defaults", {}).copy(),
        "Default": info.get("defaults", {}).copy(),
        "Aggressive": info.get("defaults", {}).copy(),
    }

    # simple preset tweaks for a few known strategies
    if strategy_name == "SMA Crossover":
        presets["Conservative"] = {"fast": 50, "slow": 200}
        presets["Aggressive"] = {"fast": 10, "slow": 30}
    elif strategy_name == "EMA Crossover":
        presets["Conservative"] = {"fast": 24, "slow": 52}
        presets["Aggressive"] = {"fast": 6, "slow": 18}
    elif strategy_name == "RSI Mean Reversion":
        presets["Conservative"] = {"n": 20, "low": 25, "high": 65}
        presets["Aggressive"] = {"n": 10, "low": 35, "high": 75}
    elif strategy_name == "Bollinger Mean Reversion":
        presets["Conservative"] = {"n": 30, "k": 2.5}
        presets["Aggressive"] = {"n": 15, "k": 1.8}
    elif strategy_name == "Donchian Breakout":
        presets["Conservative"] = {"n": 55}
        presets["Aggressive"] = {"n": 10}
    elif strategy_name == "MACD Trend":
        presets["Conservative"] = {"fast": 12, "slow": 39, "signal": 9}
        presets["Aggressive"] = {"fast": 6, "slow": 18, "signal": 6}
    elif strategy_name == "Momentum + SMA Filter":
        presets["Conservative"] = {"lookback": 252, "sma_filter": 200}
        presets["Aggressive"] = {"lookback": 60, "sma_filter": 100}
    elif strategy_name == "Trend + Vol Cap":
        presets["Conservative"] = {"fast": 50, "slow": 200, "vol_window": 20, "vol_cap": 0.25}
        presets["Aggressive"] = {"fast": 10, "slow": 30, "vol_window": 10, "vol_cap": 0.40}

    preset_choice = st.radio("Preset", ["Default", "Conservative", "Aggressive"], horizontal=True)
    base = presets[preset_choice]

    for p in info["params"]:
        name = p["name"]
        label = p["label"]
        kind = p["kind"]
        default = base.get(name, p.get("default"))

        if kind == "int":
            params[name] = st.slider(label, int(p["min"]), int(p["max"]), int(default), int(p["step"]))
        else:
            params[name] = st.slider(label, float(p["min"]), float(p["max"]), float(default), float(p["step"]))

    return params


# ============================================================
# App UI
# ============================================================
st.set_page_config(page_title="KR/US Strategy Backtester", layout="wide")
st.title("📊 KR/US Stock Strategy Backtester")

with st.sidebar:
    st.header("Data")

    market = st.selectbox("Market", ["US (미국)", "Korea (한국)"])

    today = date.today()
    start_default = today - timedelta(days=365 * 3)

    # ✅ 안전한 date_input (단일/범위 모두 방어)
    date_range = st.date_input(
        "Date Range",
        value=[start_default, today],   # 리스트 권장
        max_value=today
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        st.warning("날짜 범위를 시작/끝 2개 모두 선택해주세요.")
        st.stop()

    if start_d > end_d:
        st.error("Start date must be <= end date.")
        st.stop()

    st.divider()
    st.header("Backtest Assumptions")
    fee_bps = st.number_input("Fee (bps per trade)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)
    slip_bps = st.number_input("Slippage (bps per trade)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

    st.divider()
    mode = st.radio("Mode", ["Single Strategy", "Compare Strategies"], index=0)

    st.divider()
    st.header("Symbol")

    listing_df = krx_listing() if market.startswith("Korea") else None

    if market.startswith("US"):
        ticker = st.text_input("US Ticker (e.g., AAPL, MSFT, NVDA)", value="AAPL").strip().upper()
        auto_adj = st.checkbox("Auto-adjust (splits/dividends)", value=True)
        us_prefer = st.selectbox("US Source preference", ["yfinance (default)", "stooq (fallback-first)"])
        prefer = "yfinance" if us_prefer.startswith("yfinance") else "stooq"
    else:
        # Korea sources
        options = []
        if PYKRX_OK:
            options.append("pykrx (KRX)")
        if FDR_OK:
            options.append("FinanceDataReader (KRX)")
        if YF_OK:
            options.append("yfinance (Yahoo)")
        if not options:
            st.error("한국 데이터 소스가 없습니다. pykrx / finance-datareader / yfinance 중 하나 설치하세요.")
            st.stop()

        kr_source = st.selectbox("Korea Data Source", options)
        code6 = st.text_input("KRX Code (6 digits, e.g., 005930)", value="005930").strip()
        name_q = st.text_input("Search by Name (optional)", value="").strip()

        chosen_name = None
        if listing_df is not None and name_q:
            hits = listing_df[listing_df["Name"].astype(str).str.contains(name_q, na=False)].head(30)
            if len(hits) > 0:
                opts = (hits["Symbol"].astype(str) + " — " + hits["Name"].astype(str)).tolist()
                pick = st.selectbox("Matches", opts)
                code6 = pick.split(" — ")[0].strip()
                chosen_name = pick.split(" — ")[1].strip()

        auto_adj = st.checkbox("Adjusted price (if supported)", value=True)

    st.divider()
    st.header("Strategy")

    if mode == "Single Strategy":
        strat_name = st.selectbox("Choose Strategy", list(STRATEGIES.keys()))
        st.caption(STRATEGIES[strat_name]["desc"])
        with st.expander("Parameters", expanded=True):
            params = build_params_ui(strat_name)
    else:
        default_sel = ["Buy & Hold", "SMA Crossover", "MACD Trend", "RSI Mean Reversion"]
        selected = st.multiselect("Select strategies to compare", list(STRATEGIES.keys()), default=default_sel)
        st.caption("Compare 모드는 각 전략의 Preset=Default 파라미터로 빠르게 비교합니다.")
        params = None
        strat_name = None

    st.divider()
    run = st.button("Run Backtest", type="primary")


def load_data():
    if market.startswith("US"):
        if not ticker:
            return pd.DataFrame(), "No ticker", "none"
        df, src = fetch_us(ticker, start_d, end_d, auto_adjust=auto_adj, prefer=prefer)
        title = f"{ticker} (US) — source={src}"
        return df, title, src

    # Korea
    if not (code6.isdigit() and len(code6) == 6):
        return pd.DataFrame(), "KR code must be 6 digits", "none"

    display = chosen_name or code6

    if kr_source.startswith("pykrx"):
        df, src, sym = fetch_kr(code6, start_d, end_d, adjusted=auto_adj, source="pykrx", listing_df=listing_df)
        title = f"{display} ({sym}) — source={src}"
        return df, title, src

    if kr_source.startswith("FinanceDataReader"):
        df, src, sym = fetch_kr(code6, start_d, end_d, adjusted=auto_adj, source="fdr", listing_df=listing_df)
        title = f"{display} ({sym}) — source={src}"
        return df, title, src

    df, src, sym = fetch_kr(code6, start_d, end_d, adjusted=auto_adj, source="yfinance", listing_df=listing_df)
    title = f"{display} ({sym}) — source={src}"
    return df, title, src


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and clean."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().sort_index()
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df.dropna(subset=["Open", "High", "Low", "Close"])


if run:
    df, base_title, src_used = load_data()
    df = ensure_ohlcv(df)

    if df.empty:
        st.error(
            "데이터를 가져오지 못했습니다.\n"
            "- yfinance면 레이트리밋/공유IP 문제일 수 있어요.\n"
            "- US는 Source preference를 stooq로 바꿔보거나, 잠시 후 재시도하세요.\n"
            "- KR은 pykrx 또는 FinanceDataReader 소스를 추천합니다."
        )
        st.stop()

    st.success(f"Loaded {len(df):,} rows | {base_title}")

    if mode == "Single Strategy":
        fn = STRATEGIES[strat_name]["fn"]
        pos = fn(df, **(params or {}))
        equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{metrics.get('Total Return', np.nan)*100:,.2f}%")
        c2.metric("CAGR", f"{metrics.get('CAGR', np.nan)*100:,.2f}%")
        c3.metric("Sharpe", f"{metrics.get('Sharpe (rf=0)', np.nan):,.2f}")
        c4.metric("Max DD", f"{metrics.get('Max Drawdown', np.nan)*100:,.2f}%")
        c5.metric("Trades", f"{int(metrics.get('Trades', 0)):,}")

        tab1, tab2, tab3, tab4 = st.tabs(["Price & Signals", "Equity", "Trades", "Metrics"])

        with tab1:
            st.plotly_chart(plot_price_with_signals(df, pos, f"{base_title} — {strat_name}"), use_container_width=True)
            with st.expander("Raw data (tail)"):
                st.dataframe(df.tail(300))

        with tab2:
            st.plotly_chart(plot_equity(equity, "Equity Curve (start=1.0)"), use_container_width=True)
            st.plotly_chart(plot_drawdown(equity, "Drawdown"), use_container_width=True)

        with tab3:
            st.dataframe(trades)
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button("Download trades CSV", data=csv, file_name="trades.csv", mime="text/csv")

        with tab4:
            mdf = pd.DataFrame([metrics]).T
            mdf.columns = ["Value"]
            st.dataframe(mdf)

    else:
        if not selected:
            st.warning("Select at least one strategy.")
            st.stop()

        results = []
        fig = go.Figure()

        for name in selected:
            info = STRATEGIES[name]
            fn = info["fn"]
            defaults = info.get("defaults", {})

            pos = fn(df, **defaults)
            equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)

            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=name))

            row = {"Strategy": name}
            row.update(metrics)
            results.append(row)

        fig.update_layout(
            title=f"{base_title} — Strategy Comparison (Equity start=1.0)",
            height=520,
            xaxis_title="Date",
            yaxis_title="Equity"
        )

        st.plotly_chart(fig, use_container_width=True)

        res_df = pd.DataFrame(results)

        # Formatting
        pct_cols = ["Total Return", "CAGR", "Max Drawdown", "Win Rate", "Avg Trade Return"]
        for col in pct_cols:
            if col in res_df.columns:
                res_df[col] = (res_df[col] * 100).round(2)
        if "Sharpe (rf=0)" in res_df.columns:
            res_df["Sharpe (rf=0)"] = res_df["Sharpe (rf=0)"].round(2)

        st.subheader("Metrics (percent columns are %)")
        sort_key = "CAGR" if "CAGR" in res_df.columns else "Total Return"
        st.dataframe(res_df.sort_values(by=sort_key, ascending=False, na_position="last"))

else:
    st.info(
        "왼쪽에서 Market/종목/기간/전략을 고르고 **Run Backtest**를 누르세요.\n\n"
        "FACT:\n"
        "- yfinance는 레이트리밋이 자주 걸릴 수 있어요(특히 Streamlit Cloud 공유 IP).\n"
        "- 그래서 US는 자동으로 Stooq로 fallback 하게 해뒀습니다(설치되어 있으면)."
    )
