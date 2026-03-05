import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

# -----------------------------
# Optional deps (KRX only)
# -----------------------------
try:
    from pykrx import stock as krx_stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception:
    FDR_OK = False


# -----------------------------
# Indicators
# -----------------------------
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


# -----------------------------
# Data fetch (KRX)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pykrx_daily(code6: str, start: date, end: date, adjusted: bool = True) -> pd.DataFrame:
    if not PYKRX_OK:
        return pd.DataFrame()
    try:
        df = krx_stock.get_market_ohlcv_by_date(
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            code6,
            adjusted=adjusted,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fdr_daily(code6: str, start: date, end: date) -> pd.DataFrame:
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

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().sort_index()
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df.dropna(subset=["Open","High","Low","Close"])

def load_krx_data(code6: str, start: date, end: date) -> tuple[pd.DataFrame, str]:
    """
    깔끔 UI를 위해 데이터 소스 선택 UI는 제거하고,
    가능한 소스를 자동으로 시도합니다: pykrx -> FDR
    """
    # 1) pykrx
    df = fetch_pykrx_daily(code6, start, end, adjusted=True)
    df = ensure_ohlcv(df)
    if not df.empty:
        return df, "pykrx"

    # 2) FDR
    df = fetch_fdr_daily(code6, start, end)
    df = ensure_ohlcv(df)
    if not df.empty:
        return df, "FinanceDataReader"

    return pd.DataFrame(), "None"


# -----------------------------
# Backtest (long/flat)
# -----------------------------
def positions_to_trades(close: pd.Series, pos: pd.Series) -> pd.DataFrame:
    pos = pos.fillna(0).astype(float)
    d = pos.diff().fillna(pos)
    entries = d[d > 0].index
    exits = d[d < 0].index

    trades = []
    j = 0
    for entry_dt in entries:
        while j < len(exits) and exits[j] <= entry_dt:
            j += 1
        exit_dt = exits[j] if j < len(exits) else close.index[-1]
        if j < len(exits):
            j += 1

        entry_px = float(close.loc[entry_dt])
        exit_px = float(close.loc[exit_dt])
        if entry_px == 0:
            continue
        trades.append([entry_dt, exit_dt, entry_px, exit_px, exit_px / entry_px - 1.0])

    if not trades:
        return pd.DataFrame(columns=["Entry","Exit","EntryPx","ExitPx","Return","Days"])

    tdf = pd.DataFrame(trades, columns=["Entry","Exit","EntryPx","ExitPx","Return"])
    tdf["Days"] = (tdf["Exit"] - tdf["Entry"]).dt.days
    return tdf

def compute_metrics(equity: pd.Series, strat_ret: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    strat_ret = strat_ret.dropna()
    if len(equity) < 2:
        return {}

    ann = 252.0
    n = len(strat_ret)
    years = n / ann if n else np.nan

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
        n_trades = int(len(trades))
    else:
        win_rate, n_trades = np.nan, 0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": mdd,
        "Trades": n_trades,
        "Win Rate": win_rate,
    }

def run_backtest(df: pd.DataFrame, pos: pd.Series):
    """
    비용/슬리피지 UI 제거(=0으로 가정).
    신호는 종가 기준 계산, 수익은 다음 거래일부터 반영(pos.shift(1)).
    """
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos = pos.reindex(df.index).fillna(0.0).astype(float)
    held = pos.shift(1).fillna(0.0)

    strat_ret = held * ret
    equity = (1.0 + strat_ret).cumprod()

    trades = positions_to_trades(close, pos)
    metrics = compute_metrics(equity, strat_ret, trades)
    return equity, trades, metrics


# -----------------------------
# Strategies
# -----------------------------
def strat_buyhold(df, **params):
    return pd.Series(1.0, index=df.index)

def strat_sma_cross(df, fast=20, slow=60, **params):
    c = df["Close"].astype(float)
    return (sma(c, fast) > sma(c, slow)).astype(float)

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

STRATEGIES = {
    "Buy & Hold": {
        "fn": strat_buyhold,
        "desc": "항상 보유(기준선).",
        "params": []
    },
    "SMA Crossover": {
        "fn": strat_sma_cross,
        "desc": "FAST SMA > SLOW SMA이면 보유.",
        "params": [
            dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
            dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
        ]
    },
    "MACD Trend": {
        "fn": strat_macd_trend,
        "desc": "MACD > Signal이면 보유.",
        "params": [
            dict(name="fast", label="MACD fast EMA", kind="int", min=3, max=50, step=1, default=12),
            dict(name="slow", label="MACD slow EMA", kind="int", min=10, max=120, step=1, default=26),
            dict(name="signal", label="Signal EMA", kind="int", min=3, max=30, step=1, default=9),
        ]
    },
    "RSI Mean Reversion": {
        "fn": strat_rsi_reversion,
        "desc": "RSI<LOW면 진입, RSI>HIGH면 청산.",
        "params": [
            dict(name="n", label="RSI period", kind="int", min=5, max=50, step=1, default=14),
            dict(name="low", label="Entry (LOW)", kind="int", min=5, max=45, step=1, default=30),
            dict(name="high", label="Exit (HIGH)", kind="int", min=55, max=95, step=1, default=70),
        ]
    },
    "Bollinger Mean Reversion": {
        "fn": strat_bollinger_reversion,
        "desc": "하단밴드 이탈 진입, 중단선 회귀 청산.",
        "params": [
            dict(name="n", label="BB period", kind="int", min=5, max=60, step=1, default=20),
            dict(name="k", label="Std multiplier (k)", kind="float", min=1.0, max=4.0, step=0.1, default=2.0),
        ]
    },
    "Donchian Breakout": {
        "fn": strat_donchian_breakout,
        "desc": "채널 상단 돌파 진입, 하단 이탈 청산.",
        "params": [
            dict(name="n", label="Donchian window", kind="int", min=5, max=120, step=1, default=20),
        ]
    },
}


# -----------------------------
# Plotting
# -----------------------------
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
            x=entries, y=c.loc[entries], mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=10)
        ))
    if len(exits) > 0:
        fig.add_trace(go.Scatter(
            x=exits, y=c.loc[exits], mode="markers", name="SELL",
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

def latest_signal_from_pos(pos: pd.Series):
    pos = pos.dropna()
    if len(pos) < 2:
        return "—", "Not enough data."
    prev_pos = int(pos.iloc[-2])
    last_pos = int(pos.iloc[-1])

    if prev_pos == 0 and last_pos == 1:
        return "BUY", "0→1 (현금→보유)"
    if prev_pos == 1 and last_pos == 0:
        return "SELL", "1→0 (보유→현금)"
    if last_pos == 1:
        return "HOLD", "1→1 (보유 유지)"
    return "CASH", "0→0 (현금 유지)"


# ============================================================
# UI (clean)
# ============================================================
st.set_page_config(page_title="KRX Strategy Backtester", layout="wide")
st.title("📈 KRX Strategy Backtester")

# Top control bar (no sidebar)
c1, c2, c3, c4 = st.columns([1.0, 1.6, 1.3, 0.8])

with c1:
    code6 = st.text_input("종목코드(6자리)", value="005930", max_chars=6)

with c2:
    today = date.today()
    start_default = today - timedelta(days=365 * 3)
    dr = st.date_input("기간", value=[start_default, today], max_value=today)
    if not (isinstance(dr, (list, tuple)) and len(dr) == 2):
        st.stop()
    start_d, end_d = dr

with c3:
    strat_name = st.selectbox("전략", list(STRATEGIES.keys()))
    st.caption(STRATEGIES[strat_name]["desc"])

with c4:
    run = st.button("Run", type="primary")

# Validate inputs
code6 = (code6 or "").strip()
if not (code6.isdigit() and len(code6) == 6):
    st.error("종목코드는 6자리 숫자여야 합니다. 예: 005930")
    st.stop()

if start_d > end_d:
    st.error("기간 시작일이 종료일보다 늦을 수 없습니다.")
    st.stop()

# Strategy params (kept minimal: collapsed expander)
params = {}
pdefs = STRATEGIES[strat_name]["params"]
if len(pdefs) > 0:
    with st.expander("Parameters", expanded=False):
        for p in pdefs:
            if p["kind"] == "int":
                params[p["name"]] = st.slider(
                    p["label"], int(p["min"]), int(p["max"]), int(p["default"]), int(p["step"])
                )
            else:
                params[p["name"]] = st.slider(
                    p["label"], float(p["min"]), float(p["max"]), float(p["default"]), float(p["step"])
                )

# Run
if run:
    df, src = load_krx_data(code6, start_d, end_d)
    if df.empty:
        st.error("데이터를 가져오지 못했습니다. (네트워크/데이터소스 문제 가능)  \n종목코드와 기간을 확인해 주세요.")
        st.stop()

    st.caption(f"Data source: {src} | rows: {len(df):,}")

    fn = STRATEGIES[strat_name]["fn"]
    pos = fn(df, **params)

    action, detail = latest_signal_from_pos(pos)
    st.metric("Latest Signal", action)
    st.caption(f"{detail} | 신호는 종가 기준, 체결은 다음 거래일 가정")

    equity, trades, metrics = run_backtest(df, pos)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Return", f"{metrics.get('Total Return', np.nan)*100:,.2f}%")
    m2.metric("CAGR", f"{metrics.get('CAGR', np.nan)*100:,.2f}%")
    m3.metric("Sharpe", f"{metrics.get('Sharpe (rf=0)', np.nan):,.2f}")
    m4.metric("Max DD", f"{metrics.get('Max Drawdown', np.nan)*100:,.2f}%")
    m5.metric("Trades", f"{int(metrics.get('Trades', 0)):,}")

    tab1, tab2, tab3 = st.tabs(["Price & Signals", "Equity", "Trades"])
    with tab1:
        st.plotly_chart(plot_price_with_signals(df, pos, f"{code6} — {strat_name}"), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_equity(equity, "Equity Curve (start=1.0)"), use_container_width=True)
        st.plotly_chart(plot_drawdown(equity, "Drawdown"), use_container_width=True)
    with tab3:
        st.dataframe(trades, use_container_width=True)
        st.download_button(
            "Download trades CSV",
            data=trades.to_csv(index=False).encode("utf-8"),
            file_name="trades.csv",
            mime="text/csv",
        )
else:
    st.info("종목코드 / 기간 / 전략을 고르고 Run을 누르세요.")
