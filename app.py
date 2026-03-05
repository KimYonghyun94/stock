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

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    h = m - s
    return m, s, h

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

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(n).mean()


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
        return df[["Open","High","Low","Close","Volume"]].dropna()
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
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
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
    # 깔끔 UI: 데이터 소스 선택 UI 제거, 가능한 것 자동 시도
    df = ensure_ohlcv(fetch_pykrx_daily(code6, start, end, adjusted=True))
    if not df.empty:
        return df, "pykrx"
    df = ensure_ohlcv(fetch_fdr_daily(code6, start, end))
    if not df.empty:
        return df, "FinanceDataReader"
    return pd.DataFrame(), "None"


# -----------------------------
# Backtest (long/flat, no fees)
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
        avg_trade = float(trades["Return"].mean())
    else:
        win_rate, n_trades, avg_trade = np.nan, 0, np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": mdd,
        "Trades": n_trades,
        "Win Rate": win_rate,
        "Avg Trade Return": avg_trade,
    }

def run_backtest(df: pd.DataFrame, pos: pd.Series):
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos = pos.reindex(df.index).fillna(0.0).astype(float)
    held = pos.shift(1).fillna(0.0)  # 다음 거래일부터 반영(룩어헤드 방지)

    strat_ret = held * ret
    equity = (1.0 + strat_ret).cumprod()

    trades = positions_to_trades(close, pos)
    metrics = compute_metrics(equity, strat_ret, trades)
    return equity, strat_ret, trades, metrics


# -----------------------------
# Latest Signal (explicit)
# -----------------------------
def latest_signal_from_pos(pos: pd.Series) -> tuple[str, str]:
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


# -----------------------------
# Strategies
# -----------------------------
def strat_buyhold(df, **params):
    return pd.Series(1.0, index=df.index)

def strat_sma_cross(df, fast=20, slow=60, **params):
    c = df["Close"].astype(float)
    return (sma(c, fast) > sma(c, slow)).astype(float)

def strat_ema_cross(df, fast=12, slow=26, **params):
    c = df["Close"].astype(float)
    return (ema(c, fast) > ema(c, slow)).astype(float)

def strat_price_above_ema(df, n=50, **params):
    c = df["Close"].astype(float)
    return (c > ema(c, n)).astype(float)

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

def strat_momentum(df, n=60, **params):
    c = df["Close"].astype(float)
    mom = c.pct_change(n)
    return (mom > 0).astype(float)

def strat_keltner_breakout(df, ema_n=20, atr_n=20, mult=1.5, **params):
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    mid = ema(c, ema_n)
    a = atr(h, l, c, atr_n)
    up = mid + mult * a
    # dn = mid - mult * a  # 사용 안 해도 됨

    entry = (c > up.shift(1)).astype(int)
    exit_ = (c < mid.shift(1)).astype(int)

    pos = pd.Series(0, index=df.index, dtype=float)
    in_pos = 0
    for t in df.index:
        if in_pos == 0 and entry.loc[t] == 1:
            in_pos = 1
        elif in_pos == 1 and exit_.loc[t] == 1:
            in_pos = 0
        pos.loc[t] = float(in_pos)
    return pos

def strat_supertrend(df, atr_n=10, factor=3.0, **params):
    """
    표준 Supertrend(롱/현금).
    """
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    a = atr(h, l, c, atr_n)
    hl2 = (h + l) / 2.0
    basic_ub = hl2 + factor * a
    basic_lb = hl2 - factor * a

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(df)):
        if np.isfinite(final_ub.iloc[i-1]):
            final_ub.iloc[i] = min(basic_ub.iloc[i], final_ub.iloc[i-1]) if c.iloc[i-1] <= final_ub.iloc[i-1] else basic_ub.iloc[i]
        if np.isfinite(final_lb.iloc[i-1]):
            final_lb.iloc[i] = max(basic_lb.iloc[i], final_lb.iloc[i-1]) if c.iloc[i-1] >= final_lb.iloc[i-1] else basic_lb.iloc[i]

    trend = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if trend.iloc[i-1] == 1:
            trend.iloc[i] = 0 if c.iloc[i] < final_lb.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if c.iloc[i] > final_ub.iloc[i] else 0

    return trend.astype(float)


STRATEGIES = {
    # Baselines / Trend
    "Buy & Hold": {"fn": strat_buyhold, "desc": "항상 보유(기준선).", "params": []},
    "SMA Crossover": {"fn": strat_sma_cross, "desc": "FAST SMA > SLOW SMA이면 보유.", "params": [
        dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
        dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
    ]},
    "EMA Crossover": {"fn": strat_ema_cross, "desc": "FAST EMA > SLOW EMA이면 보유.", "params": [
        dict(name="fast", label="FAST EMA", kind="int", min=3, max=100, step=1, default=12),
        dict(name="slow", label="SLOW EMA", kind="int", min=10, max=300, step=1, default=26),
    ]},
    "Price > EMA Filter": {"fn": strat_price_above_ema, "desc": "종가가 EMA(n) 위면 보유.", "params": [
        dict(name="n", label="EMA period", kind="int", min=10, max=300, step=1, default=50),
    ]},
    "MACD Trend": {"fn": strat_macd_trend, "desc": "MACD > Signal이면 보유.", "params": [
        dict(name="fast", label="MACD fast EMA", kind="int", min=3, max=50, step=1, default=12),
        dict(name="slow", label="MACD slow EMA", kind="int", min=10, max=120, step=1, default=26),
        dict(name="signal", label="Signal EMA", kind="int", min=3, max=30, step=1, default=9),
    ]},
    "Momentum (ROC>0)": {"fn": strat_momentum, "desc": "n일 수익률(ROC)>0이면 보유.", "params": [
        dict(name="n", label="ROC window (days)", kind="int", min=10, max=252, step=1, default=60),
    ]},
    "Donchian Breakout": {"fn": strat_donchian_breakout, "desc": "채널 상단 돌파 진입, 하단 이탈 청산.", "params": [
        dict(name="n", label="Donchian window", kind="int", min=5, max=120, step=1, default=20),
    ]},
    "Keltner Breakout": {"fn": strat_keltner_breakout, "desc": "Keltner 상단 돌파 진입, 중단선 이탈 청산.", "params": [
        dict(name="ema_n", label="EMA period", kind="int", min=5, max=100, step=1, default=20),
        dict(name="atr_n", label="ATR period", kind="int", min=5, max=100, step=1, default=20),
        dict(name="mult", label="ATR multiplier", kind="float", min=0.5, max=5.0, step=0.1, default=1.5),
    ]},
    "Supertrend": {"fn": strat_supertrend, "desc": "ATR 기반 추세추종(Supertrend).", "params": [
        dict(name="atr_n", label="ATR period", kind="int", min=5, max=50, step=1, default=10),
        dict(name="factor", label="Factor", kind="float", min=1.0, max=6.0, step=0.1, default=3.0),
    ]},

    # Mean reversion
    "RSI Mean Reversion": {"fn": strat_rsi_reversion, "desc": "RSI<LOW면 진입, RSI>HIGH면 청산.", "params": [
        dict(name="n", label="RSI period", kind="int", min=5, max=50, step=1, default=14),
        dict(name="low", label="Entry (LOW)", kind="int", min=5, max=45, step=1, default=30),
        dict(name="high", label="Exit (HIGH)", kind="int", min=55, max=95, step=1, default=70),
    ]},
    "Bollinger Mean Reversion": {"fn": strat_bollinger_reversion, "desc": "하단밴드 이탈 진입, 중단선 회귀 청산.", "params": [
        dict(name="n", label="BB period", kind="int", min=5, max=60, step=1, default=20),
        dict(name="k", label="Std multiplier (k)", kind="float", min=1.0, max=4.0, step=0.1, default=2.0),
    ]},
}


# -----------------------------
# Plotting
# -----------------------------
def plot_equity_compare(equities: dict, title: str):
    fig = go.Figure()
    for name, eq in equities.items():
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=name))
    fig.update_layout(
        title=title,
        height=560,
        xaxis_title="Date",
        yaxis_title="Equity (start=1.0)",
        legend_title="Strategies",
    )
    return fig


# ============================================================
# UI (clean)
# ============================================================
st.set_page_config(page_title="KRX Strategy Backtester", layout="wide")
st.title("📈 KRX Strategy Backtester")

# Top controls
row1 = st.columns([1.0, 1.7, 2.0, 0.7])
with row1[0]:
    code6 = st.text_input("종목코드(6자리)", value="005930", max_chars=6)
with row1[1]:
    today = date.today()
    start_default = today - timedelta(days=365 * 3)
    dr = st.date_input("기간", value=[start_default, today], max_value=today)
with row1[2]:
    default_sel = ["Buy & Hold", "SMA Crossover", "EMA Crossover", "MACD Trend", "Donchian Breakout"]
    selected = st.multiselect("전략(멀티 선택)", list(STRATEGIES.keys()), default=default_sel)
with row1[3]:
    run = st.button("Run", type="primary")

# Validation
code6 = (code6 or "").strip()
if not (code6.isdigit() and len(code6) == 6):
    st.error("종목코드는 6자리 숫자여야 합니다. 예: 005930")
    st.stop()

if not (isinstance(dr, (list, tuple)) and len(dr) == 2):
    st.stop()
start_d, end_d = dr
if start_d > end_d:
    st.error("기간 시작일이 종료일보다 늦을 수 없습니다.")
    st.stop()

if not selected:
    st.warning("전략을 최소 1개 선택하세요.")
    st.stop()

# Advanced params (only for selected strategies)
params_by_strat = {}
with st.expander("Advanced Parameters (선택)", expanded=False):
    for name in selected:
        pdefs = STRATEGIES[name]["params"]
        if not pdefs:
            params_by_strat[name] = {}
            continue

        st.markdown(f"**{name}** — {STRATEGIES[name]['desc']}")
        params = {}
        for p in pdefs:
            if p["kind"] == "int":
                params[p["name"]] = st.slider(
                    p["label"], int(p["min"]), int(p["max"]), int(p["default"]), int(p["step"]),
                    key=f"{name}-{p['name']}"
                )
            else:
                params[p["name"]] = st.slider(
                    p["label"], float(p["min"]), float(p["max"]), float(p["default"]), float(p["step"]),
                    key=f"{name}-{p['name']}"
                )
        params_by_strat[name] = params

if run:
    df, src = load_krx_data(code6, start_d, end_d)
    if df.empty:
        st.error("데이터를 가져오지 못했습니다. (데이터 소스/네트워크 문제 가능) 종목코드/기간 확인해 주세요.")
        st.stop()

    st.caption(f"Data source: {src} | rows: {len(df):,}")

    # Run all selected strategies
    equities = {}
    summary_rows = []
    trades_map = {}

    for name in selected:
        fn = STRATEGIES[name]["fn"]
        params = params_by_strat.get(name, {})
        pos = fn(df, **params)

        sig, sig_detail = latest_signal_from_pos(pos)
        equity, strat_ret, trades, metrics = run_backtest(df, pos)

        equities[name] = equity
        trades_map[name] = trades

        row = {"Strategy": name, "Latest Signal": sig, "Signal Detail": sig_detail}
        row.update(metrics)
        summary_rows.append(row)

    # 1) One chart: equity comparison
    fig = plot_equity_compare(equities, f"{code6} — Equity Comparison (start=1.0)")
    st.plotly_chart(fig, use_container_width=True)

    # 2) Signals & metrics table
    res = pd.DataFrame(summary_rows)

    # pretty formatting
    pct_cols = ["Total Return", "CAGR", "Max Drawdown", "Win Rate", "Avg Trade Return"]
    for c in pct_cols:
        if c in res.columns:
            res[c] = (res[c] * 100).round(2)
    if "Sharpe (rf=0)" in res.columns:
        res["Sharpe (rf=0)"] = res["Sharpe (rf=0)"].round(2)

    st.subheader("Signals & Metrics")
    st.dataframe(
        res.sort_values(by="CAGR" if "CAGR" in res.columns else "Total Return", ascending=False, na_position="last"),
        use_container_width=True
    )
    st.caption("Latest Signal은 오늘 종가 기준으로 계산했고, 실제 체결/수익 반영은 다음 거래일부터 가정했습니다.")

    # 3) Trades detail (optional)
    st.subheader("Trades (선택)")
    pick = st.selectbox("트레이드 상세 보기", selected, index=0)
    tdf = trades_map.get(pick, pd.DataFrame())
    st.dataframe(tdf, use_container_width=True)
    st.download_button(
        "Download trades CSV",
        data=tdf.to_csv(index=False).encode("utf-8"),
        file_name=f"trades_{code6}_{pick.replace(' ', '_')}.csv",
        mime="text/csv",
    )

else:
    st.info("종목코드 / 기간 / 전략(멀티) 선택 후 Run.")
