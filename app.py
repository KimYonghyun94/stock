import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import contextlib, io, logging
import requests
from io import StringIO

# -----------------------------
# Optional deps
# -----------------------------
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import FinanceDataReader as fdr  # pip: finance-datareader
    FDR_OK = True
except Exception:
    FDR_OK = False

try:
    from pykrx import stock as krx_stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# -----------------------------
# Plotly helper (new width API + backward fallback)
# -----------------------------
def st_plotly(fig, stretch=True):
    try:
        st.plotly_chart(fig, width="stretch" if stretch else "content")
    except TypeError:
        st.plotly_chart(fig, use_container_width=stretch)

# ============================================================
# Indicators
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

# ============================================================
# Data fetch
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def krx_listing():
    """KRX listing for name search (FinanceDataReader)."""
    if not FDR_OK:
        return None
    try:
        return fdr.StockListing("KRX")
    except Exception:
        return None

def infer_yahoo_suffix(code6: str, listing_df: pd.DataFrame | None):
    """Heuristic: KOSPI .KS, KOSDAQ .KQ"""
    if listing_df is None:
        return ".KS"
    hit = listing_df[listing_df["Symbol"].astype(str) == str(code6)]
    if len(hit) == 0:
        return ".KS"
    market = str(hit.iloc[0].get("Market", "")).upper()
    return ".KQ" if "KOSDAQ" in market else ".KS"

@st.cache_data(ttl=300, show_spinner=False)  # 실패 결과가 오래 캐시되지 않게 5분
def fetch_stooq_us_daily_csv(ticker: str, start: date, end: date):
    """
    US via Stooq direct CSV with headers + domain fallback.
    Returns: (df, meta)
    """
    # BRK-B -> BRK.B 같은 케이스 대비
    sym = ticker.strip().upper().replace("-", ".")
    if not sym.endswith(".US"):
        sym = f"{sym}.US"

    bases = ["https://stooq.com", "https://stooq.pl"]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StockApp/1.0; +https://example.com)",
        "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    }

    attempts = []
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    for base in bases:
        url = f"{base}/q/d/l/?s={sym}&i=d"
        try:
            r = requests.get(url, headers=headers, timeout=20)
            head = (r.text or "")[:120].replace("\n", "\\n")
            attempts.append(f"{url} -> status={r.status_code}, head={head}")

            if r.status_code != 200:
                continue

            text = (r.text or "").lstrip()
            # CSV 여부 체크
            if not text.startswith("Date,Open,High,Low,Close"):
                continue

            df = pd.read_csv(StringIO(text))
            if df is None or df.empty or "Date" not in df.columns:
                continue

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

            need = {"Open", "High", "Low", "Close"}
            if not need.issubset(df.columns):
                continue
            if "Volume" not in df.columns:
                df["Volume"] = 0

            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            meta = {"ok": True, "used_url": url, "attempts": attempts[-3:]}
            return df, meta

        except Exception as e:
            attempts.append(f"{url} -> EXCEPTION: {type(e).__name__}: {e}")

    return pd.DataFrame(), {"ok": False, "attempts": attempts}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_daily(symbol: str, start: date, end: date, auto_adjust: bool = True) -> pd.DataFrame:
    """Daily OHLCV via yfinance (may rate-limit on Streamlit Cloud)."""
    if not YF_OK:
        return pd.DataFrame()
    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
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
        return df[keep].dropna()
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
            adjusted=adjusted,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
        df.index.name = "Date"
        return df[["Open","High","Low","Close","Volume"]].dropna()
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

# ============================================================
# Backtest engine (long/flat, close-to-close simplified)
# ============================================================
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
        avg_trade = float(trades["Return"].mean())
        n_trades = int(len(trades))
    else:
        win_rate, avg_trade, n_trades = np.nan, np.nan, 0

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
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos = pos.reindex(df.index).fillna(0.0).astype(float)
    held = pos.shift(1).fillna(0.0)  # avoid look-ahead

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
# Strategies (long/flat)
# ============================================================
def strat_buyhold(df, **params):
    return pd.Series(1.0, index=df.index)

def strat_sma_cross(df, fast=20, slow=60, **params):
    c = df["Close"].astype(float)
    return (sma(c, fast) > sma(c, slow)).astype(float)

def strat_ema_cross(df, fast=12, slow=26, **params):
    c = df["Close"].astype(float)
    return (ema(c, fast) > ema(c, slow)).astype(float)

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
    c = df["Close"].astype(float)
    mom = c / c.shift(lookback) - 1.0
    filt = c > sma(c, sma_filter)
    return ((mom > 0) & filt).astype(float)

def strat_trend_vol_cap(df, fast=20, slow=60, vol_window=20, vol_cap=0.30, **params):
    c = df["Close"].astype(float)
    signal = sma(c, fast) > sma(c, slow)
    vol = annualized_vol(c, window=vol_window)
    return (signal & (vol <= vol_cap)).astype(float)

STRATEGY_SPECS = {
    "Buy & Hold": {
        "fn": strat_buyhold,
        "desc": "항상 보유(기준선).",
        "params": [],
        "defaults": {}
    },
    "SMA Crossover": {
        "fn": strat_sma_cross,
        "desc": "FAST SMA > SLOW SMA일 때 보유(추세).",
        "params": [
            dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
            dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
        ],
        "defaults": {"fast": 20, "slow": 60}
    },
    "EMA Crossover": {
        "fn": strat_ema_cross,
        "desc": "FAST EMA > SLOW EMA일 때 보유(추세).",
        "params": [
            dict(name="fast", label="FAST EMA", kind="int", min=3, max=100, step=1, default=12),
            dict(name="slow", label="SLOW EMA", kind="int", min=5, max=200, step=1, default=26),
        ],
        "defaults": {"fast": 12, "slow": 26}
    },
    "MACD Trend": {
        "fn": strat_macd_trend,
        "desc": "MACD > Signal일 때 보유(추세).",
        "params": [
            dict(name="fast", label="MACD fast EMA", kind="int", min=3, max=50, step=1, default=12),
            dict(name="slow", label="MACD slow EMA", kind="int", min=10, max=120, step=1, default=26),
            dict(name="signal", label="Signal EMA", kind="int", min=3, max=30, step=1, default=9),
        ],
        "defaults": {"fast": 12, "slow": 26, "signal": 9}
    },
    "RSI Mean Reversion": {
        "fn": strat_rsi_reversion,
        "desc": "RSI LOW 아래 진입, HIGH 위 청산(역추세).",
        "params": [
            dict(name="n", label="RSI period", kind="int", min=5, max=50, step=1, default=14),
            dict(name="low", label="Entry (oversold)", kind="int", min=5, max=45, step=1, default=30),
            dict(name="high", label="Exit (overbought)", kind="int", min=55, max=95, step=1, default=70),
        ],
        "defaults": {"n": 14, "low": 30, "high": 70}
    },
    "Bollinger Mean Reversion": {
        "fn": strat_bollinger_reversion,
        "desc": "하단 밴드 이탈 진입, 중단 회귀 청산.",
        "params": [
            dict(name="n", label="BB period", kind="int", min=5, max=60, step=1, default=20),
            dict(name="k", label="Std multiplier (k)", kind="float", min=1.0, max=4.0, step=0.1, default=2.0),
        ],
        "defaults": {"n": 20, "k": 2.0}
    },
    "Donchian Breakout": {
        "fn": strat_donchian_breakout,
        "desc": "채널 상단 돌파 진입, 하단 이탈 청산.",
        "params": [
            dict(name="n", label="Donchian window", kind="int", min=5, max=120, step=1, default=20),
        ],
        "defaults": {"n": 20}
    },
    "Momentum + SMA Filter": {
        "fn": strat_momentum_filter,
        "desc": "모멘텀>0 & 가격>SMA(filter)일 때 보유.",
        "params": [
            dict(name="lookback", label="Momentum lookback", kind="int", min=20, max=300, step=5, default=120),
            dict(name="sma_filter", label="SMA filter", kind="int", min=50, max=300, step=10, default=200),
        ],
        "defaults": {"lookback": 120, "sma_filter": 200}
    },
    "Trend + Vol Cap": {
        "fn": strat_trend_vol_cap,
        "desc": "추세 신호 + 변동성 상한 넘으면 risk-off.",
        "params": [
            dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
            dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
            dict(name="vol_window", label="Vol window", kind="int", min=10, max=120, step=5, default=20),
            dict(name="vol_cap", label="Vol cap (ann.)", kind="float", min=0.10, max=1.00, step=0.01, default=0.30),
        ],
        "defaults": {"fast": 20, "slow": 60, "vol_window": 20, "vol_cap": 0.30}
    },
}

def build_params_ui(strategy_name: str):
    spec = STRATEGY_SPECS[strategy_name]
    params = {}
    for p in spec["params"]:
        if p["kind"] == "int":
            params[p["name"]] = st.slider(
                p["label"], int(p["min"]), int(p["max"]), int(p["default"]), int(p["step"])
            )
        else:
            params[p["name"]] = st.slider(
                p["label"], float(p["min"]), float(p["max"]), float(p["default"]), float(p["step"])
            )
    return params

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
# UI
# ============================================================
st.set_page_config(page_title="KR/US Strategy Backtester", layout="wide")
st.title("📊 KR/US Stock Strategy Backtester")

with st.sidebar:
    st.header("Cache")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared")

    st.divider()
    st.header("Data")
    market = st.selectbox("Market", ["US (미국)", "Korea (한국)"])

    today = date.today()
    start_default = today - timedelta(days=365 * 3)
    date_range = st.date_input("Date Range", value=[start_default, today], max_value=today)

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
    fee_bps = st.number_input("Fee (bps per trade)", 0.0, 200.0, 5.0, 1.0)
    slip_bps = st.number_input("Slippage (bps per trade)", 0.0, 200.0, 0.0, 1.0)

    st.divider()
    mode = st.radio("Mode", ["Single Strategy", "Compare Strategies"], index=0)

    st.divider()
    st.header("Symbol")

    listing_df = krx_listing() if market.startswith("Korea") else None

    if market.startswith("US"):
        us_ticker = st.text_input("US Ticker (e.g., AAPL, MSFT, NVDA, BRK-B)", value="AAPL").strip()
        show_us_debug = st.checkbox("Show US fetch debug", value=True)
        use_yf_fallback = st.checkbox("If Stooq fails, try yfinance (may rate limit)", value=False)
        yf_auto_adj = st.checkbox("yfinance auto-adjust", value=True)
    else:
        sources = []
        if PYKRX_OK: sources.append("pykrx (KRX)")
        if FDR_OK: sources.append("FinanceDataReader (KRX)")
        if YF_OK: sources.append("yfinance (Yahoo)")
        if not sources:
            st.error("한국 데이터 소스가 없습니다. pykrx / finance-datareader / yfinance 중 하나 설치하세요.")
            st.stop()

        kr_source = st.selectbox("Korea Data Source", sources)
        kr_code6 = st.text_input("KRX Code (6 digits, e.g., 005930)", value="005930").strip()
        name_q = st.text_input("Search by Name (optional)", value="").strip()
        chosen_name = None

        if listing_df is not None and name_q:
            hits = listing_df[listing_df["Name"].astype(str).str.contains(name_q, na=False)].head(30)
            if len(hits) > 0:
                opts = (hits["Symbol"].astype(str) + " — " + hits["Name"].astype(str)).tolist()
                pick = st.selectbox("Matches", opts)
                kr_code6 = pick.split(" — ")[0].strip()
                chosen_name = pick.split(" — ")[1].strip()

        kr_adjusted = st.checkbox("Adjusted price (if supported)", value=True)

    st.divider()
    st.header("Strategy")

    if mode == "Single Strategy":
        strat_name = st.selectbox("Choose Strategy", list(STRATEGY_SPECS.keys()))
        st.caption(STRATEGY_SPECS[strat_name]["desc"])
        with st.expander("Parameters", expanded=True):
            params = build_params_ui(strat_name)
    else:
        default_sel = ["Buy & Hold", "SMA Crossover", "MACD Trend", "RSI Mean Reversion"]
        selected = st.multiselect("Select strategies to compare", list(STRATEGY_SPECS.keys()), default=default_sel)
        st.caption("Compare 모드는 각 전략의 Default 파라미터로 비교합니다.")

    st.divider()
    run = st.button("Run Backtest", type="primary")

def load_data():
    if market.startswith("US"):
        ticker = us_ticker.upper().strip()
        df, meta = fetch_stooq_us_daily_csv(ticker, start_d, end_d)
        src = "stooq(csv)"

        # debug (optional)
        if show_us_debug:
            with st.expander("US fetch debug (Stooq)"):
                st.write(meta)

        # optional yfinance fallback
        if df.empty and use_yf_fallback and YF_OK:
            st.warning("Stooq 실패 → yfinance fallback 시도 중… (레이트리밋 가능)")
            df2 = fetch_yfinance_daily(ticker, start_d, end_d, auto_adjust=yf_auto_adj)
            if not df2.empty:
                df, src = df2, "yfinance"
        title = f"{ticker} (US) — source={src}"
        return df, title

    # Korea
    if not (kr_code6.isdigit() and len(kr_code6) == 6):
        return pd.DataFrame(), "KR code must be 6 digits"

    display = chosen_name or kr_code6

    if kr_source.startswith("pykrx"):
        df = fetch_pykrx_daily(kr_code6, start_d, end_d, adjusted=kr_adjusted)
        title = f"{display} ({kr_code6}) — source=pykrx"
        return df, title

    if kr_source.startswith("FinanceDataReader"):
        df = fetch_fdr_korea_daily(kr_code6, start_d, end_d)
        title = f"{display} ({kr_code6}) — source=fdr"
        return df, title

    sym = f"{kr_code6}{infer_yahoo_suffix(kr_code6, listing_df)}"
    df = fetch_yfinance_daily(sym, start_d, end_d, auto_adjust=kr_adjusted)
    title = f"{display} ({sym}) — source=yfinance"
    return df, title

if run:
    df, base_title = load_data()
    df = ensure_ohlcv(df)

    if df.empty:
        st.error(
            "데이터를 가져오지 못했습니다.\n"
            "- US: Stooq 응답이 CSV가 아니거나(차단/HTML), 네트워크가 막히면 빈 데이터가 올 수 있어요.\n"
            "- yfinance fallback은 Streamlit Cloud에서 레이트리밋으로 실패할 수 있어요."
        )
        st.stop()

    st.success(f"Loaded {len(df):,} rows | {base_title}")

    if mode == "Single Strategy":
        fn = STRATEGY_SPECS[strat_name]["fn"]
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
            st_plotly(plot_price_with_signals(df, pos, f"{base_title} — {strat_name}"), stretch=True)
            with st.expander("Raw data (tail)"):
                st.dataframe(df.tail(300))

        with tab2:
            st_plotly(plot_equity(equity, "Equity Curve (start=1.0)"), stretch=True)
            st_plotly(plot_drawdown(equity, "Drawdown"), stretch=True)

        with tab3:
            st.dataframe(trades)
            st.download_button(
                "Download trades CSV",
                data=trades.to_csv(index=False).encode("utf-8"),
                file_name="trades.csv",
                mime="text/csv",
            )

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
            spec = STRATEGY_SPECS[name]
            fn = spec["fn"]
            defaults = spec.get("defaults", {})

            pos = fn(df, **defaults)
            equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)

            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=name))

            row = {"Strategy": name}
            row.update(metrics)
            results.append(row)

        fig.update_layout(title=f"{base_title} — Strategy Comparison (Equity start=1.0)", height=520)
        st_plotly(fig, stretch=True)

        res_df = pd.DataFrame(results)
        for col in ["Total Return", "CAGR", "Max Drawdown", "Win Rate", "Avg Trade Return"]:
            if col in res_df.columns:
                res_df[col] = (res_df[col] * 100).round(2)
        if "Sharpe (rf=0)" in res_df.columns:
            res_df["Sharpe (rf=0)"] = res_df["Sharpe (rf=0)"].round(2)

        st.subheader("Metrics (percent columns are %)")
        sort_key = "CAGR" if "CAGR" in res_df.columns else "Total Return"
        st.dataframe(res_df.sort_values(by=sort_key, ascending=False, na_position="last"))

else:
    st.info("왼쪽에서 Market/종목/기간/전략을 고르고 **Run Backtest**를 누르세요.")
