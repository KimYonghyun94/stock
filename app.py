import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import contextlib, io, logging

# -----------------------------
# Optional dependencies
# -----------------------------
try:
    from pandas_datareader import data as pdr  # Stooq fallback/primary
    PDR_OK = True
except Exception:
    PDR_OK = False

try:
    import yfinance as yf  # optional (may rate limit on Streamlit Cloud)
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

# Reduce noisy logs (especially yfinance)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


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
# Data fetch (cached)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def krx_listing():
    if not FDR_OK:
        return None
    try:
        return fdr.StockListing("KRX")
    except Exception:
        return None

def infer_yahoo_suffix(code6: str, listing_df: pd.DataFrame | None):
    if listing_df is None:
        return ".KS"
    hit = listing_df[listing_df["Symbol"].astype(str) == str(code6)]
    if len(hit) == 0:
        return ".KS"
    market = str(hit.iloc[0].get("Market", "")).upper()
    return ".KQ" if "KOSDAQ" in market else ".KS"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq_us_daily(ticker: str, start: date, end: date) -> pd.DataFrame:
    if not PDR_OK:
        return pd.DataFrame()
    try:
        sym = ticker.upper()
        if not sym.endswith(".US"):
            sym = f"{sym}.US"
        df = pdr.DataReader(sym, "stooq", start, end).sort_index()
        if df is None or df.empty:
            return pd.DataFrame()
        df.index.name = "Date"
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_daily(symbol: str, start: date, end: date, auto_adjust: bool = True) -> pd.DataFrame:
    if not YF_OK:
        return pd.DataFrame()
    try:
        # Suppress yfinance prints in Streamlit logs
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
        df = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy().sort_index()
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df.dropna(subset=["Open", "High", "Low", "Close"])


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

STRATEGIES = {
    "Buy & Hold": (strat_buyhold, "항상 보유(기준선)."),
    "SMA Crossover": (strat_sma_cross, "FAST SMA > SLOW SMA일 때 보유(추세)."),
    "EMA Crossover": (strat_ema_cross, "FAST EMA > SLOW EMA일 때 보유(추세)."),
    "MACD Trend": (strat_macd_trend, "MACD > Signal일 때 보유(추세)."),
    "RSI Mean Reversion": (strat_rsi_reversion, "RSI LOW 아래 진입, HIGH 위 청산(역추세)."),
    "Bollinger Mean Reversion": (strat_bollinger_reversion, "하단 밴드 이탈 진입, 중단 회귀 청산."),
    "Donchian Breakout": (strat_donchian_breakout, "채널 상단 돌파 진입, 하단 이탈 청산."),
    "Momentum + SMA Filter": (strat_momentum_filter, "모멘텀>0 & 가격>SMA(filter)일 때 보유."),
    "Trend + Vol Cap": (strat_trend_vol_cap, "추세 신호 + 변동성 상한 넘으면 risk-off."),
}


# ============================================================
# Plotting (Streamlit new API: width="stretch")
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
# App UI
# ============================================================
st.set_page_config(page_title="KR/US Strategy Backtester", layout="wide")
st.title("📊 KR/US Stock Strategy Backtester")

with st.sidebar:
    st.header("Data")

    market = st.selectbox("Market", ["US (미국)", "Korea (한국)"])

    today = date.today()
    start_default = today - timedelta(days=365 * 3)

    # SAFE date range handling
    date_range = st.date_input("Date Range", value=[start_default, today], max_value=today)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        st.warning("날짜 범위를 시작/끝 2개 모두 선택해주세요.")
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
        if not PDR_OK:
            st.error("US 기본 소스(Stooq)용 pandas-datareader가 필요합니다. requirements.txt에 pandas-datareader 추가하세요.")
            st.stop()

        ticker = st.text_input("US Ticker (e.g., AAPL, MSFT, NVDA)", value="AAPL").strip().upper()
        use_yahoo = st.checkbox("Also try Yahoo (yfinance) if Stooq fails (may rate limit)", value=False)
        auto_adj = st.checkbox("Auto-adjust (Yahoo only)", value=True)

    else:
        sources = []
        if PYKRX_OK: sources.append("pykrx (KRX)")
        if FDR_OK: sources.append("FinanceDataReader (KRX)")
        if YF_OK: sources.append("yfinance (Yahoo)")
        if not sources:
            st.error("한국 데이터 소스가 없습니다. pykrx / finance-datareader / yfinance 중 하나 설치하세요.")
            st.stop()

        kr_source = st.selectbox("Korea Data Source", sources)
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
        st.caption(STRATEGIES[strat_name][1])

        with st.expander("Parameters", expanded=True):
            params = {}

            if strat_name == "SMA Crossover":
                params["fast"] = st.slider("FAST SMA", 5, 200, 20, 1)
                params["slow"] = st.slider("SLOW SMA", 10, 400, 60, 1)
            elif strat_name == "EMA Crossover":
                params["fast"] = st.slider("FAST EMA", 3, 100, 12, 1)
                params["slow"] = st.slider("SLOW EMA", 5, 200, 26, 1)
            elif strat_name == "MACD Trend":
                params["fast"] = st.slider("MACD fast EMA", 3, 50, 12, 1)
                params["slow"] = st.slider("MACD slow EMA", 10, 120, 26, 1)
                params["signal"] = st.slider("Signal EMA", 3, 30, 9, 1)
            elif strat_name == "RSI Mean Reversion":
                params["n"] = st.slider("RSI period", 5, 50, 14, 1)
                params["low"] = st.slider("Entry (oversold)", 5, 45, 30, 1)
                params["high"] = st.slider("Exit (overbought)", 55, 95, 70, 1)
            elif strat_name == "Bollinger Mean Reversion":
                params["n"] = st.slider("BB period", 5, 60, 20, 1)
                params["k"] = st.slider("Std multiplier (k)", 1.0, 4.0, 2.0, 0.1)
            elif strat_name == "Donchian Breakout":
                params["n"] = st.slider("Donchian window", 5, 120, 20, 1)
            elif strat_name == "Momentum + SMA Filter":
                params["lookback"] = st.slider("Momentum lookback", 20, 300, 120, 5)
                params["sma_filter"] = st.slider("SMA filter", 50, 300, 200, 10)
            elif strat_name == "Trend + Vol Cap":
                params["fast"] = st.slider("FAST SMA", 5, 200, 20, 1)
                params["slow"] = st.slider("SLOW SMA", 10, 400, 60, 1)
                params["vol_window"] = st.slider("Vol window", 10, 120, 20, 5)
                params["vol_cap"] = st.slider("Vol cap (ann.)", 0.10, 1.00, 0.30, 0.01)

    else:
        default_sel = ["Buy & Hold", "SMA Crossover", "MACD Trend", "RSI Mean Reversion"]
        selected = st.multiselect("Select strategies to compare", list(STRATEGIES.keys()), default=default_sel)

    st.divider()
    run = st.button("Run Backtest", type="primary")


def load_data():
    if market.startswith("US"):
        df = fetch_stooq_us_daily(ticker, start_d, end_d)
        src = "stooq"
        if df.empty and use_yahoo and YF_OK:
            df = fetch_yfinance_daily(ticker, start_d, end_d, auto_adjust=auto_adj)
            src = "yfinance" if not df.empty else src
        title = f"{ticker} (US) — source={src}"
        return df, title

    # Korea
    if not (code6.isdigit() and len(code6) == 6):
        return pd.DataFrame(), "KR code must be 6 digits"

    display = chosen_name or code6

    if kr_source.startswith("pykrx"):
        df = fetch_pykrx_daily(code6, start_d, end_d, adjusted=auto_adj)
        title = f"{display} ({code6}) — source=pykrx"
        return df, title

    if kr_source.startswith("FinanceDataReader"):
        df = fetch_fdr_korea_daily(code6, start_d, end_d)
        title = f"{display} ({code6}) — source=fdr"
        return df, title

    sym = f"{code6}{infer_yahoo_suffix(code6, listing_df)}"
    df = fetch_yfinance_daily(sym, start_d, end_d, auto_adjust=auto_adj)
    title = f"{display} ({sym}) — source=yfinance"
    return df, title


if run:
    df, base_title = load_data()
    df = ensure_ohlcv(df)

    if df.empty:
        st.error("데이터를 가져오지 못했습니다. (US는 Stooq가 기본이며, Yahoo는 레이트리밋이 걸릴 수 있어요.)")
        st.stop()

    st.success(f"Loaded {len(df):,} rows | {base_title}")

    if mode == "Single Strategy":
        fn = STRATEGIES[strat_name][0]
        pos = fn(df, **params)

        equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{metrics.get('Total Return', np.nan)*100:,.2f}%")
        c2.metric("CAGR", f"{metrics.get('CAGR', np.nan)*100:,.2f}%")
        c3.metric("Sharpe", f"{metrics.get('Sharpe (rf=0)', np.nan):,.2f}")
        c4.metric("Max DD", f"{metrics.get('Max Drawdown', np.nan)*100:,.2f}%")
        c5.metric("Trades", f"{int(metrics.get('Trades', 0)):,}")

        tab1, tab2, tab3, tab4 = st.tabs(["Price & Signals", "Equity", "Trades", "Metrics"])

        with tab1:
            st.plotly_chart(plot_price_with_signals(df, pos, f"{base_title} — {strat_name}"), width="stretch")
            with st.expander("Raw data (tail)"):
                st.dataframe(df.tail(300))

        with tab2:
            st.plotly_chart(plot_equity(equity, "Equity Curve (start=1.0)"), width="stretch")
            st.plotly_chart(plot_drawdown(equity, "Drawdown"), width="stretch")

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
            fn = STRATEGIES[name][0]
            pos = fn(df)
            equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)
            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=name))
            row = {"Strategy": name}
            row.update(metrics)
            results.append(row)

        fig.update_layout(title=f"{base_title} — Strategy Comparison (Equity start=1.0)", height=520)

        st.plotly_chart(fig, width="stretch")

        res_df = pd.DataFrame(results)
        for col in ["Total Return", "CAGR", "Max Drawdown", "Win Rate", "Avg Trade Return"]:
            if col in res_df.columns:
                res_df[col] = (res_df[col] * 100).round(2)
        if "Sharpe (rf=0)" in res_df.columns:
            res_df["Sharpe (rf=0)"] = res_df["Sharpe (rf=0)"].round(2)

        st.subheader("Metrics (percent columns are %)")
        st.dataframe(res_df.sort_values(by="CAGR", ascending=False, na_position="last"))

else:
    st.info(
        "왼쪽에서 Market/종목/기간/전략을 고르고 **Run Backtest**를 누르세요.\n\n"
        "US는 기본이 Stooq라서 Yahoo 레이트리밋을 대부분 피합니다."
    )
