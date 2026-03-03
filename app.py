import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go

# Optional deps
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    import FinanceDataReader as fdr
    FDR_OK = True
except Exception:
    FDR_OK = False

try:
    from pykrx import stock as krx_stock
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False


# =========================
# Indicators
# =========================
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


# =========================
# Data fetch
# =========================
@st.cache_data(show_spinner=False)
def krx_listing():
    if FDR_OK:
        try:
            df = fdr.StockListing("KRX")
            return df
        except Exception:
            return None
    return None

def infer_yahoo_suffix(code6: str, listing_df: pd.DataFrame | None):
    # default heuristics: KOSPI .KS, KOSDAQ .KQ
    if listing_df is None:
        return ".KS"
    hit = listing_df[listing_df["Symbol"].astype(str) == str(code6)]
    if len(hit) == 0:
        return ".KS"
    market = str(hit.iloc[0].get("Market", "")).upper()
    if "KOSDAQ" in market:
        return ".KQ"
    return ".KS"

@st.cache_data(show_spinner=False)
def fetch_yfinance(symbol: str, start: date, end: date, auto_adjust: bool = True) -> pd.DataFrame:
    if not YF_OK:
        return pd.DataFrame()
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

@st.cache_data(show_spinner=False)
def fetch_fdr_korea(code6: str, start: date, end: date) -> pd.DataFrame:
    if not FDR_OK:
        return pd.DataFrame()
    df = fdr.DataReader(code6, str(start), str(end))
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index.name = "Date"
    return df.dropna()

@st.cache_data(show_spinner=False)
def fetch_pykrx(code6: str, start: date, end: date, adjusted: bool = True) -> pd.DataFrame:
    if not PYKRX_OK:
        return pd.DataFrame()
    df = krx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        code6,
        adjusted=adjusted
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"})
    df.index.name = "Date"
    return df[["Open","High","Low","Close","Volume"]].dropna()


# =========================
# Backtest core
# =========================
def positions_to_trades(close: pd.Series, pos: pd.Series) -> pd.DataFrame:
    """Create trades table from position series (assumes daily close execution simplification)."""
    pos = pos.fillna(0).astype(float)
    d = pos.diff().fillna(pos)
    entries = d[d > 0].index
    exits = d[d < 0].index

    trades = []
    i = 0
    j = 0
    # Long-only assumption: entry -> next exit
    while i < len(entries):
        entry_dt = entries[i]
        # find first exit after entry
        exit_dt = None
        while j < len(exits) and exits[j] <= entry_dt:
            j += 1
        if j < len(exits):
            exit_dt = exits[j]
            j += 1
        else:
            exit_dt = close.index[-1]

        entry_px = float(close.loc[entry_dt])
        exit_px = float(close.loc[exit_dt])
        ret = exit_px / entry_px - 1.0
        trades.append([entry_dt, exit_dt, entry_px, exit_px, ret])
        i += 1

    if not trades:
        return pd.DataFrame(columns=["Entry","Exit","EntryPx","ExitPx","Return"])

    df = pd.DataFrame(trades, columns=["Entry","Exit","EntryPx","ExitPx","Return"])
    df["Days"] = (df["Exit"] - df["Entry"]).dt.days
    return df

def compute_metrics(equity: pd.Series, strat_ret: pd.Series, trades: pd.DataFrame) -> dict:
    equity = equity.dropna()
    strat_ret = strat_ret.dropna()

    if len(equity) < 2:
        return {}

    n = len(strat_ret)
    ann = 252.0
    total_return = float(equity.iloc[-1] - 1.0)

    # CAGR
    years = n / ann
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

    # Sharpe (rf=0)
    mu = strat_ret.mean()
    sd = strat_ret.std(ddof=0)
    sharpe = float(np.sqrt(ann) * mu / sd) if sd and np.isfinite(sd) else np.nan

    # Max drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = float(dd.min())

    # Win rate
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
    Simplified daily close-to-close backtest:
    - Position is held during day t based on pos.shift(1) (signal at t-1 close).
    - Costs charged when position changes (abs(delta_pos)).
    """
    close = df["Close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos = pos.reindex(df.index).fillna(0.0).astype(float)
    held = pos.shift(1).fillna(0.0)  # enter next bar assumption

    gross = held * ret

    # transaction costs on position changes
    dpos = pos.diff().fillna(pos.abs())
    # bps -> fraction
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = cost_rate * dpos.abs()

    strat_ret = gross - costs
    equity = (1.0 + strat_ret).cumprod()

    trades = positions_to_trades(close, pos)
    metrics = compute_metrics(equity, strat_ret, trades)
    return equity, strat_ret, trades, metrics


# =========================
# Strategies (Long/Flat)
# Each returns position series (0 or 1)
# =========================
def strat_buyhold(df, **params):
    return pd.Series(1.0, index=df.index)

def strat_sma_cross(df, fast=20, slow=60, **params):
    c = df["Close"].astype(float)
    f = sma(c, fast)
    s = sma(c, slow)
    pos = (f > s).astype(float)
    return pos

def strat_ema_cross(df, fast=12, slow=26, **params):
    c = df["Close"].astype(float)
    f = ema(c, fast)
    s = ema(c, slow)
    pos = (f > s).astype(float)
    return pos

def strat_rsi_reversion(df, n=14, low=30, high=70, **params):
    c = df["Close"].astype(float)
    r = rsi(c, n)
    # enter when oversold, exit when overbought
    entry = (r < low).astype(int)
    exit_ = (r > high).astype(int)

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
    exit_ = (c > mid).astype(int)  # mean reversion to mid

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
    # breakout long when close breaks above previous channel high
    entry = (c > up.shift(1)).astype(int)
    # exit when close breaks below previous channel low
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
        "defaults": {},
    },
    "SMA Crossover": {
        "fn": strat_sma_cross,
        "desc": "단순 이동평균(FAST) > (SLOW)일 때 매수.",
        "defaults": {"fast": 20, "slow": 60},
    },
    "EMA Crossover": {
        "fn": strat_ema_cross,
        "desc": "지수 이동평균(FAST) > (SLOW)일 때 매수.",
        "defaults": {"fast": 12, "slow": 26},
    },
    "RSI Mean Reversion": {
        "fn": strat_rsi_reversion,
        "desc": "RSI가 LOW 아래면 진입, HIGH 위면 청산(역추세).",
        "defaults": {"n": 14, "low": 30, "high": 70},
    },
    "Bollinger Mean Reversion": {
        "fn": strat_bollinger_reversion,
        "desc": "하단 밴드 이탈 시 진입, 중단선 회귀 시 청산.",
        "defaults": {"n": 20, "k": 2.0},
    },
    "Donchian Breakout": {
        "fn": strat_donchian_breakout,
        "desc": "n일 채널 상단 돌파 시 진입, 하단 이탈 시 청산(추세추종).",
        "defaults": {"n": 20},
    },
}


# =========================
# Plotting
# =========================
def plot_price_with_signals(df: pd.DataFrame, pos: pd.Series, title: str):
    c = df["Close"].astype(float)
    pos = pos.reindex(df.index).fillna(0.0)
    d = pos.diff().fillna(pos)

    entries = d[d > 0].index
    exits = d[d < 0].index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=c, mode="lines", name="Close"))
    # markers
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
    fig.update_layout(title=title, height=500, xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_equity(equity: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(title=title, height=450, xaxis_title="Date", yaxis_title="Equity (start=1.0)")
    return fig

def plot_drawdown(equity: pd.Series, title: str):
    peak = equity.cummax()
    dd = equity / peak - 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown"))
    fig.update_layout(title=title, height=350, xaxis_title="Date", yaxis_title="Drawdown")
    return fig


# =========================
# UI
# =========================
st.set_page_config(page_title="KR/US Strategy Backtester", layout="wide")
st.title("📊 KR/US Stock Strategy Backtester (Streamlit)")

with st.sidebar:
    st.header("Data")

    market = st.selectbox("Market", ["US (미국)", "Korea (한국)"])
    today = date.today()
    start_default = today - timedelta(days=365 * 3)

    start_d, end_d = st.date_input("Date Range", value=(start_default, today), max_value=today)
    if start_d > end_d:
        st.error("Start date must be <= end date.")
        st.stop()

    st.divider()
    st.header("Backtest Assumptions")
    fee_bps = st.number_input("Fee (bps per trade)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)
    slip_bps = st.number_input("Slippage (bps per trade)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)

    mode = st.radio("Mode", ["Single Strategy", "Compare Strategies"], index=0)

    st.divider()
    st.header("Strategy")

    if mode == "Single Strategy":
        strat_name = st.selectbox("Choose Strategy", list(STRATEGIES.keys()))
        st.caption(STRATEGIES[strat_name]["desc"])

        # Parameter widgets based on strategy
        defaults = STRATEGIES[strat_name]["defaults"].copy()
        params = {}

        with st.expander("Parameters", expanded=True):
            if strat_name in ["SMA Crossover"]:
                params["fast"] = st.slider("FAST SMA", 5, 200, int(defaults["fast"]), 1)
                params["slow"] = st.slider("SLOW SMA", 10, 400, int(defaults["slow"]), 1)
            elif strat_name in ["EMA Crossover"]:
                params["fast"] = st.slider("FAST EMA", 3, 100, int(defaults["fast"]), 1)
                params["slow"] = st.slider("SLOW EMA", 5, 200, int(defaults["slow"]), 1)
            elif strat_name in ["RSI Mean Reversion"]:
                params["n"] = st.slider("RSI period", 5, 50, int(defaults["n"]), 1)
                params["low"] = st.slider("Entry (oversold)", 5, 45, int(defaults["low"]), 1)
                params["high"] = st.slider("Exit (overbought)", 55, 95, int(defaults["high"]), 1)
            elif strat_name in ["Bollinger Mean Reversion"]:
                params["n"] = st.slider("BB period", 5, 60, int(defaults["n"]), 1)
                params["k"] = st.slider("Std multiplier (k)", 1.0, 4.0, float(defaults["k"]), 0.1)
            elif strat_name in ["Donchian Breakout"]:
                params["n"] = st.slider("Donchian window (n)", 5, 120, int(defaults["n"]), 1)
            else:
                st.write("No parameters.")

    else:
        selected = st.multiselect(
            "Select strategies to compare",
            list(STRATEGIES.keys()),
            default=["Buy & Hold", "SMA Crossover", "RSI Mean Reversion"]
        )
        st.caption("Compare 모드는 각 전략의 기본 파라미터(Defaults)로 빠르게 비교합니다.")
        strat_name = None
        params = None

    st.divider()
    st.header("Symbol")

    listing_df = krx_listing() if market.startswith("Korea") else None

    if market.startswith("US"):
        ticker = st.text_input("US Ticker (e.g., AAPL, MSFT, NVDA)", value="AAPL").strip().upper()
        auto_adj = st.checkbox("Auto-adjust (splits/dividends)", value=True)
        source = "yfinance"
    else:
        sources = []
        if YF_OK: sources.append("yfinance (Yahoo)")
        if PYKRX_OK: sources.append("pykrx (KRX)")
        if FDR_OK: sources.append("FinanceDataReader")
        if not sources:
            st.error("Install at least one: yfinance / pykrx / FinanceDataReader")
            st.stop()

        source = st.selectbox("Korea Data Source", sources)
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

    run = st.button("Run Backtest", type="primary")


def load_data():
    if market.startswith("US"):
        if not ticker:
            return pd.DataFrame(), "No ticker"
        df = fetch_yfinance(ticker, start_d, end_d, auto_adjust=auto_adj)
        title = f"{ticker} (US) — Daily"
        return df, title
    else:
        if not (code6.isdigit() and len(code6) == 6):
            return pd.DataFrame(), "KR code must be 6 digits"
        display = chosen_name or code6

        if source.startswith("yfinance"):
            suffix = infer_yahoo_suffix(code6, listing_df)
            sym = f"{code6}{suffix}"
            df = fetch_yfinance(sym, start_d, end_d, auto_adjust=auto_adj)
            title = f"{display} ({sym}) — Daily"
            return df, title
        elif source.startswith("pykrx"):
            df = fetch_pykrx(code6, start_d, end_d, adjusted=auto_adj)
            title = f"{display} ({code6}) — KRX (pykrx) Daily"
            return df, title
        else:
            df = fetch_fdr_korea(code6, start_d, end_d)
            title = f"{display} ({code6}) — KRX (FDR) Daily"
            return df, title


if run:
    df, base_title = load_data()
    if df.empty:
        st.error(f"Failed to fetch data. ({base_title})")
        st.stop()

    # Sanity check
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(set(df.columns)):
        st.error(f"Data missing columns: {needed - set(df.columns)}")
        st.stop()

    df = df.sort_index()
    st.success(f"Loaded {len(df):,} rows")

    if mode == "Single Strategy":
        fn = STRATEGIES[strat_name]["fn"]
        pos = fn(df, **(params or {}))

        equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)

        # Top metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{metrics.get('Total Return', np.nan)*100:,.2f}%")
        c2.metric("CAGR", f"{metrics.get('CAGR', np.nan)*100:,.2f}%")
        c3.metric("Sharpe", f"{metrics.get('Sharpe (rf=0)', np.nan):,.2f}")
        c4.metric("Max DD", f"{metrics.get('Max Drawdown', np.nan)*100:,.2f}%")
        c5.metric("Trades", f"{metrics.get('Trades', 0):,}")

        tab1, tab2, tab3, tab4 = st.tabs(["Price & Signals", "Equity", "Trades", "Metrics"])

        with tab1:
            st.plotly_chart(
                plot_price_with_signals(df, pos, f"{base_title} — {strat_name}"),
                use_container_width=True
            )
            st.dataframe(df.tail(200))

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
            defaults = info["defaults"]
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
        # nicer formatting
        for col in ["Total Return","CAGR","Max Drawdown","Win Rate","Avg Trade Return"]:
            if col in res_df.columns:
                res_df[col] = (res_df[col] * 100).round(2)
        if "Sharpe (rf=0)" in res_df.columns:
            res_df["Sharpe (rf=0)"] = res_df["Sharpe (rf=0)"].round(2)

        st.subheader("Metrics (percent columns are %)")
        st.dataframe(res_df.sort_values(by="CAGR", ascending=False, na_position="last"))

else:
    st.info("왼쪽에서 Market/종목/기간/전략을 고르고 **Run Backtest**를 눌러주세요.")
