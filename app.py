import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
import time
import unicodedata
from pathlib import Path
from datetime import date, timedelta
import plotly.graph_objects as go
from difflib import SequenceMatcher
from typing import Optional, Tuple

# =========================
# Optional deps (KR only)
# =========================
try:
    import FinanceDataReader as fdr  # pip install finance-datareader
    FDR_OK = True
except Exception:
    FDR_OK = False

try:
    from pykrx import stock as krx_stock  # pip install pykrx
    PYKRX_OK = True
except Exception:
    PYKRX_OK = False

try:
    import requests  # pip install requests
    REQ_OK = True
except Exception:
    REQ_OK = False


# =========================
# Settings (edit these)
# =========================
# 1) 레포에 CSV를 같이 넣으면(추천): 이 파일명이 로컬에서 바로 읽힘
LOCAL_SYMBOL_CSV = "krx_all_symbols.csv"

# 2) GitHub Raw URL로 자동 다운로드하려면 여기에 Raw 링크를 넣어줘
# 예: https://raw.githubusercontent.com/<USER>/<REPO>/main/krx_all_symbols.csv
DEFAULT_GITHUB_RAW_URL = ""

# 3) 로컬 캐시(다운받은 CSV 저장)
CACHE_SYMBOL_CSV = Path("krx_symbols_cache.csv")


# -----------------------------
# Plotly helper
# -----------------------------
def st_plotly(fig, stretch=True):
    try:
        st.plotly_chart(fig, width="stretch" if stretch else "content")
    except TypeError:
        st.plotly_chart(fig, use_container_width=stretch)


# ============================================================
# Text normalization (Korean search robustness)
# ============================================================
def normalize_name(x: str) -> str:
    x = unicodedata.normalize("NFKC", str(x))
    x = x.strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^\w가-힣]", "", x)  # keep Korean/English/digits/_
    return x

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


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


# ============================================================
# Symbol table (Name <-> 6-digit code)
# ============================================================
def coerce_symbol_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    표준화: Name, Symbol(6자리 숫자), NameNorm
    - Symbol에 문자가 섞인 항목은 가격 데이터 라이브러리에서 깨질 수 있어 기본적으로 제외
    Returns: (clean_df, dropped_count)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Name", "Symbol", "NameNorm"]), 0

    # Expect columns Name, Symbol (너 CSV는 이 형식)
    if "Name" not in df.columns or "Symbol" not in df.columns:
        # 최소한의 자동 감지(혹시 컬럼명이 다를 때)
        cols = {c: str(c) for c in df.columns}
        name_col = None
        sym_col = None
        for c in df.columns:
            if str(c).strip() in ["Name", "종목명", "회사명"]:
                name_col = c
            if str(c).strip() in ["Symbol", "Code", "종목코드", "코드", "Ticker", "티커"]:
                sym_col = c
        if not name_col or not sym_col:
            return pd.DataFrame(columns=["Name", "Symbol", "NameNorm"]), 0
        df = df[[name_col, sym_col]].copy()
        df.columns = ["Name", "Symbol"]
    else:
        df = df[["Name", "Symbol"]].copy()

    df["Name"] = df["Name"].astype(str).map(lambda x: unicodedata.normalize("NFKC", x).strip())
    df["Symbol"] = df["Symbol"].astype(str).str.strip()

    # 6자리 숫자만 남김
    keep = df["Symbol"].str.fullmatch(r"\d{6}")
    dropped = int((~keep).sum())
    df = df[keep].copy()

    df["NameNorm"] = df["Name"].map(normalize_name)
    df = df.drop_duplicates()
    return df, dropped


@st.cache_data(ttl=3600, show_spinner=False)
def load_symbol_table_from_local(path: str) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(path)
    return coerce_symbol_table(df)

@st.cache_data(ttl=3600, show_spinner=False)
def load_symbol_table_from_bytes(b: bytes) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(io.BytesIO(b))
    return coerce_symbol_table(df)

@st.cache_data(ttl=3600, show_spinner=False)
def load_symbol_table_from_url(url: str, token: str = "") -> Tuple[pd.DataFrame, int]:
    """
    GitHub Raw URL에서 CSV 다운로드.
    Private repo면 token 필요(권장: st.secrets["GITHUB_TOKEN"]).
    """
    if not REQ_OK:
        return pd.DataFrame(columns=["Name", "Symbol", "NameNorm"]), 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"token {token}"

    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    # CSV는 보통 UTF-8-sig/UTF-8
    content = r.content
    return load_symbol_table_from_bytes(content)


def search_candidates(symtbl: pd.DataFrame, user_input: str, topn: int = 80) -> pd.DataFrame:
    s = (user_input or "").strip()
    if symtbl is None or symtbl.empty or not s:
        return pd.DataFrame(columns=["Name", "Symbol", "NameNorm"])

    # 6자리 코드 직접 입력
    if s.isdigit() and len(s) == 6:
        return symtbl[symtbl["Symbol"] == s].head(topn)

    sn = normalize_name(s)

    # 1) exact normalized
    exact = symtbl[symtbl["NameNorm"] == sn].copy()
    if not exact.empty:
        return exact.head(topn)

    # 2) contains original
    hits1 = symtbl[symtbl["Name"].str.contains(s, na=False, regex=False)].copy()
    # 3) contains normalized
    hits2 = symtbl[symtbl["NameNorm"].str.contains(sn, na=False, regex=False)].copy()

    hits = pd.concat([hits1, hits2], ignore_index=True).drop_duplicates()
    if not hits.empty:
        return hits.head(topn)

    # 4) fuzzy
    if len(sn) < 2:
        return pd.DataFrame(columns=["Name", "Symbol", "NameNorm"])
    tmp = symtbl[["Name", "Symbol", "NameNorm"]].copy()
    tmp["score"] = tmp["NameNorm"].map(lambda x: similarity(sn, x))
    tmp = tmp.sort_values("score", ascending=False)
    tmp = tmp[tmp["score"] >= 0.50].head(topn)
    return tmp.drop(columns=["score"], errors="ignore")


def resolve_input_to_code(symtbl: pd.DataFrame, user_input: str):
    s = (user_input or "").strip()
    if s.isdigit() and len(s) == 6:
        return s, None, None

    cands = search_candidates(symtbl, s, topn=80)
    if cands.empty:
        return None, None, cands
    if len(cands) == 1:
        return cands.iloc[0]["Symbol"], cands.iloc[0]["Name"], cands
    return None, None, cands


# ============================================================
# Data fetch (KRX only)
# ============================================================
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
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
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
# Backtest engine
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

def run_backtest(df: pd.DataFrame, pos: pd.Series, fee_bps: float, slippage_bps: float):
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
# Strategies (long/flat)
# ============================================================
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
    "Buy & Hold": {"fn": strat_buyhold, "desc": "항상 보유(기준선).", "params": []},
    "SMA Crossover": {
        "fn": strat_sma_cross, "desc": "FAST SMA > SLOW SMA이면 보유.",
        "params": [
            dict(name="fast", label="FAST SMA", kind="int", min=5, max=200, step=1, default=20),
            dict(name="slow", label="SLOW SMA", kind="int", min=10, max=400, step=1, default=60),
        ],
    },
    "MACD Trend": {
        "fn": strat_macd_trend, "desc": "MACD > Signal이면 보유.",
        "params": [
            dict(name="fast", label="MACD fast EMA", kind="int", min=3, max=50, step=1, default=12),
            dict(name="slow", label="MACD slow EMA", kind="int", min=10, max=120, step=1, default=26),
            dict(name="signal", label="Signal EMA", kind="int", min=3, max=30, step=1, default=9),
        ],
    },
    "RSI Mean Reversion": {
        "fn": strat_rsi_reversion, "desc": "RSI<LOW면 진입, RSI>HIGH면 청산.",
        "params": [
            dict(name="n", label="RSI period", kind="int", min=5, max=50, step=1, default=14),
            dict(name="low", label="Entry (LOW)", kind="int", min=5, max=45, step=1, default=30),
            dict(name="high", label="Exit (HIGH)", kind="int", min=55, max=95, step=1, default=70),
        ],
    },
    "Bollinger Mean Reversion": {
        "fn": strat_bollinger_reversion, "desc": "하단밴드 이탈 진입, 중단선 회귀 청산.",
        "params": [
            dict(name="n", label="BB period", kind="int", min=5, max=60, step=1, default=20),
            dict(name="k", label="Std multiplier (k)", kind="float", min=1.0, max=4.0, step=0.1, default=2.0),
        ],
    },
    "Donchian Breakout": {
        "fn": strat_donchian_breakout, "desc": "채널 상단 돌파 진입, 하단 이탈 청산.",
        "params": [dict(name="n", label="Donchian window", kind="int", min=5, max=120, step=1, default=20)],
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
        fig.add_trace(go.Scatter(x=entries, y=c.loc[entries], mode="markers", name="BUY",
                                 marker=dict(symbol="triangle-up", size=10)))
    if len(exits) > 0:
        fig.add_trace(go.Scatter(x=exits, y=c.loc[exits], mode="markers", name="SELL",
                                 marker=dict(symbol="triangle-down", size=10)))
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
# Latest Signal helper
# ============================================================
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
# App UI
# ============================================================
st.set_page_config(page_title="KR Strategy Backtester", layout="wide")
st.title("📈 Korea Stock Strategy Backtester (KRX only)")

with st.sidebar:
    st.header("Cache")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared")

    st.divider()
    st.header("한글 검색용 종목 테이블")

    # GitHub URL (optional)
    github_raw_url = st.text_input(
        "GitHub Raw CSV URL (선택)",
        value=(st.secrets.get("GITHUB_RAW_URL", DEFAULT_GITHUB_RAW_URL) if hasattr(st, "secrets") else DEFAULT_GITHUB_RAW_URL)
    )
    github_token = ""
    if hasattr(st, "secrets"):
        github_token = st.secrets.get("GITHUB_TOKEN", "")

    st.caption(
        "추천: CSV를 레포에 같이 넣고(LOCAL_SYMBOL_CSV), 필요하면 Raw URL로 자동 갱신."
    )

    uploaded_tbl = st.file_uploader("CSV 업로드 (백업용)", type=["csv"])

    colA, colB = st.columns(2)
    refresh_from_github = colA.button("GitHub에서 갱신")
    use_github = colB.checkbox("GitHub 사용", value=bool(github_raw_url))

    # Build symbol table (priority: local repo file -> cache -> github (if enabled) -> upload)
    symtbl = pd.DataFrame(columns=["Name", "Symbol", "NameNorm"])
    sym_source = "None"
    dropped_non_numeric = 0
    debug_err = ""

    # 1) Local file in repo
    if os.path.exists(LOCAL_SYMBOL_CSV):
        try:
            symtbl, dropped_non_numeric = load_symbol_table_from_local(LOCAL_SYMBOL_CSV)
            if not symtbl.empty:
                sym_source = f"Local file: {LOCAL_SYMBOL_CSV}"
        except Exception as e:
            debug_err = f"Local load failed: {e}"

    # 2) Cache file
    if symtbl.empty and CACHE_SYMBOL_CSV.exists():
        try:
            symtbl, dropped_non_numeric = load_symbol_table_from_local(str(CACHE_SYMBOL_CSV))
            if not symtbl.empty:
                sym_source = f"Cache file: {CACHE_SYMBOL_CSV}"
        except Exception as e:
            debug_err = f"Cache load failed: {e}"

    # 3) GitHub download (on demand or when enabled and no local)
    def github_fetch_to_cache():
        nonlocal symtbl, sym_source, dropped_non_numeric, debug_err
        if not use_github or not github_raw_url:
            return
        if not REQ_OK:
            debug_err = "requests가 설치되어 있지 않아 GitHub 다운로드를 할 수 없습니다."
            return
        try:
            df_loaded, dropped = load_symbol_table_from_url(github_raw_url, token=github_token)
            if df_loaded is None or df_loaded.empty:
                debug_err = "GitHub CSV를 읽었지만 테이블이 비었습니다."
                return
            # save raw to cache (save original columns Name/Symbol only)
            to_save = df_loaded[["Name", "Symbol"]].copy()
            CACHE_SYMBOL_CSV.parent.mkdir(parents=True, exist_ok=True)
            to_save.to_csv(CACHE_SYMBOL_CSV, index=False, encoding="utf-8-sig")
            symtbl = df_loaded
            dropped_non_numeric = dropped
            sym_source = "GitHub Raw URL (cached)"
            debug_err = ""
        except Exception as e:
            debug_err = f"GitHub fetch failed: {e}"

    if refresh_from_github:
        github_fetch_to_cache()
    elif symtbl.empty and use_github and github_raw_url:
        # If no local/cache, try once
        github_fetch_to_cache()

    # 4) Upload CSV fallback
    if symtbl.empty and uploaded_tbl is not None:
        try:
            symtbl, dropped_non_numeric = load_symbol_table_from_bytes(uploaded_tbl.getvalue())
            if not symtbl.empty:
                sym_source = "Uploaded CSV"
        except Exception as e:
            debug_err = f"Upload load failed: {e}"

    st.write("Source:", sym_source)
    st.write("Rows:", int(len(symtbl)))
    if dropped_non_numeric > 0:
        st.info(f"참고: Symbol이 6자리 숫자가 아닌 항목 {dropped_non_numeric}개는 제외했어(데이터 소스 호환 때문).")

    with st.expander("Debug", expanded=False):
        st.write("REQ_OK:", REQ_OK, "| PYKRX_OK:", PYKRX_OK, "| FDR_OK:", FDR_OK)
        st.write("LOCAL_SYMBOL_CSV exists:", os.path.exists(LOCAL_SYMBOL_CSV))
        st.write("CACHE_SYMBOL_CSV exists:", CACHE_SYMBOL_CSV.exists())
        if debug_err:
            st.code(debug_err)

    st.divider()
    st.header("종목 입력")
    stock_input = st.text_input("6자리 코드 또는 한글 종목명", value="삼성전자")

    if symtbl.empty:
        st.error("종목 테이블이 비어있어서 한글 검색이 불가능합니다. (로컬 CSV 포함 또는 GitHub Raw URL/업로드 필요)")
        st.stop()

    code6, chosen_name, candidates = resolve_input_to_code(symtbl, stock_input)

    if code6 is None:
        if candidates is None or candidates.empty:
            st.error("종목명을 찾지 못했어요. (오타/띄어쓰기 확인) 또는 코드(6자리)로 입력하세요.")
            st.stop()
        if len(candidates) == 1:
            code6 = candidates.iloc[0]["Symbol"]
            chosen_name = candidates.iloc[0]["Name"]
        else:
            options = (candidates["Symbol"] + " — " + candidates["Name"]).tolist()
            pick = st.selectbox("검색 결과(여러 개면 선택)", options)
            code6 = pick.split(" — ")[0].strip()
            chosen_name = pick.split(" — ")[1].strip()

    st.divider()
    st.header("기간")
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
    st.header("데이터 소스 (KRX)")
    sources = []
    if PYKRX_OK: sources.append("pykrx (KRX)")
    if FDR_OK: sources.append("FinanceDataReader (KRX)")
    if not sources:
        st.error("데이터 소스가 없습니다. pykrx / finance-datareader 중 하나 설치하세요.")
        st.stop()

    source = st.selectbox("Data Source", sources)
    adjusted = st.checkbox("Adjusted price (if supported)", value=True)

    st.divider()
    st.header("Backtest Assumptions")
    fee_bps = st.number_input("Fee (bps per trade)", 0.0, 200.0, 5.0, 1.0)
    slip_bps = st.number_input("Slippage (bps per trade)", 0.0, 200.0, 0.0, 1.0)

    st.divider()
    mode = st.radio("Mode", ["Single Strategy", "Compare Strategies"], index=0)

    st.divider()
    st.header("전략")
    if mode == "Single Strategy":
        strat_name = st.selectbox("Choose Strategy", list(STRATEGIES.keys()))
        st.caption(STRATEGIES[strat_name]["desc"])
        params = {}
        with st.expander("Parameters", expanded=True):
            for p in STRATEGIES[strat_name]["params"]:
                if p["kind"] == "int":
                    params[p["name"]] = st.slider(
                        p["label"], int(p["min"]), int(p["max"]), int(p["default"]), int(p["step"])
                    )
                else:
                    params[p["name"]] = st.slider(
                        p["label"], float(p["min"]), float(p["max"]), float(p["default"]), float(p["step"])
                    )
    else:
        selected = st.multiselect(
            "Select strategies to compare",
            list(STRATEGIES.keys()),
            default=["Buy & Hold", "SMA Crossover", "MACD Trend", "RSI Mean Reversion"]
        )
        strat_name = None
        params = None

    st.divider()
    run = st.button("Run", type="primary")


def load_korea_data(code6_: str):
    display = chosen_name or code6_
    if source.startswith("pykrx"):
        df = fetch_pykrx_daily(code6_, start_d, end_d, adjusted=adjusted)
        return df, f"{display} ({code6_}) — pykrx"
    df = fetch_fdr_korea_daily(code6_, start_d, end_d)
    return df, f"{display} ({code6_}) — FDR"


# ============================================================
# Run
# ============================================================
if run:
    df, title = load_korea_data(code6)
    df = ensure_ohlcv(df)

    if df.empty:
        st.error(f"데이터를 가져오지 못했습니다.\nDetail: {title}")
        st.stop()

    st.success(f"Loaded {len(df):,} rows | {title}")

    if mode == "Single Strategy":
        fn = STRATEGIES[strat_name]["fn"]
        pos = fn(df, **(params or {}))

        action, detail = latest_signal_from_pos(pos)
        st.metric("Latest Signal", action)
        st.caption(f"{detail} | 신호는 종가 기준, 체결은 다음 거래일 가정")

        equity, strat_ret, trades, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return", f"{metrics.get('Total Return', np.nan)*100:,.2f}%")
        c2.metric("CAGR", f"{metrics.get('CAGR', np.nan)*100:,.2f}%")
        c3.metric("Sharpe", f"{metrics.get('Sharpe (rf=0)', np.nan):,.2f}")
        c4.metric("Max DD", f"{metrics.get('Max Drawdown', np.nan)*100:,.2f}%")
        c5.metric("Trades", f"{int(metrics.get('Trades', 0)):,}")

        tab1, tab2, tab3, tab4 = st.tabs(["Price & Signals", "Equity", "Trades", "Metrics"])
        with tab1:
            st_plotly(plot_price_with_signals(df, pos, f"{title} — {strat_name}"), stretch=True)
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
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))

    else:
        if not selected:
            st.warning("Select at least one strategy.")
            st.stop()

        results = []
        fig = go.Figure()

        for name in selected:
            fn = STRATEGIES[name]["fn"]
            pos = fn(df)
            action, _ = latest_signal_from_pos(pos)

            equity, _, _, metrics = run_backtest(df, pos, fee_bps=fee_bps, slippage_bps=slip_bps)
            fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name=name))

            row = {"Strategy": name, "Latest Signal": action}
            row.update(metrics)
            results.append(row)

        fig.update_layout(title=f"{title} — Strategy Comparison (Equity start=1.0)", height=520)
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
    st.info("왼쪽에서 종목(한글/코드)과 기간/전략을 고르고 **Run**을 누르세요.")
