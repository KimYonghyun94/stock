import pandas as pd
from datetime import timedelta

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stooq_us_daily_csv(ticker: str, start, end) -> pd.DataFrame:
    """
    Stooq direct CSV:
      https://stooq.com/q/d/l/?s=aapl.us&i=d
    """
    sym = ticker.strip().lower()
    if not sym.endswith(".us"):
        sym = f"{sym}.us"

    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"

    try:
        df = pd.read_csv(url)
        if df is None or df.empty:
            return pd.DataFrame()

        # Stooq columns: Date, Open, High, Low, Close, Volume
        if "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        need = ["Open", "High", "Low", "Close"]
        if not set(need).issubset(df.columns):
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        # filter range (end inclusive)
        df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    except Exception:
        return pd.DataFrame()
