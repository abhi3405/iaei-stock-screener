"""
IAEI Signal Engine & Universal Screener (Upstox API + Complete NSE/BSE Universe)
==============================================================================
"""

import datetime
import gzip
import io
import json
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & RESPONSIVE UI INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Universal IAEI Screener — NSE/BSE Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive CSS overrides for mobile, tablet, and ultra-wide screens
st.markdown("""
<style>
    /* Metric Card Enhancements */
    .metric-card { 
        background-color: #f8fafc; 
        border: 1px solid #e2e8f0; 
        border-radius: 8px; 
        padding: 12px 16px; 
        margin-bottom: 8px; 
        box-sizing: border-box;
    }
    .metric-title { font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 3px; }
    .status-breakout { color: #16a34a; font-weight: 800; }
    .status-accum { color: #d97706; font-weight: 800; }
    .status-avoid { color: #dc2626; font-weight: 800; }
    .status-take-profit { color: #9333ea; font-weight: 800; }

    /* Streamlit Metric Container Responsive Stacking */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 calc(20% - 1rem) !important;
        min-width: 140px !important;
    }

    @media (max-width: 992px) {
        [data-testid="column"] {
            flex: 1 1 calc(33.33% - 1rem) !important;
        }
    }

    @media (max-width: 640px) {
        [data-testid="column"] {
            flex: 1 1 calc(50% - 0.5rem) !important;
            min-width: 120px !important;
        }
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
        }
    }

    /* Keep Plotly chart container responsive without horizontal spill */
    .js-plotly-plot, .plot-container {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. UPSTOX DYNAMIC INSTRUMENT RESOLVER
# -----------------------------------------------------------------------------
UPSTOX_NSE_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_BSE_MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz"

@st.cache_data(ttl=86400, show_spinner=False)
def load_all_exchange_instruments(exchange="NSE"):
    url = UPSTOX_NSE_MASTER_URL if exchange == "NSE" else UPSTOX_BSE_MASTER_URL
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
                data = json.loads(gz.read().decode("utf-8"))
                target_segment = f"{exchange}_EQ"
                equities = [
                    {
                        "trading_symbol": item.get("trading_symbol"),
                        "name": item.get("name") or item.get("trading_symbol"),
                        "instrument_key": item.get("instrument_key"),
                        "isin": item.get("isin")
                    }
                    for item in data
                    if item.get("segment") == target_segment and item.get("instrument_type") == "EQ"
                ]
                df_inst = pd.DataFrame(equities).drop_duplicates(subset=["trading_symbol"])
                return df_inst.sort_values("trading_symbol").reset_index(drop=True)
    except Exception:
        pass

    fallback_catalog = [
        {"trading_symbol": "RELIANCE", "name": "Reliance Industries Ltd", "instrument_key": "NSE_EQ|INE002A01018"},
        {"trading_symbol": "TCS", "name": "Tata Consultancy Services Ltd", "instrument_key": "NSE_EQ|INE467B01029"}
    ]
    return pd.DataFrame(fallback_catalog)

# -----------------------------------------------------------------------------
# 3. LIVE HISTORICAL CANDLE FETCHER
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_upstox_historical_candles(instrument_key: str, access_token: str = None, days_back: int = 180):
    to_date = datetime.date.today().strftime('%Y-%m-%d')
    from_date = (datetime.date.today() - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}"
    headers = {"Accept": "application/json"}
    
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
        
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            raw_candles = resp.json()["data"]["candles"]
            if raw_candles:
                df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                return df
    except Exception:
        pass

    # Fallback to mock data if no token is provided or API fails
    return generate_mock_candle_data(days=days_back)

def generate_mock_candle_data(days=120):
    np.random.seed(int(datetime.datetime.now().timestamp()) % 1000)
    dates = pd.date_range(end=datetime.date.today(), periods=days, freq="B")
    prices = [100.0]
    volumes = []
    for i in range(1, days):
        change = np.random.normal(0.25, 1.2)
        prices.append(max(5.0, prices[-1] + change))
        volumes.append(int(max(10000, np.random.normal(200000, 40000))))
    volumes.insert(0, 200000)
    close = np.array(prices)
    high = close + np.random.uniform(0.5, 2.5, size=days)
    low = close - np.random.uniform(0.5, 2.0, size=days)
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, size=days)
    return pd.DataFrame({"timestamp": dates, "open": open_p, "high": high, "low": low, "close": close, "volume": volumes})

# -----------------------------------------------------------------------------
# 4. CORE IAEI COMPUTATION ENGINE
# -----------------------------------------------------------------------------
def calculate_iaei(df: pd.DataFrame, lookback: int = 20, pe_weight: float = 40.0, va_weight: float = 40.0, vc_weight: float = 20.0, bb_mult: float = 2.0, kc_mult: float = 1.5, ema_smoothing: int = 3) -> pd.DataFrame:
    res = df.copy()
    close, high, low, volume = res["close"], res["high"], res["low"], res["volume"]

    # Price Efficiency, Volume Absorption, Volatility Compression
    net_change = (close - close.shift(lookback)).abs()
    path = (close - close.shift(1)).abs().rolling(lookback).sum()
    pe = (net_change / path.replace(0, np.nan)).fillna(0).clip(0, 1)

    hl_range = (high - low).replace(0, np.nan)
    clv = (((close - low) - (high - close)) / hl_range).fillna(0)
    vol_sma = volume.rolling(lookback).mean().replace(0, np.nan)
    vol_ratio = (volume / vol_sma).fillna(1.0)
    va = (clv * vol_ratio).rolling(lookback).mean()
    va_norm = ((va + 1.0) / 2.0).clip(0, 1)

    bb_width = 2.0 * bb_mult * close.rolling(lookback).std()
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    kc_width = 2.0 * kc_mult * pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(lookback).mean()
    vc = ((kc_width - bb_width) / kc_width.replace(0, np.nan)).clip(lower=0).fillna(0)

    res["PE_Score"] = pe * pe_weight
    res["VA_Score"] = va_norm * va_weight
    res["VC_Score"] = vc.clip(0, 1) * vc_weight
    raw_iaei = res["PE_Score"] + res["VA_Score"] + res["VC_Score"]
    res["IAEI"] = raw_iaei.ewm(span=ema_smoothing, adjust=False).mean()

    res["SMA_20"] = close.rolling(20).mean()
    res["SMA_50"] = close.rolling(50).mean()
    res["SMA_200"] = close.rolling(200).mean()
    res["Vol_SMA"] = vol_sma
    res["10d_High"] = high.rolling(10).max().shift(1)

    def classify(r):
        sc, c, h10 = r["IAEI"], r["close"], r["10d_High"]
        vr = r["volume"] / r["Vol_SMA"] if r["Vol_SMA"] > 0 else 1.0
        if sc >= 85: return "TAKE PROFIT / CLIMAX"
        if sc >= 68 and c >= h10 and vr >= 1.25: return "BUY BREAKOUT"
        if sc >= 58 and abs(c - r["SMA_50"]) / r["SMA_50"] < 0.06: return "ACCUMULATION WATCH"
        if sc < 42: return "AVOID / EXIT"
        return "NEUTRAL"

    res["Signal"] = res.apply(classify, axis=1)
    return res

# -----------------------------------------------------------------------------
# 5. SIDEBAR CONTROLS & UNIVERSAL INSTRUMENT PICKER
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ Upstox Universal Engine")
selected_exchange = st.sidebar.radio("Select Exchange", ["NSE", "BSE"], horizontal=True)
upstox_token = st.sidebar.text_input("Upstox Access Token (Optional)", type="password", help="Paste daily token from Upstox to enable live data. Otherwise, mock data is used.")

with st.spinner(f"Loading complete {selected_exchange} master universe..."):
    master_df = load_all_exchange_instruments(selected_exchange)

st.sidebar.markdown(f"**Indexed Equities Available:** `{len(master_df):,}` stocks")
search_mode = st.sidebar.radio("Mode", ["Single Stock Deep-Dive", "Multi-Stock Screener"])

if search_mode == "Single Stock Deep-Dive":
    stock_options = [f"{row['trading_symbol']} — {row['name']}" for _, row in master_df.iterrows()]
    selected_option = st.sidebar.selectbox("Search & Select Any Stock", stock_options, index=0)
    selected_symbol = selected_option.split(" — ")[0]
    matched_row = master_df[master_df["trading_symbol"] == selected_symbol].iloc[0]
    chosen_instrument_key = matched_row["instrument_key"]
    stock_full_name = matched_row["name"]
else:
    st.sidebar.subheader("Select Universe to Scan")
    scan_preset = st.sidebar.selectbox("Preset Lists", ["Top 20 Active Movers", "Custom Stock List"])
    if scan_preset == "Custom Stock List":
        custom_input = st.sidebar.text_area("Enter Symbols (comma separated)", value="RELIANCE, TCS")
        selected_symbols = [s.strip().upper() for s in custom_input.split(",") if s.strip()]
    else:
        selected_symbols = master_df["trading_symbol"].head(20).tolist()

st.sidebar.markdown("---")
lookback_len = st.sidebar.slider("Lookback (N)", 10, 50, 20)
efficiency_wt = st.sidebar.slider("Efficiency Wt (PE)", 10.0, 50.0, 40.0, 5.0)
absorption_wt = st.sidebar.slider("Absorption Wt (VA)", 10.0, 50.0, 40.0, 5.0)
squeeze_wt = st.sidebar.slider("Squeeze Wt (VC)", 5.0, 30.0, 20.0, 5.0)

# -----------------------------------------------------------------------------
# 6. MAIN DISPLAY WORKFLOW
# -----------------------------------------------------------------------------
if search_mode == "Single Stock Deep-Dive":
    st.title(f"IAEI Analysis: {selected_symbol}")
    st.caption(f"{stock_full_name} | Exchange: {selected_exchange} | Instrument Key: `{chosen_instrument_key}`")

    df_raw = fetch_upstox_historical_candles(chosen_instrument_key, upstox_token, days_back=180)
    df = calculate_iaei(df_raw, lookback=lookback_len, pe_weight=efficiency_wt, va_weight=absorption_wt, vc_weight=squeeze_wt)
    latest, prev = df.iloc[-1], df.iloc[-2]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("LTP", f"₹{latest['close']:,.2f}", f"{((latest['close']-prev['close'])/prev['close'])*100:+.2f}%")
    with c2: st.metric("Composite IAEI", f"{latest['IAEI']:.1f} / 100")
    with c3: st.metric("Signal State", latest["Signal"])
    with c4: st.metric("Vol Multiplier", f"{latest['volume'] / latest['Vol_SMA'] if latest['Vol_SMA'] > 0 else 1.0:.2f}x")
    with c5: st.metric("Dist from 50 SMA", f"{((latest['close'] - latest['SMA_50']) / latest['SMA_50']) * 100 if not np.isnan(latest['SMA_50']) else 0.0:+.1f}%")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.30, 0.15])
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SMA_50"], name="50 SMA"), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["IAEI"], name="IAEI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], name="Volume"), row=3, col=1)
    
    fig.update_layout(
        autosize=True,
        height=680,
        margin=dict(l=10, r=10, t=25, b=25),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

else:
    st.title("Universal IAEI Screener")
    scan_results = []
    progress_bar = st.progress(0)

    for idx, sym in enumerate(selected_symbols):
        progress_bar.progress((idx + 1) / len(selected_symbols))
        matched = master_df[master_df["trading_symbol"] == sym]
        if matched.empty: continue
        df_stock = fetch_upstox_historical_candles(matched.iloc[0]["instrument_key"], upstox_token, days_back=120)
        analyzed = calculate_iaei(df_stock, lookback_len, efficiency_wt, absorption_wt, squeeze_wt)
        lt = analyzed.iloc[-1]
        
        scan_results.append({
            "Symbol": sym, "LTP (₹)": round(lt["close"], 2), "IAEI Score": round(lt["IAEI"], 1), "Signal": lt["Signal"]
        })

    progress_bar.empty()
    res_df = pd.DataFrame(scan_results)
    if not res_df.empty:
        st.dataframe(res_df.sort_values(by="IAEI Score", ascending=False).reset_index(drop=True), use_container_width=True)