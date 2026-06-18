import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Live NSE RRG Dashboard", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔧 Settings Panel")

BENCHMARK_LABEL = st.sidebar.selectbox("Select Benchmark Index", ["Nifty 50", "Nifty 100", "Nifty 200"])
BENCHMARK_MAP = {"Nifty 50": "^NSEI", "Nifty 100": "^CNX100", "Nifty 200": "^CNX200"}
BENCHMARK = BENCHMARK_MAP[BENCHMARK_LABEL]

TAIL_LENGTH = st.sidebar.slider("Counts (Tail Days Path)", min_value=3, max_value=15, value=5, step=1)

# --- NEW MULTI-SELECT STOCK ENGINE ---
st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Stock Selection Engine")

# Extended options menu covering multiple sectors
AVAILABLE_STOCKS = {
    "Reliance": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "TCS": "TCS.NS", "Infosys": "INFY.NS", "Bharti Airtel": "BHARTIARTL.NS",
    "L&T": "LT.NS", "SBI": "SBIN.NS", "ITC": "ITC.NS", "Tata Motors": "TATAMOTORS.NS",
    "Maruti": "MARUTI.NS", "Sun Pharma": "SUNPHARMA.NS", "Adani Ent": "ADANIENT.NS",
    "Tata Steel": "TATASTEEL.NS", "Axis Bank": "AXISBANK.NS", "Kotak Bank": "KOTAKBANK.NS",
    "NTPC": "NTPC.NS", "M&M": "M&M.NS", "UltraTech": "ULTRACEMCO.NS", "HCL Tech": "HCLTECH.NS"
}

selected_stock_labels = st.sidebar.multiselect(
    "Choose Stocks to Monitor:",
    options=list(AVAILABLE_STOCKS.keys()),
    default=["Reliance", "HDFC Bank", "ICICI Bank", "TCS", "Infosys", "SBI"] # Clean pre-loaded default
)

# Rebuild custom dictionaries on the fly based on user choices
STOCK_TICKERS = {label: AVAILABLE_STOCKS[label] for label in selected_stock_labels}

# Custom manual search input box
custom_ticker_input = st.sidebar.text_input("Add Any Other NSE Ticker (e.g., ADANIPORTS):", "").strip().upper()
if custom_ticker_input:
    formatted_ticker = custom_ticker_input if custom_ticker_input.endswith(".NS") else f"{custom_ticker_input}.NS"
    STOCK_TICKERS[custom_ticker_input] = formatted_ticker

st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")

INDEX_SECTORS = {
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Infra": "^CNXINFRA",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Midcap 100": "^CRSMID",
    "Nifty Smallcap 100": "^CNXSMALL",
    "Nifty Defence (ETF)": "DEFENCE.NS",    
    "Nifty Real Estate": "^CNXREALTY",      
    "Gold BeES (Commodity)": "GOLDBEES.NS",  
    "Silver BeES (Commodity)": "SILVERBEES.NS" 
}

@st.cache_data(ttl=1800)
def load_all_market_data(ticker_list):
    if not ticker_list: return pd.DataFrame()
    return yf.download(ticker_list, period="2y")['Close']

def calculate_and_plot_rrg(df, items_dict, benchmark_ticker, tail_len):
    if not items_dict or df.empty:
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.text(0.5, 0.5, "Select or Add stocks from the sidebar to generate graph", ha='center', va='center', fontsize=12)
        return fig
        
    window_rs = 12  
    window_mom = 12 
    rrg_history = {}
    
    for name, ticker in items_dict.items():
        if ticker in df.columns and not df[ticker].dropna().empty and benchmark_ticker in df.columns:
            df[f'RS_{name}'] = (df[ticker] / df[benchmark_ticker]) * 100
            df[f'RS_Ratio_{name}'] = df[f'RS_{name}'].rolling(window=window_rs).mean()
            
            rolling_mean_ratio = df[f'RS_Ratio_{name}'].rolling(60).mean()
            rolling_std_ratio = df[f'RS_Ratio_{name}'].rolling(60).std().replace(0, np.nan)
            df[f'RS_Ratio_Norm_{name}'] = 100 + ((df[f'RS_Ratio_{name}'] - rolling_mean_ratio) / rolling_std_ratio)
            
            df[f'RS_Mom_{name}'] = df[f'RS_Ratio_Norm_{name}'].pct_change(periods=window_mom) * 100
            rolling_mean_mom = df[f'RS_Mom_{name}'].rolling(60).mean()
            rolling_std_mom = df[f'RS_Mom_{name}'].rolling(60).std().replace(0, np.nan)
            df[f'RS_Mom_Norm_{name}'] = 100 + ((df[f'RS_Mom_{name}'] - rolling_mean_mom) / rolling_std_mom)
            
            clean_df = pd.DataFrame({'x': df[f'RS_Ratio_Norm_{name}'], 'y': df[f'RS_Mom_Norm_{name}']}).dropna()
            if len(clean_df) >= tail_len:
                rrg_history[name] = clean_df.tail(tail_len).to_dict('records')

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axhline(100, color='gray', linestyle='--', linewidth=1)
    ax.axvline(100, color='gray', linestyle='--', linewidth=1)
    
    ax.text(100.3, 103.5, "LEADING", color="green", fontsize=11, fontweight="bold")
    ax.text(100.3, 96.5, "WEAKENING", color="orange", fontsize=11, fontweight="bold")
    ax.text(96.5, 96.5, "LAGGING", color="red", fontsize=11, fontweight="bold")
    ax.text(96.5, 103.5, "IMPROVING", color="blue", fontsize=11, fontweight="bold")
    
    for name, path in rrg_history.items():
        x_history = [pt['x'] for pt in path]
        y_history = [pt['y'] for pt in path]
        current_x, current_y = x_history[-1], y_history[-1]
        
        col = 'green' if current_x >= 100 and current_y >= 100 else 'orange' if current_x >= 100 and current_y < 100 else 'red' if current_x < 100 and current_y < 100 else 'blue'
        
        ax.plot(x_history, y_history, color=col, alpha=0.35, linewidth=2)
        prev_x, prev_y = x_history[-2], y_history[-2]
        ax.annotate('', xy=(current_x, current_y), xytext=(prev_x, prev_y), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=14, zorder=6))
        ax.scatter(current_x, current_y, color=col, s=120, edgecolors='black', zorder=7)
        ax.annotate(name, (current_x, current_y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=8.5)
    
    ax.set_xlim(96, 104)
    ax.set_ylim(96, 104)
    ax.set_xlabel("Relative Strength (RS-Ratio)", fontsize=10)
    ax.set_ylabel("Relative Momentum (RS-Momentum)", fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    return fig

try:
    required_tickers = list(BENCHMARK_MAP.values()) + list(INDEX_SECTORS.values()) + list(STOCK_TICKERS.values())
    master_data = load_all_market_data(required_tickers)
    
    tab1, tab2 = st.tabs(["🏛️ Sectoral Indices Matrix", "🚀 Custom Stocks Canvas"])
    
    with tab1:
        st.subheader(f"Sector & Cap Indices vs {BENCHMARK_LABEL}")
        fig_index = calculate_and_plot_rrg(master_data.copy(), INDEX_SECTORS, BENCHMARK, TAIL_LENGTH)
        st.pyplot(fig_index, use_container_width=True)
        
    with tab2:
        st.subheader(f"Selected Stocks vs {BENCHMARK_LABEL}")
        fig_stock = calculate_and_plot_rrg(master_data.copy(), STOCK_TICKERS, BENCHMARK, TAIL_LENGTH)
        st.pyplot(fig_stock, use_container_width=True)
        
    st.success(f"Analysis successfully loaded.")

except Exception as e:
    st.error(f"Execution Error Check: {e}")
