import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Live NSE RRG Dashboard", layout="wide")
st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")
st.subheader("Live Sectoral Momentum Analysis (Refreshes on Page Load)")

# 1. Define Benchmark (Nifty 50) and Key Sector Tickers
BENCHMARK = "^NSEI"
SECTORS = {
    "Nifty Bank": "NIFTY_BANK.NS",
    "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Metal": "^CNXMETAL"
}

@st.cache_data(ttl=3600)  # Caches data for 1 hour to stay fast, but refreshes daily/live
def load_rrg_data():
    all_tickers = [BENCHMARK] + list(SECTORS.values())
    # Download past 1 year of daily historical data to calculate true momentum
    data = yf.download(all_tickers, period="1y")['Close']
    return data

try:
    df = load_rrg_data()
    
    # 2. Mathematical RRG Calculations (RS-Ratio and RS-Momentum)
    window_rs = 12  # Standard RS tracking period
    window_mom = 12 # Standard Momentum tracking period
    
    rrg_results = {}
    
    for name, ticker in SECTORS.items():
        # Relative Strength (Sector Price / Benchmark Price) * 100
        df[f'RS_{name}'] = (df[ticker] / df[BENCHMARK]) * 100
        
        # RS-Ratio: 12-period simple moving average of RS normalized
        df[f'RS_Ratio_{name}'] = df[f'RS_{name}'].rolling(window=window_rs).mean()
        # Simple Z-score normalization around baseline 100
        df[f'RS_Ratio_Norm_{name}'] = 100 + ((df[f'RS_Ratio_{name}'] - df[f'RS_Ratio_{name}'].rolling(20).mean()) / df[f'RS_Ratio_{name}'].rolling(20).std())
        
        # RS-Momentum: Rate of change of RS-Ratio
        df[f'RS_Mom_{name}'] = df[f'RS_Ratio_Norm_{name}'].pct_change(periods=window_mom) * 100
        df[f'RS_Mom_Norm_{name}'] = 100 + ((df[f'RS_Mom_{name}'] - df[f'RS_Mom_{name}'].rolling(20).mean()) / df[f'RS_Mom_{name}'].rolling(20).std())
        
        # Extract the most recent live data coordinate points
        rrg_results[name] = {
            'x': df[f'RS_Ratio_Norm_{name}'].iloc[-1],
            'y': df[f'RS_Mom_Norm_{name}'].iloc[-1]
        }

    # 3. Plotting the Live Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(100, color='gray', linestyle='--', linewidth=1)
    ax.axvline(100, color='gray', linestyle='--', linewidth=1)
    
    # Add Quadrant Background Color Labels
    ax.text(100.5, 103, "LEADING", color="green", fontsize=12, fontweight="bold")
    ax.text(100.5, 96.5, "WEAKENING", color="orange", fontsize=12, fontweight="bold")
    ax.text(96.5, 96.5, "LAGGING", color="red", fontsize=12, fontweight="bold")
    ax.text(96.5, 103, "IMPROVING", color="blue", fontsize=12, fontweight="bold")
    
    # Scatter plot each sector's current live coordinate
    for name, coords in rrg_results.items():
        x, y = coords['x'], coords['y']
        
        # Determine Color Dynamic based on Position
        if x >= 100 and y >= 100: col = 'green'
        elif x >= 100 and y < 100: col = 'orange'
        elif x < 100 and y < 100: col = 'red'
        else: col = 'blue'
        
        ax.scatter(x, y, color=col, s=200, edgecolors='black', zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold')
    
    ax.set_xlim(95, 105)
    ax.set_ylim(95, 105)
    ax.set_xlabel("Relative Strength (RS-Ratio)")
    ax.set_ylabel("Relative Momentum (RS-Momentum)")
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    st.pyplot(fig)
    st.success("Data successfully fetched live from NSE. Refresh your browser tab to update anytime!")

except Exception as e:
    st.error(f"Live Feed Error: {e}. The market session might be closed or API limits reached.")
