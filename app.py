import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Live NSE RRG Dashboard", layout="wide")
st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")
st.subheader("Live Sectoral Momentum Analysis")

# 1. FIXED TICKERS: Swapped to fully compatible Yahoo Finance codes
BENCHMARK = "^NSEI"  # Nifty 50
SECTORS = {
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Infra": "^CNXINFRA",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Energy": "^CNXENERGY"
}

@st.cache_data(ttl=3600)
def load_rrg_data():
    all_tickers = [BENCHMARK] + list(SECTORS.values())
    # Extended timeline to 2y to guarantee ample rolling mathematical window history
    data = yf.download(all_tickers, period="2y")['Close']
    return data

try:
    df = load_rrg_data()
    
    window_rs = 12  
    window_mom = 12 
    
    rrg_results = {}
    
    for name, ticker in SECTORS.items():
        if ticker in df.columns and not df[ticker].dropna().empty:
            # Mathematical RRG calculations 
            df[f'RS_{name}'] = (df[ticker] / df[BENCHMARK]) * 100
            df[f'RS_Ratio_{name}'] = df[f'RS_{name}'].rolling(window=window_rs).mean()
            
            # Safe padding logic for statistical standard deviation metrics
            rolling_mean_ratio = df[f'RS_Ratio_{name}'].rolling(60).mean()
            rolling_std_ratio = df[f'RS_Ratio_{name}'].rolling(60).std().replace(0, np.nan)
            
            df[f'RS_Ratio_Norm_{name}'] = 100 + ((df[f'RS_Ratio_{name}'] - rolling_mean_ratio) / rolling_std_ratio)
            df[f'RS_Mom_{name}'] = df[f'RS_Ratio_Norm_{name}'].pct_change(periods=window_mom) * 100
            
            rolling_mean_mom = df[f'RS_Mom_{name}'].rolling(60).mean()
            rolling_std_mom = df[f'RS_Mom_{name}'].rolling(60).std().replace(0, np.nan)
            
            df[f'RS_Mom_Norm_{name}'] = 100 + ((df[f'RS_Mom_{name}'] - rolling_mean_mom) / rolling_std_mom)
            
            # Clean data validation before plotting points
            final_x = df[f'RS_Ratio_Norm_{name}'].dropna().iloc[-1]
            final_y = df[f'RS_Mom_Norm_{name}'].dropna().iloc[-1]
            
            rrg_results[name] = {'x': final_x, 'y': final_y}

    # 2. REDUCED SIZE: Changed figsize from (10, 8) down to a compact (7, 5)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)
    
    # Position quadrant background text labels comfortably 
    ax.text(100.2, 102, "LEADING", color="green", fontsize=9, fontweight="bold")
    ax.text(100.2, 97.5, "WEAKENING", color="orange", fontsize=9, fontweight="bold")
    ax.text(97.5, 97.5, "LAGGING", color="red", fontsize=9, fontweight="bold")
    ax.text(97.5, 102, "IMPROVING", color="blue", fontsize=9, fontweight="bold")
    
    for name, coords in rrg_results.items():
        x, y = coords['x'], coords['y']
        
        if x >= 100 and y >= 100: col = 'green'
        elif x >= 100 and y < 100: col = 'orange'
        elif x < 100 and y < 100: col = 'red'
        else: col = 'blue'
        
        # Reduced bubble marker sizing down to 100 for a cleaner spatial footprint
        ax.scatter(x, y, color=col, s=100, edgecolors='black', zorder=5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontweight='bold', fontsize=8)
    
    ax.set_xlim(96, 104)
    ax.set_ylim(96, 104)
    ax.set_xlabel("Relative Strength (RS-Ratio)", fontsize=9)
    ax.set_ylabel("Relative Momentum (RS-Momentum)", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    
    # Display the refined chart cleanly in Streamlit layout columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)
        
    st.success("Data safely processed across sectors! Refresh browser tab to pull latest updates.")

except Exception as e:
    st.error(f"Live Feed Processing Notice: {e}. Check asset connectivity options.")
