import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Live NSE RRG Dashboard", layout="wide")
st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")
st.subheader("Live Sectoral Momentum Analysis with Directional Tails")

# 1. Compatible Tickers
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
    data = yf.download(all_tickers, period="2y")['Close']
    return data

try:
    df = load_rrg_data()
    
    window_rs = 12  
    window_mom = 12 
    
    # We will save arrays of historical path paths instead of single values
    rrg_history = {}
    TAIL_LENGTH = 5  # Length of the history trail line (5 trading days)
    
    for name, ticker in SECTORS.items():
        if ticker in df.columns and not df[ticker].dropna().empty:
            df[f'RS_{name}'] = (df[ticker] / df[BENCHMARK]) * 100
            df[f'RS_Ratio_{name}'] = df[f'RS_{name}'].rolling(window=window_rs).mean()
            
            rolling_mean_ratio = df[f'RS_Ratio_{name}'].rolling(60).mean()
            rolling_std_ratio = df[f'RS_Ratio_{name}'].rolling(60).std().replace(0, np.nan)
            df[f'RS_Ratio_Norm_{name}'] = 100 + ((df[f'RS_Ratio_{name}'] - rolling_mean_ratio) / rolling_std_ratio)
            
            df[f'RS_Mom_{name}'] = df[f'RS_Ratio_Norm_{name}'].pct_change(periods=window_mom) * 100
            rolling_mean_mom = df[f'RS_Mom_{name}'].rolling(60).mean()
            rolling_std_mom = df[f'RS_Mom_{name}'].rolling(60).std().replace(0, np.nan)
            df[f'RS_Mom_Norm_{name}'] = 100 + ((df[f'RS_Mom_{name}'] - rolling_mean_mom) / rolling_std_mom)
            
            # Extract the trail window history slice cleanly
            clean_df = pd.DataFrame({
                'x': df[f'RS_Ratio_Norm_{name}'],
                'y': df[f'RS_Mom_Norm_{name}']
            }).dropna()
            
            if len(clean_df) >= TAIL_LENGTH:
                # Capture the last 5 coordinates to construct the tail
                rrg_history[name] = clean_df.tail(TAIL_LENGTH).to_dict('records')

    # 2. Rendering Graph Configuration
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)
    
    ax.text(100.2, 102.5, "LEADING", color="green", fontsize=9, fontweight="bold")
    ax.text(100.2, 97.5, "WEAKENING", color="orange", fontsize=9, fontweight="bold")
    ax.text(97.2, 97.5, "LAGGING", color="red", fontsize=9, fontweight="bold")
    ax.text(97.2, 102.5, "IMPROVING", color="blue", fontsize=9, fontweight="bold")
    
    # 3. Dynamic Loop to draw paths and final markers
    for name, path in rrg_history.items():
        x_history = [pt['x'] for pt in path]
        y_history = [pt['y'] for pt in path]
        
        # Determine base quadrant color utilizing target point location metrics
        current_x, current_y = x_history[-1], y_history[-1]
        if current_x >= 100 and current_y >= 100: col = 'green'
        elif current_x >= 100 and current_y < 100: col = 'orange'
        elif current_x < 100 and current_y < 100: col = 'red'
        else: col = 'blue'
        
        # Step A: Draw the fading trail path leading backward
        ax.plot(x_history, y_history, color=col, alpha=0.4, linewidth=1.5, linestyle='-')
        
        # Step B: Draw directional vector arrow pointing directly to the final tracking coordinate
        prev_x, prev_y = x_history[-2], y_history[-2]
        dx = current_x - prev_x
        dy = current_y - prev_y
        
        ax.annotate('', xy=(current_x, current_y), xytext=(prev_x, prev_y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=12, zorder=6))
        
        # Step C: Plot the main header label marker bubble
        ax.scatter(current_x, current_y, color=col, s=80, edgecolors='black', zorder=7)
        ax.annotate(name, (current_x, current_y), textcoords="offset points", xytext=(0,8), ha='center', fontweight='bold', fontsize=7.5)
    
    ax.set_xlim(96, 104)
    ax.set_ylim(96, 104)
    ax.set_xlabel("Relative Strength (RS-Ratio)", fontsize=9)
    ax.set_ylabel("Relative Momentum (RS-Momentum)", fontsize=9)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig)
        
    st.success("Tails updated live! The line trace path displays movement across the last 5 trading sessions.")

except Exception as e:
    st.error(f"Data Visualizer Exception Check: {e}")
