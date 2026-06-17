import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Force the page to use full horizontal workspace dimensions
st.set_page_config(page_title="Live NSE RRG Dashboard", layout="wide")

# Inject structural CSS modifications to strip whitespace padding channels
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        iframe {
            width: 100% !important;
        }
    </style>
""", unsafe_style_codes=True)

# 2. Interactive Sidebar Panel Setup (Matches Strike Layout Structure)
st.sidebar.title("🔧 Configuration")
TAIL_LENGTH = st.sidebar.slider("Counts (Tail Days Path)", min_value=3, max_value=15, value=5, step=1)
st.sidebar.caption("Data Anchor Base: Nifty 50 Index")

st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")

BENCHMARK = "^NSEI"
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
    rrg_history = {}
    
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
            
            clean_df = pd.DataFrame({
                'x': df[f'RS_Ratio_Norm_{name}'],
                'y': df[f'RS_Mom_Norm_{name}']
            }).dropna()
            
            if len(clean_df) >= TAIL_LENGTH:
                rrg_history[name] = clean_df.tail(TAIL_LENGTH).to_dict('records')

    # 3. Dynamic Screen Fitting Graph Size Configuration
    # Adjusting aspect ratio to utilize widescreen canvas coordinates (13 width x 7.5 height)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.axhline(100, color='gray', linestyle='--', linewidth=1)
    ax.axvline(100, color='gray', linestyle='--', linewidth=1)
    
    # Quadrant Identifier Overlay Placements
    ax.text(100.3, 103.5, "LEADING", color="green", fontsize=11, fontweight="bold")
    ax.text(100.3, 96.5, "WEAKENING", color="orange", fontsize=11, fontweight="bold")
    ax.text(96.5, 96.5, "LAGGING", color="red", fontsize=11, fontweight="bold")
    ax.text(96.5, 103.5, "IMPROVING", color="blue", fontsize=11, fontweight="bold")
    
    for name, path in rrg_history.items():
        x_history = [pt['x'] for pt in path]
        y_history = [pt['y'] for pt in path]
        
        current_x, current_y = x_history[-1], y_history[-1]
        if current_x >= 100 and current_y >= 100: col = 'green'
        elif current_x >= 100 and current_y < 100: col = 'orange'
        elif current_x < 100 and current_y < 100: col = 'red'
        else: col = 'blue'
        
        # Plot structural path connecting tail matrix records
        ax.plot(x_history, y_history, color=col, alpha=0.35, linewidth=2, linestyle='-')
        
        # Build vector arrow indicators
        prev_x, prev_y = x_history[-2], y_history[-2]
        ax.annotate('', xy=(current_x, current_y), xytext=(prev_x, prev_y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, mutation_scale=14, zorder=6))
        
        ax.scatter(current_x, current_y, color=col, s=120, edgecolors='black', zorder=7)
        ax.annotate(name, (current_x, current_y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', fontsize=8.5)
    
    ax.set_xlim(96, 104)
    ax.set_ylim(96, 104)
    ax.set_xlabel("Relative Strength (RS-Ratio)", fontsize=10)
    ax.set_ylabel("Relative Momentum (RS-Momentum)", fontsize=10)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    
    # 4. Render direct layout execution across the wide canvas area
    st.pyplot(fig, use_container_width=True)
    st.caption("⚡ Live analysis active. Modify the parameter controls in the left sidebar to tweak real-time tail sizes.")

except Exception as e:
    st.error(f"Layout Build Exception Track: {e}")
