import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration for Widescreen Layout
st.set_page_config(page_title="Live NSE RRG Dashboard", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Configuration Control Sidebar Panel
st.sidebar.title("🔧 Settings Panel")

BENCHMARK_LABEL = st.sidebar.selectbox(
    "Select Benchmark Index",
    ["Nifty 50", "Nifty 100", "Nifty 200"]
)

BENCHMARK_MAP = {
    "Nifty 50": "^NSEI",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200"
}
BENCHMARK = BENCHMARK_MAP[BENCHMARK_LABEL]

TAIL_LENGTH = st.sidebar.slider("Counts (Tail Days Path)", min_value=3, max_value=15, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 Add Custom Stock")
custom_ticker_input = st.sidebar.text_input("Type NSE Ticker (e.g., ADANIENT):", "").strip().upper()

st.title("🇮🇳 Indian Market Relative Rotation Graph (RRG)")

# Asset Lists
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
    "Nifty Defence (ETF)": "GROWWDEFNC.NS",  
    "Nifty Real Estate": "^CNXREALTY",      
    "Gold BeES (Commodity)": "GOLDBEES.NS",  
    "Silver BeES (Commodity)": "SILVERBEES.NS" 
}

# Base Blue-Chip Stocks List
STOCK_TICKERS = {
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "L&T": "LT.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "Tata Motors": "TATAMOTORS.NS"
}

if custom_ticker_input:
    formatted_ticker = custom_ticker_input if custom_ticker_input.endswith(".NS") else f"{custom_ticker_input}.NS"
    STOCK_TICKERS[custom_ticker_input] = formatted_ticker

@st.cache_data(ttl=1800)
def load_all_market_data(ticker_list):
    try:
        data = yf.download(ticker_list, period="2y")['Close']
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.DataFrame()

def calculate_and_plot_plotly_rrg(df, items_dict, benchmark_ticker, tail_len):
    window_rs = 12  
    window_mom = 12 
    
    # Initialize interactive figure layout
    fig = go.Figure()
    
    # Draw Background Quadrant Shading & Divider Lines
    fig.add_hline(y=100, line_dash="dash", line_color="gray", line_width=1)
    fig.add_vline(x=100, line_dash="dash", line_color="gray", line_width=1)

    for name, ticker in items_dict.items():
        if ticker in df.columns and not df[ticker].dropna().empty and benchmark_ticker in df.columns:
            valid_series = df[ticker].dropna()
            available_history_count = len(valid_series)
            current_rolling_window = min(60, max(15, available_history_count // 3))
            
            aligned_benchmark = df[benchmark_ticker].loc[valid_series.index]
            rs_series = (valid_series / aligned_benchmark) * 100
            rs_ratio = rs_series.rolling(window=window_rs).mean()
            
            rolling_mean_ratio = rs_ratio.rolling(current_rolling_window).mean()
            rolling_std_ratio = rs_ratio.rolling(current_rolling_window).std().replace(0, np.nan)
            rs_ratio_norm = 100 + ((rs_ratio - rolling_mean_ratio) / rolling_std_ratio)
            
            rs_mom = rs_ratio_norm.pct_change(periods=window_mom) * 100
            rolling_mean_mom = rs_mom.rolling(current_rolling_window).mean()
            rolling_std_mom = rs_mom.rolling(current_rolling_window).std().replace(0, np.nan)
            rs_mom_norm = 100 + ((rs_mom - rolling_mean_mom) / rolling_std_mom)
            
            clean_df = pd.DataFrame({'x': rs_ratio_norm, 'y': rs_mom_norm}).dropna()
            
            if len(clean_df) >= tail_len:
                tail_data = clean_df.tail(tail_len)
                x_history = tail_data['x'].tolist()
                y_history = tail_data['y'].tolist()
                
                current_x, current_y = x_history[-1], y_history[-1]
                
                # Dynamic Quadrant Color Selection Engine
                if current_x >= 100 and current_y >= 100: col = 'green'
                elif current_x >= 100 and current_y < 100: col = 'orange'
                elif current_x < 100 and current_y < 100: col = 'red'
                else: col = 'blue'
                
                # Plotly Trace 1: Draw the history trailing path line
                fig.add_trace(go.Scatter(
                    x=x_history, y=y_history,
                    mode='lines',
                    line=dict(color=col, width=2),
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Plotly Trace 2: Draw the interactive header point bubble
                fig.add_trace(go.Scatter(
                    x=[current_x], y=[current_y],
                    mode='markers+text',
                    name=name,
                    marker=dict(color=col, size=12, line=dict(color='black', width=1)),
                    text=[name],
                    textposition="top center",
                    textfont=dict(size=10, weight="bold"),
                    hovertemplate=f"<b>{name}</b><br>RS-Ratio: %{{x|..2f}}<br>RS-Momentum: %{{y|..2f}}<extra></extra>"
                ))

    # Configure clean axes limits and permanent static text quadrant annotations
    fig.update_layout(
        xaxis=dict(title="Relative Strength (RS-Ratio)", range=[96, 104]),
        yaxis=dict(title="Relative Momentum (RS-Momentum)", range=[96, 104]),
        margin=dict(l=20, r=20, t=20, b=20),
        height=680,
        plot_bgcolor='white',
        hovermode='closest',
        annotations=[
            dict(x=102, y=103, text="LEADING", showarrow=False, font=dict(color="green", size=14, weight="bold")),
            dict(x=102, y=97, text="WEAKENING", showarrow=False, font=dict(color="orange", size=14, weight="bold")),
            dict(x=98, y=97, text="LAGGING", showarrow=False, font=dict(color="red", size=14, weight="bold")),
            dict(x=98, y=103, text="IMPROVING", showarrow=False, font=dict(color="blue", size=14, weight="bold"))
        ]
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig

try:
    required_tickers = list(BENCHMARK_MAP.values()) + list(INDEX_SECTORS.values()) + list(STOCK_TICKERS.values())
    master_data = load_all_market_data(required_tickers)
    
    tab1, tab2 = st.tabs(["🏛️ Sectoral Indices Matrix", "🚀 Blue-Chip Individual Stocks"])
    
    with tab1:
        st.subheader(f"Sector & Commodity Indices vs {BENCHMARK_LABEL}")
        plotly_index_fig = calculate_and_plot_plotly_rrg(master_data.copy(), INDEX_SECTORS, BENCHMARK, TAIL_LENGTH)
        st.plotly_chart(plotly_index_fig, use_container_width=True)
        
    with tab2:
        st.subheader(f"Top Heavyweight & Custom Stocks vs {BENCHMARK_LABEL}")
        plotly_stock_fig = calculate_and_plot_plotly_rrg(master_data.copy(), STOCK_TICKERS, BENCHMARK, TAIL_LENGTH)
        st.plotly_chart(plotly_stock_fig, use_container_width=True)
        
    st.success("Widescreen interactive canvas active! Hover over clustered items to separate overlapping sectors.")

except Exception as e:
    st.error(f"Execution Error Check: {e}")
