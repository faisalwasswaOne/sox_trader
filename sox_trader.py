import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SoX Trader - Weekly Trading Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SoX Trader branding
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    .signal-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
INDICES = {
    "FTSE 100": "^FTSE",
    "NASDAQ Composite": "^IXIC",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI"
}

# Static fallback list in case API fails
FALLBACK_FTSE100_COMPANIES = {
    "AstraZeneca": "AZN.L",
    "HSBC Holdings": "HSBA.L",
    "Shell": "SHEL.L",
    "Unilever": "ULVR.L",
    "Rolls-Royce Group": "RR.L"
}

# FTSE 100 constituent tickers for dynamic top 5 selection
FTSE100_CONSTITUENTS = {
    "AstraZeneca": "AZN.L",
    "HSBC Holdings": "HSBA.L",
    "Shell": "SHEL.L",
    "Unilever": "ULVR.L",
    "Rolls-Royce Group": "RR.L",
    "British American Tobacco": "BATS.L",
    "Diageo": "DGE.L",
    "Vodafone": "VOD.L",
    "BP": "BP.L",
    "Lloyds Banking Group": "LLOY.L",
    "Barclays": "BARC.L",
    "Tesco": "TSCO.L",
    "BT Group": "BT-A.L",
    "Rio Tinto": "RIO.L",
    "Aviva": "AV.L",
    "Legal & General": "LGEN.L",
    "National Grid": "NG.L",
    "Sainsbury (J)": "SBRY.L",
    "Marks & Spencer": "MKS.L",
    "Centrica": "CNA.L",
    "Compass Group": "CPG.L",
    "GSK": "GSK.L",
    "Prudential": "PRU.L",
    "Standard Chartered": "STAN.L",
    "Kingfisher": "KGF.L",
    "Next": "NXT.L",
    "Persimmon": "PSN.L",
    "Taylor Wimpey": "TW.L",
    "Barratt Developments": "BDEV.L",
    "Berkeley Group": "BKG.L"
}

# Helper Functions
@st.cache_data(ttl=604800)  # Cache for 1 week (604800 seconds)
def get_top5_ftse100_companies():
    """Dynamically fetch top 5 FTSE 100 companies by market cap."""
    try:
        company_data = []
        
        # Show update message only during initial load
        with st.spinner("🔄 Updating top 5 FTSE 100 companies based on current market cap..."):
            for i, (name, ticker) in enumerate(FTSE100_CONSTITUENTS.items()):
                try:
                    # Get company info including market cap
                    company = yf.Ticker(ticker)
                    info = company.info
                    
                    if 'marketCap' in info and info['marketCap']:
                        market_cap = info['marketCap']
                        company_data.append({
                            'name': name,
                            'ticker': ticker,
                            'market_cap': market_cap
                        })
                    
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")
                    continue
        
        # Sort by market cap and get top 5
        if company_data:
            company_data.sort(key=lambda x: x['market_cap'], reverse=True)
            top_5 = company_data[:5]
            
            top_5_dict = {company['name']: company['ticker'] for company in top_5}
            
            # Store the update time and data for display
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create a data structure that includes both the companies and metadata
            result = {
                'companies': top_5_dict,
                'market_caps': {company['name']: company['market_cap'] for company in top_5},
                'updated_at': current_time
            }
            
            return result
        else:
            st.warning("⚠️ Could not fetch market cap data. Using fallback list.")
            return {
                'companies': FALLBACK_FTSE100_COMPANIES,
                'market_caps': {},
                'updated_at': 'Fallback data'
            }
            
    except Exception as e:
        st.error(f"Error updating top 5 companies: {e}")
        st.info("Using fallback list of top 5 FTSE 100 companies.")
        return {
            'companies': FALLBACK_FTSE100_COMPANIES,
            'market_caps': {},
            'updated_at': 'Error - using fallback'
        }

# Get dynamic top 5 companies
TOP_FTSE100_DATA = get_top5_ftse100_companies()
TOP_FTSE100_COMPANIES = TOP_FTSE100_DATA['companies']

ALL_TICKERS = {**INDICES, **TOP_FTSE100_COMPANIES}
@st.cache_data(ttl=3600)
def get_data(ticker, period="1y"):
    """Fetches historical data for a given ticker with error handling."""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
            return pd.DataFrame()
        
        # Handle MultiIndex columns (when downloading single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)  # 15 minute cache for current prices
def get_current_price(ticker):
    """Gets current price data for a ticker."""
    try:
        ticker_obj = yf.Ticker(ticker)
        current_data = ticker_obj.history(period="1d", interval="5m")
        if not current_data.empty:
            # Handle MultiIndex columns if present
            if isinstance(current_data.columns, pd.MultiIndex):
                current_data.columns = current_data.columns.droplevel(1)
            return float(current_data['Close'].iloc[-1])
        return None
    except Exception as e:
        st.error(f"Error fetching current price for {ticker}: {e}")
        return None

def calculate_weekly_indicators(df):
    """Calculates weekly technical indicators optimized for 7-day trading cycles."""
    if df.empty:
        return df
    
    # Debug: Print column names
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.error(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    # Resample to weekly data
    weekly_df = df.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # Weekly Moving Averages
    weekly_df['SMA_20'] = ta.trend.sma_indicator(weekly_df['Close'], window=20)
    weekly_df['SMA_50'] = ta.trend.sma_indicator(weekly_df['Close'], window=50)
    weekly_df['EMA_12'] = ta.trend.ema_indicator(weekly_df['Close'], window=12)
    weekly_df['EMA_26'] = ta.trend.ema_indicator(weekly_df['Close'], window=26)
    
    # Weekly RSI
    weekly_df['RSI'] = ta.momentum.rsi(weekly_df['Close'], window=14)
    
    # Weekly MACD
    weekly_df['MACD'] = ta.trend.macd(weekly_df['Close'])
    weekly_df['MACD_Signal'] = ta.trend.macd_signal(weekly_df['Close'])
    weekly_df['MACD_Hist'] = ta.trend.macd_diff(weekly_df['Close'])
    
    # Weekly Bollinger Bands
    weekly_df['BB_upper'] = ta.volatility.bollinger_hband(weekly_df['Close'])
    weekly_df['BB_middle'] = ta.volatility.bollinger_mavg(weekly_df['Close'])
    weekly_df['BB_lower'] = ta.volatility.bollinger_lband(weekly_df['Close'])
    
    # Weekly Average True Range (ATR) for volatility
    weekly_df['ATR'] = ta.volatility.average_true_range(
        weekly_df['High'], weekly_df['Low'], weekly_df['Close'], window=14
    )
    
    return weekly_df

def get_individual_signals(df):
    """Returns individual signal components for UI display."""
    if df.empty or len(df) < 50:
        return {
            'forecast': 'NEUTRAL',
            'ma': 'NEUTRAL', 
            'rsi': 'NEUTRAL',
            'macd': 'NEUTRAL',
            'bb': 'NEUTRAL'
        }
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = last_row['Close']
    
    # Get 7-day forecast for trend analysis
    forecast_price, upper_range, lower_range = calculate_7day_forecast(df)
    
    # Forecast trend signal
    forecast_signal = "NEUTRAL"
    if forecast_price is not None:
        price_change_percent = ((forecast_price - current_price) / current_price) * 100
        if price_change_percent > 1:
            forecast_signal = "BULLISH"
        elif price_change_percent < -0.5:
            forecast_signal = "BEARISH"
        else:
            forecast_signal = "NEUTRAL"
    
    # Moving Average Signals
    ma_signal = "NEUTRAL"
    if not pd.isna(last_row['SMA_20']) and not pd.isna(last_row['SMA_50']):
        if last_row['SMA_20'] > last_row['SMA_50']:
            ma_signal = "BULLISH"
        elif last_row['SMA_20'] < last_row['SMA_50']:
            ma_signal = "BEARISH"
    
    # RSI Signals
    rsi_signal = "NEUTRAL"
    if not pd.isna(last_row['RSI']):
        if last_row['RSI'] < 30:
            rsi_signal = "BULLISH"
        elif last_row['RSI'] > 70:
            rsi_signal = "BEARISH"
    
    # MACD Signals
    macd_signal = "NEUTRAL"
    if not pd.isna(last_row['MACD']) and not pd.isna(last_row['MACD_Signal']):
        if last_row['MACD'] > last_row['MACD_Signal']:
            macd_signal = "BULLISH"
        elif last_row['MACD'] < last_row['MACD_Signal']:
            macd_signal = "BEARISH"
    
    # Bollinger Bands Signals
    bb_signal = "NEUTRAL"
    if not pd.isna(last_row['BB_upper']) and not pd.isna(last_row['BB_lower']):
        if last_row['Close'] > last_row['BB_upper']:
            bb_signal = "BEARISH"
        elif last_row['Close'] < last_row['BB_lower']:
            bb_signal = "BULLISH"
    
    return {
        'forecast': forecast_signal,
        'ma': ma_signal,
        'rsi': rsi_signal,
        'macd': macd_signal,
        'bb': bb_signal
    }

def generate_weekly_signals(df):
    """Generates weekly trading signals with confidence scoring."""
    if df.empty or len(df) < 50:
        return "HOLD", "LOW", ["Not enough historical data for reliable weekly signals."]
    
    signals = []
    confidence_factors = []
    
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    current_price = last_row['Close']
    
    # Get 7-day forecast for trend analysis
    forecast_price, upper_range, lower_range = calculate_7day_forecast(df)
    
    # Forecast trend signal
    forecast_signal = "NEUTRAL"
    if forecast_price is not None:
        price_change_percent = ((forecast_price - current_price) / current_price) * 100
        
        # Debug info
        print(f"DEBUG - Current: {current_price:.2f}, Forecast: {forecast_price:.2f}, Change: {price_change_percent:.1f}%")
        
        if price_change_percent > 1:  # Forecast shows >1% increase
            forecast_signal = "BULLISH"
            signals.append(f"📈 Forecast Trend: +{price_change_percent:.1f}% expected gain")
            confidence_factors.append(0.9)
        elif price_change_percent < -0.5:  # Forecast shows >0.5% decrease
            forecast_signal = "BEARISH"
            signals.append(f"📉 Forecast Trend: {price_change_percent:.1f}% expected decline")
            confidence_factors.append(0.8)
        else:
            forecast_signal = "NEUTRAL"
            signals.append(f"📊 Forecast Trend: {price_change_percent:+.1f}% (neutral)")
            confidence_factors.append(0.3)
    else:
        signals.append("⚠️ Forecast unavailable - using indicators only")
        confidence_factors.append(0.1)
    
    # Moving Average Signals
    ma_signal = "NEUTRAL"
    if not pd.isna(last_row['SMA_20']) and not pd.isna(last_row['SMA_50']):
        if last_row['SMA_20'] > last_row['SMA_50']:
            ma_signal = "BULLISH"
            if prev_row['SMA_20'] <= prev_row['SMA_50']:
                signals.append("📈 Golden Cross: 20-week SMA crossed above 50-week SMA")
                confidence_factors.append(1)
        elif last_row['SMA_20'] < last_row['SMA_50']:
            ma_signal = "BEARISH"
            if prev_row['SMA_20'] >= prev_row['SMA_50']:
                signals.append("📉 Death Cross: 20-week SMA crossed below 50-week SMA")
                confidence_factors.append(1)
    
    # RSI Signals
    rsi_signal = "NEUTRAL"
    if not pd.isna(last_row['RSI']):
        if last_row['RSI'] < 30:
            rsi_signal = "BULLISH"
            signals.append(f"📊 RSI Oversold: {last_row['RSI']:.1f} (< 30)")
            confidence_factors.append(0.8)
        elif last_row['RSI'] > 70:
            rsi_signal = "BEARISH"
            signals.append(f"📊 RSI Overbought: {last_row['RSI']:.1f} (> 70)")
            confidence_factors.append(0.8)
    
    # MACD Signals
    macd_signal = "NEUTRAL"
    if not pd.isna(last_row['MACD']) and not pd.isna(last_row['MACD_Signal']):
        if last_row['MACD'] > last_row['MACD_Signal']:
            macd_signal = "BULLISH"
            if prev_row['MACD'] <= prev_row['MACD_Signal']:
                signals.append("➕ MACD Bullish Crossover")
                confidence_factors.append(0.9)
        elif last_row['MACD'] < last_row['MACD_Signal']:
            macd_signal = "BEARISH"
            if prev_row['MACD'] >= prev_row['MACD_Signal']:
                signals.append("➖ MACD Bearish Crossover")
                confidence_factors.append(0.9)
    
    # Price vs Bollinger Bands
    bb_signal = "NEUTRAL"
    if not pd.isna(last_row['BB_upper']) and not pd.isna(last_row['BB_lower']):
        if last_row['Close'] > last_row['BB_upper']:
            bb_signal = "BEARISH"
            signals.append("🔴 Price above Upper Bollinger Band (potential reversal)")
            confidence_factors.append(0.6)
        elif last_row['Close'] < last_row['BB_lower']:
            bb_signal = "BULLISH"
            signals.append("🟢 Price below Lower Bollinger Band (potential bounce)")
            confidence_factors.append(0.6)
    
    # Aggregate signals (including forecast trend)
    all_signals = [forecast_signal, ma_signal, rsi_signal, macd_signal, bb_signal]
    bullish_signals = all_signals.count("BULLISH")
    bearish_signals = all_signals.count("BEARISH")
    
    # Debug signal counts
    print(f"DEBUG - Forecast: {forecast_signal}, MA: {ma_signal}, RSI: {rsi_signal}, MACD: {macd_signal}, BB: {bb_signal}")
    print(f"DEBUG - Bullish: {bullish_signals}, Bearish: {bearish_signals}")
    
    # Additional check: if forecast shows ANY decline, be more conservative
    forecast_decline = forecast_price is not None and forecast_price < current_price
    
    # Determine final signal with forecast trend weighted heavily
    if forecast_signal == "BULLISH" and bullish_signals >= bearish_signals:
        final_signal = "BUY"
        print(f"DEBUG - Final signal: BUY (forecast bullish + indicators support)")
    elif forecast_signal == "BEARISH":
        final_signal = "SELL"
        print(f"DEBUG - Final signal: SELL (forecast bearish)")
    elif forecast_decline and bullish_signals > bearish_signals:
        # Even small forecast decline should make us cautious about BUY
        final_signal = "HOLD"
        signals.append("⚠️ Conservative approach: Forecast shows price decline - HOLD recommended despite bullish indicators")
        print(f"DEBUG - Final signal: HOLD (forecast decline overrides bullish indicators)")
    elif forecast_signal == "BULLISH" and bearish_signals > bullish_signals:
        # Forecast is bullish but indicators are bearish - be cautious
        final_signal = "HOLD"
        signals.append("⚠️ Mixed signals: Forecast bullish but indicators bearish - HOLD recommended")
        print(f"DEBUG - Final signal: HOLD (forecast bullish but indicators bearish)")
    elif bullish_signals > bearish_signals:
        final_signal = "BUY"
        print(f"DEBUG - Final signal: BUY (indicators only - no forecast conflict)")
    elif bearish_signals > bullish_signals:
        final_signal = "SELL"
        print(f"DEBUG - Final signal: SELL (indicators only - no forecast)")
    else:
        final_signal = "HOLD"
        print(f"DEBUG - Final signal: HOLD (neutral)")
    
    # Calculate confidence
    if confidence_factors:
        avg_confidence = np.mean(confidence_factors)
        signal_strength = abs(bullish_signals - bearish_signals) / 4
        final_confidence = (avg_confidence + signal_strength) / 2
        
        if final_confidence > 0.7:
            confidence = "HIGH"
        elif final_confidence > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        confidence = "LOW"
    
    if not signals:
        signals.append("No strong signals detected this week.")
    
    return final_signal, confidence, signals

def calculate_7day_forecast(df):
    """Calculate 7-day price forecast using trend analysis."""
    if df.empty or len(df) < 20:
        return None, None, None
    
    # Use last 20 weeks for trend calculation
    recent_data = df.tail(20)
    
    # Calculate trend using linear regression
    x = np.arange(len(recent_data))
    y = recent_data['Close'].values
    
    # Simple linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Project 1 week ahead (next point)
    forecast_price = slope * len(recent_data) + intercept
    
    # Calculate volatility-based range
    volatility = recent_data['Close'].std()
    upper_range = forecast_price + (volatility * 1.5)
    lower_range = forecast_price - (volatility * 1.5)
    
    return forecast_price, upper_range, lower_range

# Streamlit UI
st.markdown('<div class="main-header">SoX Trader</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Weekly Trading Signal System</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page_selection = st.sidebar.radio("Select Page", ["Market Overview", "Weekly Analysis", "About"])

if page_selection == "Market Overview":
    st.header("🌍 Market Overview")
    st.write("Current status of major indices and top FTSE 100 companies")
    
    # Add refresh button for top 5 companies
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("*Top 5 companies update weekly automatically*")
    with col2:
        if st.button("🔄 Refresh Top 5"):
            st.cache_data.clear()
            st.rerun()
    
    # Display Indices
    st.subheader("Global Indices")
    index_cols = st.columns(len(INDICES))
    
    for i, (name, ticker) in enumerate(INDICES.items()):
        with index_cols[i]:
            current_price = get_current_price(ticker)
            if current_price:
                # Get yesterday's close for comparison
                hist_data = get_data(ticker, period="2d")
                if not hist_data.empty and len(hist_data) > 1:
                    yesterday_close = float(hist_data['Close'].iloc[-2])
                    change = float(current_price - yesterday_close)
                    change_percent = float((change / yesterday_close) * 100)
                    
                    st.metric(
                        label=name,
                        value=f"{current_price:,.2f}",
                        delta=f"{change:,.2f} ({change_percent:+.2f}%)"
                    )
                else:
                    st.metric(label=name, value=f"{current_price:,.2f}", delta="N/A")
            else:
                st.metric(label=name, value="N/A", delta="N/A")
    
    st.subheader("Top 5 FTSE 100 Companies")
    
    # Show update information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"📅 Last updated: {TOP_FTSE100_DATA['updated_at']}")
    with col2:
        st.caption("🔄 Auto-updates weekly")
    
    # Display current top 5 with market cap
    if TOP_FTSE100_DATA['market_caps']:
        st.write("**Current ranking by market capitalization:**")
        for i, (name, ticker) in enumerate(TOP_FTSE100_COMPANIES.items(), 1):
            market_cap = TOP_FTSE100_DATA['market_caps'].get(name, 0)
            market_cap_billions = market_cap / 1e9
            st.write(f"{i}. {name} ({ticker}) - £{market_cap_billions:.1f}B market cap")
        st.markdown("---")
    
    ftse_cols = st.columns(len(TOP_FTSE100_COMPANIES))
    
    for i, (name, ticker) in enumerate(TOP_FTSE100_COMPANIES.items()):
        with ftse_cols[i]:
            current_price = get_current_price(ticker)
            if current_price:
                # Get yesterday's close for comparison
                hist_data = get_data(ticker, period="2d")
                if not hist_data.empty and len(hist_data) > 1:
                    yesterday_close = float(hist_data['Close'].iloc[-2])
                    change = float(current_price - yesterday_close)
                    change_percent = float((change / yesterday_close) * 100)
                    
                    st.metric(
                        label=name,
                        value=f"{current_price:,.2f}",
                        delta=f"{change:,.2f} ({change_percent:+.2f}%)"
                    )
                else:
                    st.metric(label=name, value=f"{current_price:,.2f}", delta="N/A")
            else:
                st.metric(label=name, value="N/A", delta="N/A")
    
    st.info("💡 Market data is provided by Yahoo Finance and may be delayed. Data updates every 15 minutes.")

elif page_selection == "Weekly Analysis":
    st.header("📈 Weekly Trading Analysis")
    
    # Ticker selection
    selected_name = st.selectbox(
        "Select Financial Instrument:",
        list(ALL_TICKERS.keys()),
        index=0
    )
    selected_ticker = ALL_TICKERS[selected_name]
    
    # Time period selection
    period = st.radio(
        "Analysis Period:",
        ("6mo", "1y", "2y", "5y"),
        index=1,
        horizontal=True
    )
    
    # Fetch and analyze data
    df = get_data(selected_ticker, period=period)
    
    if not df.empty:
        weekly_df = calculate_weekly_indicators(df)
        signal, confidence, signal_details = generate_weekly_signals(weekly_df)
        
        # Display weekly signal
        st.subheader(f"📊 Weekly Signal for {selected_name}")
        
        signal_class = f"signal-{signal.lower()}"
        confidence_class = f"confidence-{confidence.lower()}"
        
        signal_html = f"""
        <div class="{signal_class}">
            <h3>Signal: {signal}</h3>
            <p>Confidence: <span class="{confidence_class}">{confidence}</span></p>
        </div>
        """
        st.markdown(signal_html, unsafe_allow_html=True)
        
        # 7-day forecast
        forecast, upper_range, lower_range = calculate_7day_forecast(weekly_df)
        
        if forecast:
            st.subheader("🔮 7-Day Price Forecast")
            # Use the most recent weekly close price for consistency
            current_price = float(weekly_df['Close'].iloc[-1])
            forecast_change = ((forecast - current_price) / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{current_price:.2f}")
            with col2:
                st.metric("Forecast Price", f"{forecast:.2f}", f"{forecast_change:+.2f}%")
            with col3:
                st.metric("Range", f"{lower_range:.2f} - {upper_range:.2f}")
        
        # Individual Signal Components
        st.subheader("📊 Signal Components Breakdown")
        
        # Create columns for the signal breakdown
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Extract individual signals from debug info (we'll need to modify the function to return these)
        individual_signals = get_individual_signals(weekly_df)
        
        with col1:
            st.markdown("**📈 Forecast**")
            forecast_signal = individual_signals.get('forecast', 'NEUTRAL')
            forecast_color = "🟢" if forecast_signal == "BULLISH" else "🔴" if forecast_signal == "BEARISH" else "🟡"
            st.markdown(f"{forecast_color} {forecast_signal}")
            
        with col2:
            st.markdown("**📊 Moving Average**")
            ma_signal = individual_signals.get('ma', 'NEUTRAL')
            ma_color = "🟢" if ma_signal == "BULLISH" else "🔴" if ma_signal == "BEARISH" else "🟡"
            st.markdown(f"{ma_color} {ma_signal}")
            
        with col3:
            st.markdown("**⚡ RSI**")
            rsi_signal = individual_signals.get('rsi', 'NEUTRAL')
            rsi_color = "🟢" if rsi_signal == "BULLISH" else "🔴" if rsi_signal == "BEARISH" else "🟡"
            st.markdown(f"{rsi_color} {rsi_signal}")
            
        with col4:
            st.markdown("**🔄 MACD**")
            macd_signal = individual_signals.get('macd', 'NEUTRAL')
            macd_color = "🟢" if macd_signal == "BULLISH" else "🔴" if macd_signal == "BEARISH" else "🟡"
            st.markdown(f"{macd_color} {macd_signal}")
            
        with col5:
            st.markdown("**🎯 Bollinger Bands**")
            bb_signal = individual_signals.get('bb', 'NEUTRAL')
            bb_color = "🟢" if bb_signal == "BULLISH" else "🔴" if bb_signal == "BEARISH" else "🟡"
            st.markdown(f"{bb_color} {bb_signal}")
        
        # Summary counts
        st.markdown("---")
        bullish_count = list(individual_signals.values()).count("BULLISH")
        bearish_count = list(individual_signals.values()).count("BEARISH")
        neutral_count = list(individual_signals.values()).count("NEUTRAL")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("🟢 Bullish Signals", bullish_count)
        with summary_col2:
            st.metric("🔴 Bearish Signals", bearish_count)
        with summary_col3:
            st.metric("🟡 Neutral Signals", neutral_count)
        
        # Signal details
        st.subheader("📋 Detailed Signal Analysis")
        for detail in signal_details:
            st.write(f"• {detail}")
        
        # Weekly chart
        st.subheader(f"📊 Weekly Chart - {selected_name}")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Price chart with indicators
        fig.add_trace(
            go.Scatter(
                x=weekly_df.index,
                y=weekly_df['Close'],
                mode='lines+markers',
                name='Weekly Close',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in weekly_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=weekly_df.index,
                    y=weekly_df['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in weekly_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=weekly_df.index,
                    y=weekly_df['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=weekly_df.index,
                y=weekly_df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=weekly_df.index,
                y=weekly_df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='green', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=weekly_df.index,
                y=weekly_df['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Weekly Technical Analysis - {selected_name}",
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Unable to fetch data for the selected instrument.")

elif page_selection == "About":
    st.header("ℹ️ About SoX Trader")
    
    st.markdown("""
    ### What is SoX Trader?
    
    SoX Trader is a weekly trading signal system designed to help traders make informed decisions 
    about buying, selling, or holding financial instruments over a 7-day time horizon.
    
    ### Key Features:
    
    - **Weekly Focus**: Optimized for swing trading with 7-day forecast windows
    - **Signal Generation**: BUY/SELL/HOLD signals with confidence scoring
    - **Technical Analysis**: Weekly Moving Averages, RSI, MACD, and Bollinger Bands
    - **Real-time Data**: Powered by Yahoo Finance via yfinance
    - **Risk Management**: Includes volatility analysis and price range forecasting
    
    ### How to Use:
    
    1. **Market Overview**: Check current market conditions
    2. **Weekly Analysis**: Select an instrument and get detailed weekly signals
    3. **Follow Signals**: Use BUY/SELL/HOLD recommendations with confidence levels
    4. **Monitor Forecasts**: Track 7-day price predictions and ranges
    
    ### Technical Indicators Explained:
    
    - **Moving Averages**: Identify trend direction and momentum
    - **RSI**: Measure overbought/oversold conditions (0-100 scale)
    - **MACD**: Detect changes in momentum and trend strength
    - **Bollinger Bands**: Assess price volatility and potential reversal points
    
    ### Supported Instruments:
    
    **Global Indices:**
    - FTSE 100, NASDAQ, CAC 40, DAX
    
    **Top 5 FTSE 100 Companies:**
    - **Dynamic Selection**: Top 5 companies automatically update weekly based on current market capitalization
    - **Real-time Rankings**: Companies are ranked by actual market cap, not static lists
    - **Automatic Updates**: System refreshes every 7 days to reflect market changes
    - **Manual Refresh**: Users can manually update the top 5 list on the Market Overview page
    """)

# Sidebar disclaimer
st.sidebar.markdown("---")
st.sidebar.header("⚠️ Important Disclaimer")
st.sidebar.warning("""
**RISK WARNING**: Trading involves substantial risk of loss. 

SoX Trader provides **educational analysis only** and is NOT financial advice. 
All trading decisions are your responsibility.

- Past performance does not guarantee future results
- Always use proper risk management
- Consider consulting a financial advisor
- Never risk more than you can afford to lose

**Data Source**: Yahoo Finance (may be delayed)
""")

st.sidebar.markdown("---")
st.sidebar.caption("SoX Trader v1.0 - Weekly Trading Signals")