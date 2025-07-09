import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate SMA
def calculate_sma(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    return sma

# Function to calculate EMA
def calculate_ema(data, window=20):
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    return ema

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    stddev = data['Close'].rolling(window=window).std()
    upper_band = sma + (stddev * 2)
    lower_band = sma - (stddev * 2)
    
    upper_band = upper_band.fillna(method='backfill')
    lower_band = lower_band.fillna(method='backfill')
    
    return sma, upper_band, lower_band

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Function to load all CSV files from a directory
def load_kaggle_dataset(directory):
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
            st.write(f"Loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns.")
    return dataframes

# Function to fetch stock data using yfinance
def fetch_stock_data(symbol):
    data = yf.download(symbol, start='2020-01-01', end='2024-10-01')
    return data

# Function to plot stock data with various indicators
def plot_data(data, rsi=None, macd=None, signal=None, bollinger_bands=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Stock Price
    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.set_title('Stock Price', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Price', fontsize=14)
    ax1.legend()
    ax1.grid(True)

    # RSI
    if rsi is not None:
        ax2.plot(rsi, label='RSI', color='orange')
        ax2.axhline(70, linestyle='--', color='red')
        ax2.axhline(30, linestyle='--', color='green')
        ax2.set_title('Relative Strength Index (RSI)', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('RSI', fontsize=14)
        ax2.legend()
        ax2.grid(True)

    # Bollinger Bands
    if bollinger_bands is not None:
        sma, upper_band, lower_band = bollinger_bands
        ax1.plot(upper_band, label='Upper Bollinger Band', linestyle='--', color='red')
        ax1.plot(lower_band, label='Lower Bollinger Band', linestyle='--', color='green')
        ax1.fill_between(data.index, upper_band, lower_band, color='lightgray', alpha=0.3)

    # MACD
    if macd is not None and signal is not None:
        ax3.plot(macd, label='MACD', color='purple')
        ax3.plot(signal, label='Signal Line', color='brown', linestyle='--')
        ax3.set_title('Moving Average Convergence Divergence (MACD)', fontsize=18, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=14)
        ax3.set_ylabel('MACD', fontsize=14)
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# Function to predict future prices
def predict_future_prices(data, days=5):
    # Prepare the data for linear regression
    data['Days'] = np.arange(len(data))
    X = data[['Days']]
    y = data['Close']

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_days = np.array([[len(data) + i] for i in range(1, days + 1)])
    future_prices = model.predict(future_days)

    return future_prices

# Improved UI/UX Design Section
st.markdown("""
    <style>
    .main-container {
        background-color: #f4f4f8;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .title {
        text-align: center;
        color: #ffd700;
        font-weight: bold;
        font-size: 50px;
    }
    .instruction {
        text-align: center;
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #2980b9;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>🏦 Advanced Stock Analysis Chatbot</div>", unsafe_allow_html=True)

# Sidebar navigation with improved design
st.sidebar.markdown("<div style='text-align:center;font-size:28px;color:#ffd700;font-weight:bold;'>📊 Stock Analysis Menu</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align:center;font-size:20px;color:#ecf0f1;'>Select Stock Symbol and Analysis Options</div>", unsafe_allow_html=True)

# Sidebar for stock symbols and user input
# Updated list with U.S. and Indian stock symbols
stock_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "BRK.B", "NVDA", "JPM", "V", 
    "JNJ", "UNH", "PG", "HD", "DIS", "PYPL", "NFLX", "VZ", "INTC", "CMCSA", "PEP", 
    "CSCO", "ADBE", "NKE", "T", "MRK", "XOM", "IBM", "ABT", "CVX", "CRM", "WMT", 
    "LLY", "BMY", "TXN", "MDT", "BAC", "MCD", "WFC", "LIN", "ABBV", "ORCL", "TMO", 
    "COST", "ACN", "NEE", "AVGO", "QCOM", "KO", "DHR", "TMUS", "RTX", "HON", 
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", 
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "LT.NS", "AXISBANK.NS", 
    "ULTRACEMCO.NS", "HDFC.NS", "WIPRO.NS", "SUNPHARMA.NS", "TITAN.NS", 
    "NESTLEIND.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "TATAMOTORS.NS", 
    "JSWSTEEL.NS", "HCLTECH.NS", "DRREDDY.NS", "POWERGRID.NS", "GRASIM.NS", 
    "ONGC.NS", "COALINDIA.NS"
]


symbol = st.sidebar.selectbox("Choose a Stock Symbol:", stock_symbols)

# User input for specific analysis request
user_query = st.sidebar.text_input("What would you like to know about this stock? (e.g., Show RSI, Show SMA, Predict Future Price)")

# User input for prediction days
days_ahead = st.sidebar.number_input("How many days ahead to predict:", min_value=1, max_value=30, value=5)

# Load Kaggle dataset
kaggle_data_directory = r'C:\stock_analysis_project\data'  # Adjust this path as necessary
kaggle_data = load_kaggle_dataset(kaggle_data_directory)

if st.sidebar.checkbox("Show Kaggle Dataset Preview"):
    for df in kaggle_data:
        st.write(df.head())

# Stock analysis section
if symbol:
    data = fetch_stock_data(symbol)
    data = data.dropna(subset=['Close'])  # Ensure valid data
    
    rsi = calculate_rsi(data)
    sma = calculate_sma(data)
    ema = calculate_ema(data)
    bollinger_bands = calculate_bollinger_bands(data)
    macd, signal = calculate_macd(data)
    
    # User options based on input
    if "rsi" in user_query.lower():
        st.markdown("### RSI Analysis")
        st.line_chart(rsi)
    if "bollinger" in user_query.lower():
        st.markdown("### Bollinger Bands")
        st.line_chart(bollinger_bands)
    
    # Display full chart
    st.markdown("## Full Stock Analysis Chart")
    plot_data(data, rsi=rsi, macd=macd, signal=signal, bollinger_bands=bollinger_bands)
    
    # Prediction
    future_prices = predict_future_prices(data, days=days_ahead)
    st.markdown(f"### Predicted prices for the next {days_ahead} days:")
    st.write(future_prices)
    st.sidebar.subheader("Instructions:")
st.sidebar.write("1. Select a stock symbol from the dropdown menu.")
st.sidebar.write("2. Type your analysis request in the input box. (e.g., 'Show RSI', 'Show SMA', etc.)")
st.sidebar.write("3. Check the box to preview the Kaggle dataset if needed.")
st.sidebar.markdown("---")

# Footer
st.markdown("""
    <div style="text-align: center; font-size: 14px; color: #95a5a6;">
        Created by FANTANSTIC FOURS for Stock Market Insights 🚀
    </div>
""", unsafe_allow_html=True)
