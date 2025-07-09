import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

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
    
    # Fill NaN values to avoid plotting errors
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

# Function to calculate Stock Return
def calculate_stock_return(data):
    stock_return = data['Close'].pct_change() * 100
    return stock_return

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

# Function to fetch recent news headlines for sentiment analysis using Financial Modeling Prep API
def fetch_news_headlines(symbol):
    api_key = 'q6NfiFWcAxo2BVfQKHuTXXaqjtR1pytr'  # Financial Modeling Prep API key
    url = f'https://financialmodelingprep.com/api/v3/stock/{symbol}/news?apikey={api_key}'  # Corrected URL
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad responses
        news_data = response.json()

        # Debugging: print the raw response
        st.write("API Response:", news_data)

        if 'news' in news_data and isinstance(news_data['news'], list):
            headlines = [article['title'] for article in news_data['news'] if 'title' in article]
            return headlines
        else:
            error_message = "No articles found or unexpected response format."
            st.write(f"Error fetching news: {error_message}")
            return []
    except requests.exceptions.HTTPError as err:
        st.write(f"HTTP error occurred: {err}")  # Handle HTTP errors
        return []
    except Exception as e:
        st.write(f"An error occurred: {e}")  # Handle any other errors
        return []

# Function to perform sentiment analysis
def analyze_sentiment(headlines):
    sentiment_scores = []
    for headline in headlines:
        score = sentiment_analyzer.polarity_scores(headline)
        sentiment_scores.append(score)
    return sentiment_scores

# Function to plot stock data with various indicators
def plot_data(data, rsi=None, macd=None, signal=None, bollinger_bands=None, stock_return=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Stock Price
    ax1.plot(data['Close'], label='Close Price', color='blue')
    ax1.set_title('Stock Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()

    # RSI
    if rsi is not None:
        ax2.plot(rsi, label='RSI', color='orange')
        ax2.axhline(70, linestyle='--', color='red')
        ax2.axhline(30, linestyle='--', color='green')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.legend()

    # Bollinger Bands
    if bollinger_bands is not None:
        sma, upper_band, lower_band = bollinger_bands
        if np.isfinite(upper_band).all() and np.isfinite(lower_band).all():
            ax1.plot(upper_band, label='Upper Bollinger Band', linestyle='--', color='red')
            ax1.plot(lower_band, label='Lower Bollinger Band', linestyle='--', color='green')
            ax1.fill_between(data.index, upper_band, lower_band, color='lightgray', alpha=0.3)

    # MACD
    if macd is not None and signal is not None:
        ax3.plot(macd, label='MACD', color='purple')
        ax3.plot(signal, label='Signal Line', color='brown', linestyle='--')
        ax3.set_title('Moving Average Convergence Divergence (MACD)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('MACD')
        ax3.legend()

    plt.tight_layout()
    st.pyplot(fig)

# Streamlit app layout
st.title("Advanced Stock Analysis Chatbot")

# Predefined list of stock symbols (expanded list)
stock_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "BRK.B", "NVDA", "JPM", "V", 
    "JNJ", "UNH", "PG", "HD", "DIS", "PYPL", "NFLX", "VZ", "INTC", "CMCSA", "PEP", 
    "CSCO", "ADBE", "NKE", "T", "MRK", "XOM", "IBM", "ABT", "CVX", "CRM", "WMT", 
    "LLY", "BMY", "TXN", "MDT", "BAC", "MCD", "WFC", "LIN", "ABBV", "ORCL", "TMO", 
    "COST", "ACN", "NEE", "AVGO", "QCOM", "KO", "DHR", "TMUS", "RTX", "HON", "TXN", 
    "UPS", "SCHW", "GILD", "SPGI", "CVS", "CAT", "AXP", "SBUX", "MS", "BA", "BLK"
]

# User input for stock symbol
symbol = st.selectbox("Select Stock Symbol:", stock_symbols)

# User input for specific analysis request
user_query = st.text_input(
    "What would you like to know about this stock? (e.g., Show RSI, Show price, Show SMA, Show EMA, Show MACD, Show Bollinger Bands, Show Volume, Show Stock Return, Show Sentiment Analysis, or Plot):"
)

# Load Kaggle dataset (adjust the path to your actual data directory)
kaggle_data_directory = r'C:\stock_analysis_project\data'  # Adjust this path as necessary
kaggle_data = load_kaggle_dataset(kaggle_data_directory)

# Example: Display the first few rows of the Kaggle dataset
if st.checkbox("Show Kaggle Dataset Preview"):
    for df in kaggle_data:
        st.write(df.head())

if symbol:
    data = fetch_stock_data(symbol)
    
    # Ensure the stock data has valid values
    data = data.dropna(subset=['Close'])
    
    rsi = calculate_rsi(data)
    sma = calculate_sma(data)
    ema = calculate_ema(data)
    macd, signal = calculate_macd(data)
    bollinger_bands = calculate_bollinger_bands(data)
    stock_return = calculate_stock_return(data)

    # Respond to user query
    if "RSI" in user_query.upper():
        st.write(f"The current RSI for {symbol} is: {rsi.iloc[-1]:.2f}")
        plot_data(data, rsi=rsi)
    elif "SMA" in user_query.upper():
        st.write("Simple Moving Average (SMA):")
        st.line_chart(sma)
    elif "EMA" in user_query.upper():
        st.write("Exponential Moving Average (EMA):")
        st.line_chart(ema)
    elif "MACD" in user_query.upper():
        st.write("Moving Average Convergence Divergence (MACD):")
        st.line_chart(macd)
    elif "BOLLINGER BANDS" in user_query.upper():
        st.write("Bollinger Bands:")
        plot_data(data, bollinger_bands=bollinger_bands)
    elif "VOLUME" in user_query.upper():
        st.write("Volume Data:")
        st.line_chart(data['Volume'])
    elif "STOCK RETURN" in user_query.upper():
        st.write("Stock Return Data:")
        st.line_chart(stock_return)
    elif "SENTIMENT ANALYSIS" in user_query.upper():
        headlines = fetch_news_headlines(symbol)
        sentiment_scores = analyze_sentiment(headlines)
        st.write("Sentiment Scores:")
        for i, score in enumerate(sentiment_scores):
            st.write(f"{headlines[i]}: {score['compound']:.2f}")
    elif "PLOT" in user_query.upper():
        plot_data(data, rsi=rsi, macd=macd, signal=signal, bollinger_bands=bollinger_bands)


