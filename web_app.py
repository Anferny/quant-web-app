import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="QuantSentiment Pro Web")

# NLTK Setup (Handles the first-run download silently)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- SIDEBAR ---
st.sidebar.title("QUANT TERMINAL 🚀")
ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
timeframe = st.sidebar.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

# --- CACHED FUNCTIONS (Speed up the web app) ---
@st.cache_data
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    # Adjust interval based on period to manage data density
    interval = "1d"
    if period in ["1d", "5d"]: interval = "15m"
    history = stock.history(period=period, interval=interval)
    return history, stock.info

@st.cache_data
def get_news_sentiment(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.content, "html.parser")
        table = soup.find(id="news-table")
        if not table: return pd.DataFrame(), 0.0
        
        rows = []
        for tr in table.find_all("tr"):
            a = tr.find("a")
            td = tr.find("td")
            if not a or not td: continue
            link = a["href"]
            title = a.text.strip()
            if title == "Loading...": continue
            rows.append([title, link])
            if len(rows) > 20: break # Limit to recent 20 items
            
        df = pd.DataFrame(rows, columns=["title", "link"])
        vader = SentimentIntensityAnalyzer()
        df["score"] = df["title"].apply(lambda t: vader.polarity_scores(t)["compound"])
        avg_score = df["score"].mean()
        return df, avg_score
    except:
        return pd.DataFrame(), 0.0

# --- MAIN APP LOGIC ---

if ticker:
    # 1. Fetch Data
    with st.spinner(f"Analyzing {ticker}..."):
        history, info = get_stock_data(ticker, timeframe)
        news_df, sentiment_score = get_news_sentiment(ticker)
    
    if history.empty:
        st.error(f"No data found for {ticker}. Check ticker symbol.")
        st.stop()

    # 2. Calculate Indicators (Manual Math to match your Exe)
    history['SMA20'] = history['Close'].rolling(20).mean()
    history['STD20'] = history['Close'].rolling(20).std()
    history['Upper'] = history['SMA20'] + (history['STD20'] * 2)
    history['Lower'] = history['SMA20'] - (history['STD20'] * 2)
    
    # RSI Calculation
    delta = history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    history['RSI'] = 100 - (100 / (1 + rs))

    # Squeeze Detection
    bandwidth = ((history['Upper'] - history['Lower']) / history['SMA20'])
    threshold = bandwidth.rolling(60).quantile(0.10)
    # Check if last candle is in a squeeze
    squeeze_on = False
    if not bandwidth.empty and not pd.isna(threshold.iloc[-1]):
        squeeze_on = bandwidth.iloc[-1] < threshold.iloc[-1]

    # 3. DASHBOARD ROW
    current_price = history['Close'].iloc[-1]
    prev_price = history['Close'].iloc[-2]
    change = current_price - prev_price
    pct_change = (change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price", f"${current_price:.2f}", f"{pct_change:.2f}%")
    
    with col2:
        rsi_val = history['RSI'].iloc[-1]
        st.metric("RSI", f"{rsi_val:.1f}")
    
    with col3:
        sent_color = "🟢 Bullish" if sentiment_score > 0.15 else "🔴 Bearish" if sentiment_score < -0.15 else "⚪ Neutral"
        st.metric("Sentiment", sent_color, f"{sentiment_score:.3f}")
        
    with col4:
        sq_text = "🔥 ON" if squeeze_on else "Off"
        st.metric("Vol Squeeze", sq_text)

    # 4. INTERACTIVE CHART (Plotly)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Candlestick
    fig.add_trace(go.Candlestick(x=history.index,
                                 open=history['Open'], high=history['High'],
                                 low=history['Low'], close=history['Close'],
                                 name='Price'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=history.index, y=history['Upper'], line=dict(color='rgba(0, 255, 0, 0.5)', width=1), name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=history['Lower'], line=dict(color='rgba(255, 0, 0, 0.5)', width=1), name='Lower BB'), row=1, col=1)
    
    # Fill (Cloud) - Creates the shaded area
    fig.add_trace(go.Scatter(x=history.index, y=history['Upper'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=history.index, y=history['Lower'], fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', line=dict(width=0), name='BB Fill'), row=1, col=1)

    # Volume Bar
    colors = ['green' if row['Close'] > row['Open'] else 'red' for index, row in history.iterrows()]
    fig.add_trace(go.Bar(x=history.index, y=history['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Layout Polish
    fig.update_layout(height=700, template="plotly_dark", 
                      xaxis_rangeslider_visible=False, 
                      title=f"{ticker} Daily Analysis",
                      margin=dict(l=10, r=10, t=40, b=10))
    
    st.plotly_chart(fig, use_container_width=True)

    # 5. NEWS SECTION
    st.subheader("Latest News & Sentiment")
    if not news_df.empty:
        # Create a cleaner display for news
        for i, row in news_df.iterrows():
            score = row['score']
            color = "green" if score > 0 else "red" if score < 0 else "gray"
            with st.expander(f"{row['title']}"):
                st.write(f"Sentiment Score: :{color}[{score}]")
                st.write(f"[Read Article]({row['link']})")
    else:
        st.info("No recent news found.")