import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="QuantSentiment Web Terminal")

# Initialize Session State for seamless switching
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "NVDA"
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Deep Analysis"

# NLTK Setup
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# --- CUSTOM STYLES ---
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED FUNCTIONS ---

@st.cache_data(ttl=1800)
def get_trending_tickers():
    """Scrapes Finviz for the top 20 most active/trending stocks."""
    fallback_list = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "COIN", "MARA", "PLTR", "SOFI", "SPY", "QQQ", "IWM", "GME", "HOOD", "MSTR", "RIVN"]
    
    url = "https://finviz.com/screener.ashx?v=111&s=ta_mostactive&ft=4"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        req = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(req.content, "html.parser")
        tickers = []
        for link in soup.find_all('a', class_='screener-link-primary'):
            text = link.text.strip()
            if text and len(text) <= 5 and text.isalpha() and text not in tickers:
                tickers.append(text)
        if len(tickers) < 5: return fallback_list
        return tickers[:30]
    except: return fallback_list

@st.cache_data(ttl=300) 
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        interval = "1d"
        if period in ["1d", "5d"]: interval = "15m"
        history = stock.history(period=period, interval=interval)
        return history
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
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
            if len(rows) > 10: break 
            
        df = pd.DataFrame(rows, columns=["title", "link"])
        vader = SentimentIntensityAnalyzer()
        df["score"] = df["title"].apply(lambda t: vader.polarity_scores(t)["compound"])
        avg_score = df["score"].mean()
        return df, avg_score
    except:
        return pd.DataFrame(), 0.0

def calculate_metrics(history):
    if history.empty: return None, False
    
    # Bollinger Bands
    history['SMA20'] = history['Close'].rolling(20).mean()
    history['STD20'] = history['Close'].rolling(20).std()
    history['Upper'] = history['SMA20'] + (history['STD20'] * 2)
    history['Lower'] = history['SMA20'] - (history['STD20'] * 2)
    
    # RSI
    delta = history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    history['RSI'] = 100 - (100 / (1 + rs))

    # Squeeze
    bandwidth = ((history['Upper'] - history['Lower']) / history['SMA20'])
    threshold = bandwidth.rolling(60).quantile(0.10)
    squeeze_on = False
    if not bandwidth.empty and not pd.isna(threshold.iloc[-1]):
        squeeze_on = bandwidth.iloc[-1] < threshold.iloc[-1]
        
    return history, squeeze_on

def scan_single_stock(ticker):
    try:
        hist = get_stock_data(ticker, "6mo")
        if hist.empty: return None
        hist, squeeze = calculate_metrics(hist)
        
        curr = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        pct = ((curr - prev) / prev) * 100
        rsi = hist['RSI'].iloc[-1]
        
        signal = "NEUTRAL"
        if rsi < 30: signal = "OVERSOLD 🟢"
        elif rsi > 70: signal = "OVERBOUGHT 🔴"
        if squeeze: 
            if signal != "NEUTRAL": signal += " + SQUEEZE 🔥"
            else: signal = "SQUEEZE 🔥"
        
        return {
            "Ticker": ticker,
            "Price": curr,
            "Change %": pct,
            "RSI": rsi,
            "Signal": signal
        }
    except: return None

# --- SIDEBAR NAV ---
st.sidebar.title("QUANT TERMINAL 🚀")

# Mode Switcher (Controlled by Session State)
mode = st.sidebar.radio("Select Mode", ["Deep Analysis", "Market Scanner"], key="app_mode")

if mode == "Deep Analysis":
    st.sidebar.divider()
    st.sidebar.subheader("Configuration")
    
    # Text Input controlled by Session State
    ticker = st.sidebar.text_input("Ticker Symbol", key="selected_ticker").upper()
    timeframe = st.sidebar.selectbox("Timeframe", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    
    if ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            history = get_stock_data(ticker, timeframe)
            news_df, sentiment_score = get_news_sentiment(ticker)
        
        if history.empty:
            st.error("Ticker not found.")
            st.stop()
            
        history, squeeze_on = calculate_metrics(history)
        
        # --- DASHBOARD ---
        st.title(f"{ticker} Analysis")
        
        curr = history['Close'].iloc[-1]
        change = curr - history['Close'].iloc[-2]
        pct = (change / history['Close'].iloc[-2]) * 100
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${curr:.2f}", f"{pct:.2f}%")
        c2.metric("RSI", f"{history['RSI'].iloc[-1]:.1f}")
        c3.metric("Sentiment", f"{sentiment_score:.3f}", "Bullish" if sentiment_score > 0 else "Bearish")
        c4.metric("Squeeze", "ON 🔥" if squeeze_on else "OFF")
        
        # --- CHART ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=history.index, open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=history.index, y=history['Upper'], line=dict(color='rgba(0, 255, 0, 0.5)', width=1), name='Upper BB'), row=1, col=1)
        fig.add_trace(go.Scatter(x=history.index, y=history['Lower'], line=dict(color='rgba(255, 0, 0, 0.5)', width=1), name='Lower BB'), row=1, col=1)
        colors = ['green' if row['Close'] > row['Open'] else 'red' for index, row in history.iterrows()]
        fig.add_trace(go.Bar(x=history.index, y=history['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        if not news_df.empty:
            st.subheader("Recent News")
            for _, row in news_df.iterrows():
                st.write(f"**{row['title']}** [Read]({row['link']})")

elif mode == "Market Scanner":
    st.title("Live Dynamic Market Scanner 📡")
    st.info("💡 **Tip:** Click on any row to instantly analyze that ticker.")

    with st.spinner("Fetching today's trending stocks..."):
        trending_tickers = get_trending_tickers()
    
    if st.button("Scan Market 🚀"):
        results = []
        bar = st.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(scan_single_stock, t): t for t in trending_tickers}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data: results.append(data)
                completed += 1
                bar.progress(completed / len(trending_tickers))
        
        bar.empty()
        
        if results:
            df = pd.DataFrame(results).set_index("Ticker")
            
            # --- INTERACTIVE DATAFRAME ---
            # This is where the magic happens. selection_mode="single-row" allows clicking.
            st.write("### Scan Results")
            event = st.dataframe(
                df.style.format({"Price": "${:.2f}", "Change %": "{:+.2f}%", "RSI": "{:.1f}"}),
                use_container_width=True,
                height=600,
                on_select="rerun",  # Triggers a rerun when clicked
                selection_mode="single-row"
            )
            
            # --- CLICK HANDLER ---
            if event.selection.rows:
                # Get the Ticker from the selected row index
                selected_row_index = event.selection.rows[0]
                clicked_ticker = df.index[selected_row_index]
                
                # Update Session State
                st.session_state.selected_ticker = clicked_ticker
                st.session_state.app_mode = "Deep Analysis"
                
                # Force Rerun to switch tabs immediately
                st.rerun()
