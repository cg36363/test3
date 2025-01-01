import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from plotly.graph_objects import Figure, Candlestick
import requests
from textblob import TextBlob

# Fetch news sentiment
def fetch_news_sentiment(ticker):
    """Fetch financial news and calculate sentiment scores."""
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=60a8bc1ae92f46efb9220eeb0f2a47f4"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    sentiments = []
    
    for article in articles:
        content = f"{article.get('title', '')} {article.get('description', '')}"
        sentiment = TextBlob(content).sentiment.polarity
        sentiments.append(sentiment)
    
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    return avg_sentiment

# Calculate technical indicators
def calculate_indicators(df):
    """Calculate standard technical indicators."""
    df['SMA_5'] = SMAIndicator(df['Close'], window=5).sma_indicator()
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_5'] = EMAIndicator(df['Close'], window=5).ema_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    rsi = RSIIndicator(df['Close'])
    df['RSI'] = rsi.rsi()
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
    df['ATR'] = atr.average_true_range()
    return df

# Train ML model
def train_ml_model(df):
    """Train a machine learning model to predict trends."""
    df = df.dropna()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = df[['SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    labels = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Generate predictions
def make_predictions(model, df, sentiment_score):
    """Generate predictions using the trained model and sentiment score."""
    df = df.dropna()
    features = df[['SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 'ATR']]
    df['Prediction'] = model.predict(features)
    df['Prediction_Label'] = df['Prediction'].apply(
        lambda x: "Likely Up" if x == 1 and sentiment_score > 0 else "Not Trending Up"
    )
    return df

# Visualize candlestick charts
def visualize_candlestick(df, ticker):
    """Generate a candlestick chart for the given stock."""
    fig = Figure(data=[Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ticker
    )])
    fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# Streamlit dashboard
def main():
    st.title("Stock Watchlist with Sentiment and Visualizations")
    st.sidebar.header("Options")
    
    # Input tickers
    tickers_input = st.sidebar.text_input("Enter stock tickers (comma-separated)", value="AAPL, TSLA, MSFT")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    # Date range
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    st.write("Fetching stock data...")
    stock_data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

    st.write("Calculating indicators and training model...")
    for ticker, df in stock_data.items():
        stock_data[ticker] = calculate_indicators(df)

    model_ticker = tickers[0]
    model, accuracy = train_ml_model(stock_data[model_ticker])
    st.write(f"Model trained on {model_ticker} with accuracy: {accuracy:.2f}")
    
    results = []
    for ticker, df in stock_data.items():
        sentiment_score = fetch_news_sentiment(ticker)
        st.write(f"Sentiment score for {ticker}: {sentiment_score:.2f}")
        df = make_predictions(model, df, sentiment_score)
        results.append({
            'Ticker': ticker,
            'Latest Close': df['Close'].iloc[-1],
            'Prediction': df['Prediction_Label'].iloc[-1],
            'Sentiment Score': sentiment_score
        })
        st.subheader(f"{ticker}")
        visualize_candlestick(df, ticker)
    
    watchlist_df = pd.DataFrame(results)
    st.write("Generated watchlist with predictions:")
    st.write(watchlist_df)
    
    if st.sidebar.button("Download CSV"):
        watchlist_df.to_csv("stock_watchlist_with_predictions.csv", index=False)
        st.sidebar.success("CSV saved as stock_watchlist_with_predictions.csv")

if __name__ == "__main__":
    main()
