import streamlit as st

st.title("🎈 My new app")
st.write(!pip install yfinance pandas numpy scikit-learn streamlit plotly
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import streamlit as st

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate SMA
def calculate_sma(data, period=14):
    return data['Close'].rolling(window=period).mean()

# Fetch historical data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['RSI'] = calculate_rsi(data)
    data['SMA'] = calculate_sma(data)
    data.dropna(inplace=True)
    return data

# Train the model and predict future prices
def predict_prices(data):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA']
    X = data[features]
    y = data['Close'].shift(-1)  # Predict next day's closing price
    X = X[:-1]  # Remove the last row (no target for it)
    y = y[:-1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    data['Predicted'] = model.predict(X)
    data['Signal'] = np.where(data['Predicted'] > data['Close'], 'Buy', 'Sell')
    return data

# Plot candlestick chart with signals
def plot_candlestick(data):
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    ))

    # Buy signals
    buy_signals = data[data['Signal'] == 'Buy']
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'],
        mode='markers',
        marker=dict(color='green', size=10),
        name='Buy Signal'
    ))

    # Sell signals
    sell_signals = data[data['Signal'] == 'Sell']
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['Close'],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Sell Signal'
    ))

    fig.update_layout(title='Candlestick Chart with Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price')
    return fig

# Streamlit app
def main():
    st.title("AI Trading Bot")
    st.sidebar.header("User Input")

    # User inputs
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD):", "2020-01-01")
    end_date = st.sidebar.text_input("End Date (YYYY-MM-DD):", "2025-02-02")

    # Fetch data
    data = fetch_data(ticker, start_date, end_date)

    # Predict prices and generate signals
    data = predict_prices(data)

    # Display data
    st.write("### Historical Data with Predictions")
    st.write(data.tail())

    # Plot candlestick chart
    st.write("### Candlestick Chart with Buy/Sell Signals")
    fig = plot_candlestick(data)
    st.plotly_chart(fig)

# Run the app
if __name__ == "__main__":
    main()