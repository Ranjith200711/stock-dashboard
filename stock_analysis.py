# 📊 Stock Market Technical Analysis Dashboard (Streamlit Version)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import ta
import matplotlib.pyplot as plt


st.set_page_config(page_title="Stock Technical Analysis", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.header("Stock Analysis Settings")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input(
    "End Date", value=pd.to_datetime("2025-01-01"))

# --- Fetch Data ---


@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


df = load_data(symbol, start_date, end_date)

if df.empty:
    st.error("No data found. Please check symbol or date range.")
    st.stop()

# --- Technical Indicators ---
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()
df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['Signal_Line'] = macd.macd_signal()

# --- Buy/Sell Signals ---
df['Signal'] = 0
df['Signal'][50:] = np.where(df['MA50'][50:] > df['MA200'][50:], 1, -1)
df['Position'] = df['Signal'].diff()

# --- Backtest ---
initial_cash = 100000
cash, shares = initial_cash, 0

for i in range(len(df)):
    if df['Position'].iloc[i] == 2:  # Buy
        shares = cash / df['Close'].iloc[i]
        cash = 0
    elif df['Position'].iloc[i] == -2:  # Sell
        cash = shares * df['Close'].iloc[i]
        shares = 0

final_value = cash if shares == 0 else shares * df['Close'].iloc[-1]
strategy_return = (final_value - initial_cash) / initial_cash * 100
buy_hold_return = (df['Close'].iloc[-1] -
                   df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

# --- Display Metrics ---
st.title("📊 Stock Technical Analysis Dashboard")
st.markdown(f"### **{symbol}** from {start_date} to {end_date}")
col1, col2 = st.columns(2)
col1.metric("Strategy Return", f"{strategy_return:.2f}%")
col2.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")

# --- Candlestick Chart ---
buy_signals = np.where(df['Position'] == 2, df['Close'], np.nan)
sell_signals = np.where(df['Position'] == -2, df['Close'], np.nan)

apds = [
    mpf.make_addplot(df['MA50'], color='blue'),
    mpf.make_addplot(df['MA200'], color='red'),
    mpf.make_addplot(df['RSI'], panel=1, color='purple', ylabel='RSI'),
    mpf.make_addplot(df['MACD'], panel=2, color='green', ylabel='MACD'),
    mpf.make_addplot(df['Signal_Line'], panel=2, color='orange'),
    mpf.make_addplot(buy_signals, type='scatter',
                     marker='^', markersize=100, color='green'),
    mpf.make_addplot(sell_signals, type='scatter',
                     marker='v', markersize=100, color='red')
]

fig, axlist = mpf.plot(
    df, type='candle', style='yahoo',
    addplot=apds, volume=True, returnfig=True,
    figsize=(14, 8)
)

st.pyplot(fig)

# --- Data Table ---
with st.expander("Show Raw Data"):
    st.dataframe(df)
