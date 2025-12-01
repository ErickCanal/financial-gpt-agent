import streamlit as st
st.set_page_config(page_title="💼 Financial GPT Agent", layout="centered")
st.title("Financial Analyst with GPT")
import pandas as pd
import requests
import os
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

password = st.text_input("Enter access password", type="password")

if password != st.secrets.get("APP_PASSWORD"):
    st.warning("Access denied. Incorrect password")
    st.stop()

st.success("Access granted.")

# Initialize ChatGPT (LangChain wrapper)
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
from langchain_openai import ChatOpenAI

import streamlit as st

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

# Cache Alpha Vantage data
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, alpha_key=None):
    alpha_key = alpha_key or os.getenv("ALPHA_VANTAGE_KEY")
    if not alpha_key:
        raise ValueError("Alpha Vantage API key is required.")

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": alpha_key,
        "outputsize": "compact"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        return None, data

    try:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        }, inplace=True)
        return df, None
    except Exception as e:
        return None, {"error": str(e)}

symbol = st.text_input("Enter stock symbol", value="AAPL")
question = st.text_area("Ask a financial question", value="Should I consider investing now?")

if st.button("Analyze"):
    df, err = fetch_stock_data(symbol)
    if df is None:
        st.error("Failed to retrieve stock data.")
        st.write(err)
    else:
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
        st.line_chart(df[["Close", "SMA20", "SMA50"]])

        latest_price = df["Close"].iloc[-1]
        sma20 = df["SMA20"].iloc[-1]
        sma50 = df["SMA50"].iloc[-1]

        prompt = f"{question}\n\nThe current price of {symbol} is {latest_price:.2f}.\nThe 20-day SMA is {sma20:.2f} and the 50-day SMA is {sma50:.2f}. Provide a brief analysis based on this data."
        response = llm.invoke(prompt)
        st.markdown("### GPT-3.5 Financial Insight")
        st.write(response)
