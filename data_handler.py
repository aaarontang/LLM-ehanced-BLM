import yfinance as yf
import pandas as pd
import numpy as np

def download_hsi_constituents():
    return ['1398.HK','1093.HK','1810.HK','9988.HK','2628.HK','0883.HK','0992.HK','0241.HK','3690.HK','2269.HK']

def download_monthly_prices(tickers, start, end):
    print("  Downloading data (with buffer)...")
    start_buf = pd.to_datetime(start) - pd.DateOffset(months=2)
    data = yf.download(tickers, start=start_buf, end=end, interval='1mo', progress=False, auto_adjust=True)['Close']
    return data

def calculate_monthly_returns(prices):
    return prices.pct_change().dropna()

def get_current_market_caps(tickers):
    caps = {}
    for t in tickers:
        try: caps[t] = yf.Ticker(t).info.get('marketCap', 1e9)
        except: caps[t] = 1e9
    return caps

def estimate_historical_caps(tickers, cur_caps, p_date, p_cur):
    h_caps = []
    for t in tickers:
        try: h_caps.append(cur_caps.get(t,1e9) * (p_date[t]/p_cur[t]))
        except: h_caps.append(1e9)
    return np.array(h_caps)

def prepare_lookback_data(prices, date, months=3):
    date = pd.to_datetime(date)
    return prices.loc[(prices.index >= date - pd.DateOffset(months=months)) & (prices.index < date)]