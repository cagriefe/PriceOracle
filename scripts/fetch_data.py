# scripts/fetch_data.py
from binance.client import Client
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def fetch_historical_data(symbol, interval, start_str):
    api_key = os.getenv('YOUR_API_KEY')
    api_secret = os.getenv('YOUR_API_SECRET')
    client = Client(api_key=api_key, api_secret=api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1HOUR
    start_str = '1 Jan 2015'
    data = fetch_historical_data(symbol, interval, start_str)
    data.to_csv('data/btc_usdt.csv')