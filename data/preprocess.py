import pandas as pd
from sklearn.preprocessing import StandardScaler
import ta

def preprocess_data(file_path):
    data = pd.read_csv(file_path, index_col='timestamp')
    data['close'] = data['close'].astype(float)
    
    # Add technical indicators
    data['SMA'] = ta.trend.sma_indicator(data['close'], window=20)
    data['EMA'] = ta.trend.ema_indicator(data['close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['close'], window=14)
    data['MACD'] = ta.trend.macd(data['close'])
    data['Volume'] = data['volume'].astype(float)  # Add volume as a feature
    
    # Fill missing values
    data.fillna(method='bfill', inplace=True)
    
    # Select relevant columns
    data = data[['close', 'SMA', 'EMA', 'RSI', 'MACD', 'Volume']]
    
    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create lagged features
    df_scaled = pd.DataFrame(scaled_data, columns=['close', 'SMA', 'EMA', 'RSI', 'MACD', 'Volume'])
    for lag in range(1, 6):
        df_scaled[f'close_lag_{lag}'] = df_scaled['close'].shift(lag)
    
    df_scaled.dropna(inplace=True)
    
    return df_scaled.values, scaler

if __name__ == "__main__":
    file_path = 'data/btc_usdt.csv'
    scaled_data, scaler = preprocess_data(file_path)
    pd.DataFrame(scaled_data, columns=['close', 'SMA', 'EMA', 'RSI', 'MACD', 'Volume', 'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_4', 'close_lag_5']).to_csv('data/scaled_btc_usdt.csv', index=False)