import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
predictions = pd.read_csv('data/predictions.csv')

# Load actual data
actual_data = pd.read_csv('data/scaled_btc_usdt.csv')

# Plot predictions vs actual data
plt.figure(figsize=(12, 6))
plt.plot(actual_data['close'], label='Actual Close Price')
plt.plot(predictions['predicted_close'], label='Predicted Close Price')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.title('Bitcoin Price Prediction')
plt.legend()
plt.show()