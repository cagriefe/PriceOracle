import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from models.gru_model import BitcoinPricePredictorGRU
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length, 0]  # Predicting the 'close' price
        sequences.append((seq, label))
    return sequences

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.float().to(device)
            labels = labels.float().to(device).view(-1, 1)  # Reshape labels to match output shape
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    mse = mean_squared_error(all_labels, all_outputs)
    mae = mean_absolute_error(all_labels, all_outputs)
    return avg_loss, mse, mae

def plot_predictions(labels, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='Actual')
    plt.plot(predictions, 'r.', label='Predicted')  # Use red dots for predictions
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    with open('scripts/config.json') as f:
        config = json.load(f)

    data = pd.read_csv('data/scaled_btc_usdt.csv').values
    seq_length = config['seq_length']
    sequences = create_sequences(data, seq_length)
    
    # Separate sequences and labels
    X, y = zip(*sequences)
    X, y = np.array(X), np.array(y)
    
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_test.shape[2]  # Update input_size based on the number of features
    model = BitcoinPricePredictorGRU(input_size=input_size, hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size']).to(device)
    model.load_state_dict(torch.load('models/bitcoin_price_predictor_gru.pth'))
    criterion = nn.MSELoss()

    avg_loss, mse, mae = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {avg_loss:.4f}, Test MSE: {mse:.4f}, Test MAE: {mae:.4f}')

    # Plot predictions
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.float().to(device)
            labels = labels.float().to(device).view(-1, 1)
            outputs = model(sequences)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    plot_predictions(all_labels, all_outputs)