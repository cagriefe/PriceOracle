# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from models.gru_model import BitcoinPricePredictorGRU

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length, 0]  # Predicting the 'close' price
        sequences.append((seq, label))
    return sequences

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)

    data = pd.read_csv('data/scaled_btc_usdt.csv').values
    seq_length = config['seq_length']
    sequences = create_sequences(data, seq_length)
    sequences = np.array(sequences)
    X, y = sequences[:, 0], sequences[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BitcoinPricePredictorGRU(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    num_epochs = config['num_epochs']

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    torch.save(model.state_dict(), 'models/bitcoin_price_predictor_gru.pth')