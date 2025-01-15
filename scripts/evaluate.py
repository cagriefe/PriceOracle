import torch
import torch.nn as nn  # Import torch.nn
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

def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for sequences, labels in test_loader:
            sequences = sequences.float().to(device)
            labels = labels.float().to(device).view(-1, 1)  # Reshape labels to match output shape
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

if __name__ == "__main__":
    with open('scripts/config.json') as f:
        config = json.load(f)

    data = pd.read_csv('data/scaled_btc_usdt.csv').values
    seq_length = config['seq_length']
    sequences = create_sequences(data, seq_length)
    
    # Separate sequences and labels
    X, y = zip(*sequences)
    X, y = np.array(X), np.array(y)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BitcoinPricePredictorGRU(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size']).to(device)
    model.load_state_dict(torch.load('models/bitcoin_price_predictor_gru.pth', map_location=device))
    criterion = nn.MSELoss()

    test_loss = evaluate_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}')