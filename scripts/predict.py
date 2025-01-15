import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import json
from models.gru_model import BitcoinPricePredictorGRU

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return sequences

def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences in data_loader:
            sequences = sequences.float().to(device)
            outputs = model(sequences)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions)

if __name__ == "__main__":
    with open('scripts/config.json') as f:
        config = json.load(f)

    data = pd.read_csv('data/scaled_btc_usdt.csv').values
    seq_length = config['seq_length']
    sequences = create_sequences(data, seq_length)
    sequences = np.array(sequences)
    
    data_loader = torch.utils.data.DataLoader(sequences, batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BitcoinPricePredictorGRU(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size']).to(device)
    model.load_state_dict(torch.load('models/bitcoin_price_predictor_gru.pth'))

    predictions = predict(model, data_loader)
    pd.DataFrame(predictions, columns=['predicted_close']).to_csv('data/predictions.csv', index=False)