import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from datetime import datetime, timedelta, timezone
import streamlit as st
import requests

import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

### Configurations
DATA_FILENAME = 'data/data_temp.csv'
MODEL_FILENAME = 'data/cnn_model.pth'
SCALER_FILENAME = 'data/scaler.pkl'
PREDICTIONS_FILENAME = 'data/predictions.csv'
GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
NAME_REPO = "Claas99/sprottenflotte_pred_tool"

# Initialize the Bidirectional LSTM model
input_size = 10  # Number of features
hidden_size = 8
num_stacked_layers = 2
learning_rate = 0.001
num_epochs = 10

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # Hauptunterschied: bidirectional=True
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
            bidirectional=True  # Dies macht das LSTM bidirektional
        )

        # Da bidirektional, verdoppelt sich die Ausgabegröße
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialisiere hidden states für beide Richtungen (daher * 2)
        h0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        # Nimm nur den letzten Zeitschritt
        out = self.fc(out[:, -1, :])
        return out



### Functions
def update_csv_on_github(new_content, filepath, repo, token, branch="main"):
    url = f'https://api.github.com/repos/{repo}/contents/{filepath}'
    headers = {'Authorization': f'token {token}'}

    # Zuerst die alte Dateiinformation laden, um den SHA zu bekommen
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        log.error(f"Failed to get file info: {r.content}")
        return
    
    old_content = r.json()
    sha = old_content['sha']

    # Update vorbereiten
    content_base64 = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')
    payload = {
        "message": "Update Prediction CSV file",
        "content": content_base64,
        "sha": sha,
        "branch": branch,
    }

    # Update durchführen
    r = requests.put(url, json=payload, headers=headers)
    if r.status_code == 200:
        log.info("----- Prediction file updated successfully on GitHub -----")
    else:
        log.error(f"----- Failed to update Prediction file on GitHub: {r.content} ------")

def inverse_scale_target(scaler, scaled_target, target_feature_index, original_feature_count):
    # Prepare a dummy matrix with zeros
    dummy = np.zeros((scaled_target.shape[0], original_feature_count))

    # Place scaled target feature where it originally belonged in full dataset
    dummy[:, target_feature_index] = scaled_target.flatten()

    # Use inverse_transform, which applies only to non-zero entries when split like this
    inversed_full = scaler.inverse_transform(dummy)

    # Extract only the inversely transformed target value
    return inversed_full[:, target_feature_index]
