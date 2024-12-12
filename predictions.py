#!/usr/bin/env python3

# requirements.txt !
import os
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import requests
import streamlit as st
import base64
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

### Configurations
DATA_FILENAME = 'data/data_temp.csv'
MODEL_FILENAME = 'data/model_rf.joblib' # 'data/cnn_model.pth'
SCALER_FILENAME = 'data/scaler_rf.joblib' # 'data/scaler.pkl'
PREDICTIONS_FILENAME = 'data/predictions_random_forest.csv'
GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
NAME_REPO = "Claas99/sprottenflotte_pred_tool"

### Model
# Model hyperparameters
input_size = 24
in_channels = 1
out_channels = 2
kernel_size = 4
stride = 2
dropout_prob = 0.2
prediction_length_steps = 5
activation = torch.nn.ReLU()

original_feature_count = 1 # full_dataset.shape[1]
target_feature_index = 0

class ConvModel(nn.Module):

    def __init__(self, input_size, out_channels, kernel_size, stride, dropout_prob):
        super(ConvModel, self).__init__()
        
        self.input_size = input_size # size of features # sequence length, not feature count
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv1d_output_size=out_channels*math.floor((input_size-kernel_size)/stride +1)
        
        self.hidden_layer_size=int(self.conv1d_output_size/2)
        self.lin = nn.Linear(self.conv1d_output_size, self.hidden_layer_size)  
        self.lin2 = nn.Linear(self.hidden_layer_size, prediction_length_steps)  
        
        
    def forward(self, x):
        x_conv_output = activation(self.conv1d(x))
        x_reshape = x_conv_output.reshape(x_conv_output.size(0), -1)
        x_lin1 = activation(self.lin(x_reshape))
        x_lin1 = self.dropout(x_lin1)
        return self.lin2(x_lin1)
    

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
        "message": f"Update {filepath} file",
        "content": content_base64,
        "sha": sha,
        "branch": branch,
    }

    # Update durchführen
    r = requests.put(url, json=payload, headers=headers)
    if r.status_code == 200:
        log.info(f"----- Prediction file {filepath} updated successfully on GitHub -----")
    else:
        log.error(f"----- Failed to update Prediction file {filepath} on GitHub: {r.content} ------")


def read_csv_from_github(filepath, repo, token, branch="main"):
    url = f'https://api.github.com/repos/{repo}/contents/{filepath}'
    headers = {'Authorization': f'token {token}'}

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        log.error(f"----- Failed to get Prediction file {filepath} from Github: {r.content} ------")
        return None

    file_content = r.json()['content']
    decoded_content = base64.b64decode(file_content).decode('utf-8')
    
    # Convert the decoded content into a pandas DataFrame
    df = pd.read_csv(StringIO(decoded_content))
    return df


def inverse_scale_target(scaler, scaled_target, target_feature_index, original_feature_count):
    # Prepare a dummy matrix with zeros
    dummy = np.zeros((scaled_target.shape[0], original_feature_count))

    # Place scaled target feature where it originally belonged in full dataset
    dummy[:, target_feature_index] = scaled_target.flatten()

    # Use inverse_transform, which applies only to non-zero entries when split like this
    inversed_full = scaler.inverse_transform(dummy)

    # Extract only the inversely transformed target value
    return inversed_full[:, target_feature_index]


# Return the prediction
def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data)


def update_predictions(data_df):
    
    log.info('Prediction process started')

    # make überprüfung, ob predictions needed at this time? otherwise the predictions would be generated every time the application gets refreshed
    try:
        # load in data_temp
        # Laden des existierenden DataFrame
        # data_temp = pd.read_csv(DATA_FILENAME)
        data_temp = data_df.copy()
        data_temp['time_utc'] = pd.to_datetime(data_temp['time_utc'])
        latest_data_time = data_temp['time_utc'].max()
    except Exception as e:
        log.info(f'No {DATA_FILENAME} file found.')
        log.info(f'Error: {e}')
        
    # Prüfen, ob predictions.csv vorhanden ist
    if os.path.exists(PREDICTIONS_FILENAME):
        # # Laden des existierenden DataFrame
        # data_temp_predictions = pd.read_csv(PREDICTIONS_FILENAME)
        ########
    
        data_temp_predictions = read_csv_from_github(PREDICTIONS_FILENAME, NAME_REPO, GITHUB_TOKEN)

        ########

        data_temp_predictions['prediction_time_utc'] = pd.to_datetime(data_temp_predictions['prediction_time_utc'])
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min()
        # überprüfen ob neue predictions necessary
        if earliest_prediction_time > latest_data_time:
            log.info("---------- No new predictions necessary, predictions are up to date.")
            message_type = 'info'
            message_text = 'Es sind bereits Predictions für alle Stationen vorhanden.'
            log.info('Prediction process completed')
            return data_temp_predictions, message_type, message_text # Beenden der Funktion, wenn keine neuen Predictions nötig sind
        else:
            # Altes Daten löschen, da neue Predictions notwendig sind
            data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber'])

    else:
        # Erstellen eines leeren DataFrame, wenn die Datei nicht existiert
        data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber']) # to be adjusted

    try:
            # model saved torch.save(cnn_model.state_dict(), 'cnn_model.pth')
        # load in the model
        # # Modellinitialisierung (Stellen Sie sicher, dass Sie alle benötigten Hyperparameter angeben)
        # loaded_model = ConvModel(input_size, out_channels, kernel_size, stride, dropout_prob)
        # # Laden der Modellparameter
        # loaded_model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))

        # load in the model
        model = joblib.load(MODEL_FILENAME)

    except Exception as e:
        log.info(f'---------- No {MODEL_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    try:
            # scalar saved joblib.dump(scaler, 'scaler.pkl')
        # load in the scalar
        scaler = joblib.load(SCALER_FILENAME)

    except Exception as e:
        log.info(f'---------- No {SCALER_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    # make predictions
    try:
        dataframes = []
        # for every unique entity id make predictions
        entityId_list = data_temp.entityId.unique()
        for entity in entityId_list:
            data_for_prediction = data_temp[data_temp['entityId'] == entity].copy()

            # new ->
            data_for_prediction['Month'] = data_for_prediction['time_utc'].dt.month
            data_for_prediction['Day'] = data_for_prediction['time_utc'].dt.day
            data_for_prediction['Hour'] = data_for_prediction['time_utc'].dt.hour
            data_for_prediction = data_for_prediction[['Month', 'Day', 'Hour', 'availableBikeNumber']]

            # Daten vorverarbeiten (z. B. Skalierung)
            data_for_prediction_scaled = scaler.transform(data_for_prediction)
            # Vorhersagen generieren
            data_for_prediction_scaled_flat = data_for_prediction_scaled.flatten().reshape(1, -1)  # Modell benötigt flache Eingabeform

            predictions_scaled = model.predict(data_for_prediction_scaled_flat)

            # Inverse Transformation zur Originalskala
            preds = predictions_scaled.flatten()

            feature_index = 3  
            num_features = data_for_prediction.shape[1]

            dummy_matrix = np.zeros((preds.shape[0], num_features))
            dummy_matrix[:, feature_index] = preds

            predictions_original_scale = scaler.inverse_transform(dummy_matrix)[:, feature_index]

            # append to dataframe with entityId and predictions
            # Assign dates to each prediction
            start_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
            # Erzeugen einer Liste von Zeitstempeln für jede Vorhersage
            date_list = [start_date + timedelta(hours=i) for i in range(preds.shape[0])]
            
            # Create DataFrame for current entity predictions
            temp_df = pd.DataFrame({
                'entityId': entity,
                'prediction_time_utc': date_list,
                'prediction_availableBikeNumber': predictions_original_scale.tolist()
            })

            # Hinzufügen des temporären DataFrame zur Liste
            dataframes.append(temp_df)
            # <- new

            #############################
            # # Select the 'availableBikeNumber' column, convert to float and create a tensor
            # data_for_prediction_values = data_for_prediction['availableBikeNumber'].values.reshape(-1, 1)

            # # Scale the data
            # data_for_prediction_values_scaled = scaler.transform(data_for_prediction_values)

            # # make the data in such form for model to use
            # # Select the 'availableBikeNumber' column, convert to float and create a tensor
            # data_for_prediction = torch.tensor(data_for_prediction['availableBikeNumber'].values).float()
            # data_for_prediction = data_for_prediction.unsqueeze(0).unsqueeze(0)  # Das Ergebnis ist ebenfalls [1, 1, 24]

            # # make predictions
            # entityId_predictions = predict(loaded_model, data_for_prediction)
            # entityId_predictions = entityId_predictions.unsqueeze(-1)

            # # make predictions real numbers, if model used scaled data for prediction
            # num_samples, prediction_length, _ = entityId_predictions.shape
            # entityId_predictions_reshaped = entityId_predictions.reshape(num_samples * prediction_length, -1)
            # # Inverse transform for target feature predictions
            # entityId_predictions_bikes = inverse_scale_target(scaler, entityId_predictions_reshaped, target_feature_index, original_feature_count).reshape(num_samples, prediction_length, -1)

            # # append to dataframe with entityId and predictions
            # # Assign dates to each prediction
            # start_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
            # # Erzeugen einer Liste von Zeitstempeln für jede Vorhersage
            # date_list = [start_date + timedelta(hours=i) for i in range(prediction_length)]
            
            # # Create DataFrame for current entity predictions
            # temp_df = pd.DataFrame({
            #     'entityId': entity,
            #     'prediction_time_utc': date_list,
            #     'prediction_availableBikeNumber': entityId_predictions_bikes.squeeze().tolist()
            # })

            # # Hinzufügen des temporären DataFrame zur Liste
            # dataframes.append(temp_df)
            #############################

        # Zusammenführen aller temporären DataFrames zu einem finalen DataFrame
        data_temp_predictions = pd.concat(dataframes, ignore_index=True)

        # Update the csv-file in the github repo
        log.info("----- Start updating file on GitHub -----")
        csv_to_github = data_temp_predictions.to_csv(index=False)
        update_csv_on_github(csv_to_github, PREDICTIONS_FILENAME, NAME_REPO, GITHUB_TOKEN)
        
        message_type = 'success'
        message_text = 'Es wurden neue Predictions für alle Stationen gemacht.'
        
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min()

        log.info('---------- Predictions made successfully and saved for all STATION_IDS.')
        log.info(f'---------- Time in UTC:\n          Earliest Prediction for: {earliest_prediction_time}\n          Latest Data for:         {latest_data_time}')
        log.info('Prediction process completed')

        return data_temp_predictions, message_type, message_text

    except Exception as e:
        log.info(f'---------- Error: {e}')
        message_type = 'error'
        message_text = 'Fehler beim machen der Predictions.'
        log.info('Prediction process completed')

        return None, message_type, message_text
