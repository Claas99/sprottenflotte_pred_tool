#!/usr/bin/env python3

import os
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
import joblib
import base64

import logging


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

### Configurations
DATA_FILENAME = 'data/data_temp.csv'
MODEL_FILENAME = 'data/biLSTM_whole_weights.pth'
SCALER_X_FILENAME = 'data/scaler_X_2.joblib'
SCALER_Y_FILENAME = 'data/scaler_y_2.joblib'
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
        h0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size)#.to(device)
        c0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size)#.to(device)

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

# def inverse_scale_target(scaler, scaled_target, target_feature_index, original_feature_count):
#     # Prepare a dummy matrix with zeros
#     dummy = np.zeros((scaled_target.shape[0], original_feature_count))

#     # Place scaled target feature where it originally belonged in full dataset
#     dummy[:, target_feature_index] = scaled_target.flatten()

#     # Use inverse_transform, which applies only to non-zero entries when split like this
#     inversed_full = scaler.inverse_transform(dummy)

#     # Extract only the inversely transformed target value
#     return inversed_full[:, target_feature_index]

# Return the prediction
def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(data)


def make_dataframe_for_prediction_model(data_df, weather_data_df, stations_df):
    # Filter weather data for a specific station
    specific_weather_data = weather_data_df[weather_data_df['entityId'] == 5423951]

    # Merge data_df with the specific weather data
    combined_df = pd.merge(data_df, specific_weather_data[['time_utc', 'precipitation', 'temperature', 'windSpeed']], on='time_utc', how='left')
    
    # Merge the combined data with stations data to add latitude and longitude
    final_df = pd.merge(combined_df, stations_df[['entityId', 'latitude', 'longitude']], on='entityId', how='left')
    
    # Select and rename columns as needed
    final_df = final_df[['entityId', 'time_utc', 'availableBikeNumber', 'precipitation', 'temperature', 'windSpeed', 'latitude', 'longitude']]
    
    # make sin and cos day and year
    # Extract seconds in day and calculate sine/cosine transformations
    day = 24 * 60 * 60  # Total seconds in a day
    year = 365.2425 * day  # Approximate total seconds in a year

    # Ensure that all datetime objects are tz-naive (idk, had some error to prevent here)
    # final_df['time_utc'] = final_df['time_utc'].dt.tz_localize(None)

    # Timestamp seconds (linear value for point in time)
    final_df['Seconds'] = final_df['time_utc'].map(pd.Timestamp.timestamp)

    # Apply sine and cosine transformations
    final_df['day_sin'] = np.sin(final_df['Seconds'] * (2* np.pi / day))
    final_df['day_cos'] = np.cos(final_df['Seconds'] * (2 * np.pi / day))
    final_df['year_sin'] = np.sin(final_df['Seconds'] * (2 * np.pi / year))
    final_df['year_cos'] = np.cos(final_df['Seconds'] * (2 * np.pi / year))

    # Drop temporary columns
    final_df.drop(columns=['Seconds'], inplace=True)

    return final_df


def update_predictions(data_df, weather_data_df, stations_df):
    
    log.info('Prediction process started')

    # make überprüfung, ob predictions needed at this time? otherwise the predictions would be generated every time the application gets refreshed
    try:
        # load in data_temp
        # Laden des existierenden DataFrame
        # data_temp = pd.read_csv(DATA_FILENAME)

        data_temp = make_dataframe_for_prediction_model(data_df, weather_data_df, stations_df)

        # data_temp = data_df.copy()
        data_temp['time_utc'] = pd.to_datetime(data_temp['time_utc'])
        latest_data_time = data_temp['time_utc'].max()
    except Exception as e:
        log.info(f'No {DATA_FILENAME} file found.')
        log.info(f'Error: {e}')
        
    # Prüfen, ob predictions.csv vorhanden ist
    if os.path.exists(PREDICTIONS_FILENAME):
        # Laden des existierenden DataFrame
        data_temp_predictions = pd.read_csv(PREDICTIONS_FILENAME)
        data_temp_predictions['prediction_time_utc'] = pd.to_datetime(data_temp_predictions['prediction_time_utc'])
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min()

        #### vorerst geskippt, um immer predictions zu machen
        # überprüfen ob neue predictions necessary
        # if earliest_prediction_time > latest_data_time:
        #     log.info("---------- No new predictions necessary, predictions are up to date.")
        #     message_type = 'info'
        #     message_text = 'Es sind bereits Predictions für alle Stationen vorhanden.'
        #     log.info('Prediction process completed')
        #     return data_temp_predictions, message_type, message_text # Beenden der Funktion, wenn keine neuen Predictions nötig sind


        # else:
            # Altes Daten löschen, da neue Predictions notwendig sind
            # data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber'])

    # else:
        # Erstellen eines leeren DataFrame, wenn die Datei nicht existiert
        # data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber']) # to be adjusted

    try:
            # model saved torch.save(cnn_model.state_dict(), 'cnn_model.pth')
        # load in the model
        # Modellinitialisierung (Stellen Sie sicher, dass Sie alle benötigten Hyperparameter angeben)
        loaded_model = BiLSTM(input_size, hidden_size, num_stacked_layers)
        # Laden der Modellparameter
        loaded_model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))

    except Exception as e:
        log.info(f'---------- No {MODEL_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    try:
            # scalar saved joblib.dump(scaler, 'scaler.pkl')
        # load in the scalar
        scaler_X = joblib.load(SCALER_X_FILENAME)

    except Exception as e:
        log.info(f'---------- No {SCALER_X_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    try:
            # scalar saved joblib.dump(scaler, 'scaler.pkl')
        # load in the scalar
        scaler_Y = joblib.load(SCALER_Y_FILENAME)

    except Exception as e:
        log.info(f'---------- No {SCALER_Y_FILENAME} file found.')
        log.info(f'---------- Error: {e}')

    # make predictions
    try:
        dataframes = []
        # for every unique entity id make predictions
        entityId_list = data_temp.entityId.unique()
        for entity in entityId_list:
            data_for_prediction = data_temp[data_temp['entityId'] == entity]

            ######## make input of model, in such form for model to use
            data = data_for_prediction[['availableBikeNumber', 'longitude', 'latitude',
                            'day_sin', 'day_cos', 'year_sin', 'year_cos',
                            'temperature', 'precipitation', 'windSpeed']].to_numpy().astype(np.float32)
            
            data = scaler_X.transform(data)
            data = data.reshape(1, 24, 10)
            
            # Make predictions
            predictions = predict(loaded_model, torch.tensor(data).float())
            
            # # Inverse scale the predictions with scalarY
            predictions = scaler_Y.inverse_transform(predictions.numpy().reshape(-1, 1))

            #.tolist()
            return predictions



            # shape soll 1, 24, 10
            # den scalieren mit scalarX
            # in das modell

            # entityId_predictions = predict(loaded_model, data_for_prediction)

            # predictions mit scalarY zurück scalieren



            #### create final prediction dataframe
            # append to dataframe with entityId and predictions
            # Assign dates to each prediction
            start_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
            # Erzeugen einer Liste von Zeitstempeln für jede Vorhersage
            date_list = [start_date + timedelta(hours=i) for i in range(prediction_length)]
            
            # Create DataFrame for current entity predictions
            temp_df = pd.DataFrame({
                'entityId': entity,
                'prediction_time_utc': date_list,
                'prediction_availableBikeNumber': entityId_predictions_bikes.squeeze().tolist()
            })

            # Hinzufügen des temporären DataFrame zur Liste
            dataframes.append(temp_df)

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