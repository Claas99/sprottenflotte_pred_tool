#!/usr/bin/env python3

import os
import base64
import logging
from datetime import datetime, timedelta, timezone
from io import StringIO

import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from data import update_csv_on_github, read_csv_from_github


# --- Logging ---
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()


# --- Configurations ---
# --- Data/Models ---
DATA_FILENAME = 'data/data_temp.csv'
MODEL_FILENAME = 'models/model_rf.joblib'
SCALER_FILENAME = 'models/scaler_rf.joblib'
PREDICTIONS_FILENAME = 'data/predictions_rf.csv'
# --- Github ---
GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
NAME_REPO = "Claas99/sprottenflotte_pred_tool"
    

# --- Functions ---
# def update_csv_on_github(new_content, filepath, repo, token, branch="main"):
#     url = f'https://api.github.com/repos/{repo}/contents/{filepath}'
#     headers = {'Authorization': f'token {token}'}

#     # Zuerst die alte Dateiinformation laden, um den SHA zu bekommen
#     r = requests.get(url, headers=headers)
#     if r.status_code != 200:
#         log.error(f"Failed to get file info: {r.content}")
#         return
    
#     old_content = r.json()
#     sha = old_content['sha']

#     # Update vorbereiten
#     content_base64 = base64.b64encode(new_content.encode('utf-8')).decode('utf-8')
#     payload = {
#         "message": f"Update {filepath} file",
#         "content": content_base64,
#         "sha": sha,
#         "branch": branch,
#     }

#     # Update durchführen
#     r = requests.put(url, json=payload, headers=headers)
#     if r.status_code == 200:
#         log.info(f"----- Prediction file {filepath} updated successfully on GitHub -----")
#     else:
#         log.error(f"----- Failed to update Prediction file {filepath} on GitHub: {r.content} ------")


# def read_csv_from_github(filepath, repo, token, branch="main"):
#     url = f'https://api.github.com/repos/{repo}/contents/{filepath}'
#     headers = {'Authorization': f'token {token}'}

#     r = requests.get(url, headers=headers)
#     if r.status_code != 200:
#         log.error(f"----- Failed to get Prediction file {filepath} from Github: {r.content} ------")
#         return None

#     file_content = r.json()['content']
#     decoded_content = base64.b64decode(file_content).decode('utf-8')
    
#     # Convert the decoded content into a pandas DataFrame
#     df = pd.read_csv(StringIO(decoded_content))
#     return df


def inverse_scale_target(scaler, scaled_target, target_feature_index, original_feature_count):
    # Prepare a dummy matrix with zeros
    dummy = np.zeros((scaled_target.shape[0], original_feature_count))

    # Place scaled target feature where it originally belonged in full dataset
    dummy[:, target_feature_index] = scaled_target.flatten()

    # Use inverse_transform, which applies only to non-zero entries when split like this
    inversed_full = scaler.inverse_transform(dummy)

    # Extract only the inversely transformed target value
    return inversed_full[:, target_feature_index]


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
        # else:
            # Altes Daten löschen, da neue Predictions notwendig sind
            # data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber'])

    # else:
        # Erstellen eines leeren DataFrame, wenn die Datei nicht existiert
        # data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber']) # to be adjusted

    try:
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
