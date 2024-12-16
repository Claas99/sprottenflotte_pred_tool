#!/usr/bin/env python3

import os
import logging
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import streamlit as st

from data import update_csv_on_github


# --- Logging ---
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger()


# --- Configurations ---
# --- Data/Models ---
DATA_FILENAME = 'data/data_temp.csv'
MODEL_FILENAME = 'models/5pred_biLSTM_whole_ver2_weights.pth'
SCALER_X_FILENAME = 'models/scaler_X_ver2.joblib'
SCALER_Y_FILENAME = 'models/scaler_y_ver2.joblib'
PREDICTIONS_FILENAME = 'data/predictions_dl.csv'
# --- Github ---
GITHUB_TOKEN = st.secrets['GITHUB_TOKEN']
NAME_REPO = "Claas99/sprottenflotte_pred_tool"


# --- initialisation of the Bidirectional LSTM model --- 
# Define constants for the BiLSTM network configuration
input_size = 10  # Number of input features in the input tensor
hidden_size = 8  # Dimensionality of the output space for LSTM layers
num_stacked_layers = 2  # Number of LSTM layers to be stacked
learning_rate = 0.001  # Learning rate for the optimizer
num_epochs = 10  # Number of times to iterate over the training data
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # Initialize the LSTM layer.
        # bidirectional=True converts the LSTM model to a bidirectional LSTM, processing data in both forward and backward directions
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_stacked_layers,
            batch_first=True,
            bidirectional=True
        )

        # The output from the bidirectional LSTM will have double the size of the hidden_size
        # because it concatenates the hidden states from both directions.
        # Here we map the concatenated hidden states to 5 output features.
        self.fc = nn.Linear(hidden_size * 2, 5)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden and cell states for LSTM layers.
        # Note: the size needs to consider both number of layers and directions (hence *2 for bidirectional)
        h0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_stacked_layers * 2, batch_size, self.hidden_size)

        # Forward propagate the LSTM.
        # Input x shape: (batch, seq, feature)
        # Output out shape: (batch, seq, num_directions * hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step through a fully connected layer.
        out = self.fc(out[:, -1, :])

        return out


# --- Functions ---
def make_dataframe_for_prediction_model(data_df, weather_data_df, stations_df):
    """
    Prepares a dataframe for a prediction model by merging bike availability data with weather and station data.

    This function first filters weather data for a specific station and merges it with the main data frame (data_df) that 
    includes bike availability. Then, it merges this combined data frame with station geographic data. Moreover, it
    compounds sine and cosine transformations of time data to capture cyclical nature of days and years, providing useful 
    features for time series prediction models.

    Parameters:
    - data_df (pandas.DataFrame): DataFrame containing primary data including 'entityId' and 'time_utc'.
    - weather_data_df (pandas.DataFrame): DataFrame containing weather data including 'entityId' for stations, 
                                          'time_utc', and weather conditions like 'temperature', 'windSpeed', and 'precipitation'.
    - stations_df (pandas.DataFrame): DataFrame containing station data including 'entityId' and geographical data 
                                      ('latitude', 'longitude').

    Returns:
    - pandas.DataFrame: A DataFrame ready for use in machine learning models, containing combined information from the
                        input DataFrames and additional calculated features useful for predictions.

    Note:
    - The function assumes that the 'time_utc' in the `data_df` and `weather_data_df` are aligned and the 'entityId' in 
      `data_df` matches with 'entityId' in `stations_df` for correct data merging.
    """
    # TODO: make it so that it is the neares station
    # Filter weather data for a specific station
    specific_weather_data = weather_data_df[weather_data_df['entityId'] == 5423951]

    # Merge general data with specific weather conditions based on Universal Coordinate Time (UTC)
    combined_df = pd.merge(data_df, specific_weather_data[['time_utc', 'precipitation', 'temperature', 'windSpeed']], on='time_utc', how='left')
    
    # Further merge combined data with station data to add geographical coordinates
    final_df = pd.merge(combined_df, stations_df[['entityId', 'latitude', 'longitude']], on='entityId', how='left')
    
    # Ensure the final DataFrame presents specific columns in an orderly manner
    final_df = final_df[['entityId', 'time_utc', 'availableBikeNumber', 'precipitation', 'temperature', 'windSpeed', 'latitude', 'longitude']]
    
    # Constants representing seconds in a day and a year
    day = 24 * 60 * 60  # Total seconds in a day
    year = 365.2425 * day  # Approximate total seconds in a year

    # Add timestamp representation
    final_df['Seconds'] = final_df['time_utc'].map(pd.Timestamp.timestamp)

    # Apply trigonometric transformations to capture cyclical nature of time
    final_df['day_sin'] = np.sin(final_df['Seconds'] * (2* np.pi / day))
    final_df['day_cos'] = np.cos(final_df['Seconds'] * (2 * np.pi / day))
    final_df['year_sin'] = np.sin(final_df['Seconds'] * (2 * np.pi / year))
    final_df['year_cos'] = np.cos(final_df['Seconds'] * (2 * np.pi / year))

    # Remove the Seconds column as it is no longer needed after transformations
    final_df.drop(columns=['Seconds'], inplace=True)

    return final_df


def predict(model, data):
    """
    Invokes the prediction method on a given model with provided data.

    The function sets the model to evaluation mode (disabling dropout and batch normalization layers 
    during inference) and disables gradient calculations to improve performance and reduce memory usage 
    during inference.

    Parameters:
    - model (torch.nn.Module): The trained PyTorch model to be used for prediction.
    - data (torch.Tensor): The input data as a PyTorch tensor on which the prediction is to be made.

    Returns:
    - torch.Tensor: The output from the model as a PyTorch tensor, representing predictions for the input data.
    """
    model.eval()
    with torch.no_grad():
        return model(data)
    

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
            
            # # # Inverse scale the predictions with scalarY
            predictions = scaler_Y.inverse_transform(predictions.reshape(-1, 5))

            # return predictions.shape[1]

            #### create final prediction dataframe
            # append to dataframe with entityId and predictions
            # Assign dates to each prediction
            start_date = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) 
            # Erzeugen einer Liste von Zeitstempeln für jede Vorhersage
            date_list = [start_date + timedelta(hours=i) for i in range(predictions.shape[1])]
            
            # Create DataFrame for current entity predictions
            temp_df = pd.DataFrame({
                'entityId': entity,
                'prediction_time_utc': date_list,
                'prediction_availableBikeNumber': predictions.flatten()#.tolist()
            })
            log.info(f"{entity}: {predictions}")

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
