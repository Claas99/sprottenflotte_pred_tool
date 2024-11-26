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


def update_and_save_predictions(DATA_FILENAME, MODEL_FILENAME, SCALER_FILENAME, PREDICTIONS_FILENAME):
    # make überprüfung, ob predictions needed at this time? otherwise the predictions would be generated every time the application gets refreshed
    try:
        # load in data_temp
        # Laden des existierenden DataFrame
        data_temp = pd.read_csv(DATA_FILENAME)
        data_temp['time_utc'] = pd.to_datetime(data_temp['time_utc'])
        latest_data_time = data_temp['time_utc'].max()
    except Exception as e:
        print(f'No {DATA_FILENAME} file found.')
        print(f'Error: {e}')
        
    # Prüfen, ob predictions.csv vorhanden ist
    if os.path.exists(PREDICTIONS_FILENAME):
        # Laden des existierenden DataFrame
        data_temp_predictions = pd.read_csv(PREDICTIONS_FILENAME)
        data_temp_predictions['prediction_time_utc'] = pd.to_datetime(data_temp_predictions['prediction_time_utc'])
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min()
        # überprüfen ob neue predictions necessary
        if earliest_prediction_time > latest_data_time:
            print("No new predictions necessary, predictions are up to date.")
            print('-------------')
            print(f'Time in UTC:\nEarliest Prediction for: {earliest_prediction_time}\nLatest Data for: {latest_data_time}')
            return  # Beenden der Funktion, wenn keine neuen Predictions nötig sind
        else:
            # Altes Daten löschen, da neue Predictions notwendig sind
            data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber'])

    else:
        # Erstellen eines leeren DataFrame, wenn die Datei nicht existiert
        data_temp_predictions = pd.DataFrame(columns=['entityId', 'prediction_time_utc', 'prediction_availableBikeNumber']) # to be adjusted

    try:
            # model saved torch.save(cnn_model.state_dict(), 'cnn_model.pth')
        # load in the model
        # Modellinitialisierung (Stellen Sie sicher, dass Sie alle benötigten Hyperparameter angeben)
        loaded_model = ConvModel(input_size, out_channels, kernel_size, stride, dropout_prob)
        # Laden der Modellparameter
        loaded_model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))

    except Exception as e:
        print(f'No {MODEL_FILENAME} file found.')
        print(f'Error: {e}')

    try:
            # scalar saved joblib.dump(scaler, 'scaler.pkl')
        # load in the scalar
        scaler = joblib.load(SCALER_FILENAME)

    except Exception as e:
        print(f'No {SCALER_FILENAME} file found.')
        print(f'Error: {e}')

    # make predictions
    try:
        dataframes = []
        # for every unique entity id make predictions
        entityId_list = data_temp.entityId.unique()
        for entity in entityId_list:
            data_for_prediction = data_temp[data_temp['entityId'] == entity]

            # make the data in such form for model to use
            # Select the 'availableBikeNumber' column, convert to float and create a tensor
            data_for_prediction = torch.tensor(data_for_prediction['availableBikeNumber'].values).float()
            data_for_prediction = data_for_prediction.unsqueeze(0).unsqueeze(0)  # Das Ergebnis ist ebenfalls [1, 1, 24]

            # make predictions
            entityId_predictions = predict(loaded_model, data_for_prediction)
            entityId_predictions = entityId_predictions.unsqueeze(-1)

            # make predictions real numbers, if model used scaled data for prediction
            num_samples, prediction_length, _ = entityId_predictions.shape
            entityId_predictions_reshaped = entityId_predictions.reshape(num_samples * prediction_length, -1)
            # Inverse transform for target feature predictions
            entityId_predictions_bikes = inverse_scale_target(scaler, entityId_predictions_reshaped, target_feature_index, original_feature_count).reshape(num_samples, prediction_length, -1)

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
        # save them in predictions.csv
        data_temp_predictions.to_csv(PREDICTIONS_FILENAME, index=False)
        earliest_prediction_time = data_temp_predictions['prediction_time_utc'].min()

        print(f'Predictions made successfully and saved for STATION_IDS:{entityId_list}')
        print('-------------')
        print(f'Time in UTC:\nEarliest Prediction for: {earliest_prediction_time}\nLatest Data for: {latest_data_time}')


    except Exception as e:
        print(f'Error in function.')
        print(f'Error: {e}')


### Configurations
DATA_FILENAME = 'data_temp.csv'
MODEL_FILENAME = 'cnn_model.pth'
SCALER_FILENAME = 'scaler.pkl'
PREDICTIONS_FILENAME = 'predictions.csv'


### Usage

update_and_save_predictions(DATA_FILENAME, MODEL_FILENAME, SCALER_FILENAME, PREDICTIONS_FILENAME)