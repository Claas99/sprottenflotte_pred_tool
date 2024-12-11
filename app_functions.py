#!/usr/bin/env python3

import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import plotly.express as px
import data as data
import predictions as predictions
import predictions_test as predictions_test
import numpy as np

# --- Helper Functions ---
def print_message(message_type, message_text):
    if message_type and message_text:
        if message_type == 'info':
            return st.info(message_text)
        elif message_type == 'success':
            return st.success(message_text)
        elif message_type == 'error':
            return st.error(message_text)


def make_dataframe_of_subarea(selected_option, stations_df):
    """Creates a DataFrame for the selected subarea based on the 'subarea' column."""
    if selected_option == 'Alle':
        subarea_df = stations_df.copy()
    else:
        subarea_df = stations_df[stations_df['subarea'] == selected_option]

    subarea_df = subarea_df.sort_values(
        ['Prio', 'Delta'],
        ascending=[False, False],  # Sortiere 'Prio' absteigend und 'Delta' in Bezug auf den absoluten Wert
        key=lambda col: (col if col.name != 'Delta' else abs(col))
    ).reset_index(drop=True)
    subarea_df.index += 1  # Setze den Index auf 1 beginnend
    return subarea_df


def make_subareas_dataframe(stations_df):
    """Creates a DataFrame for the subareas, mean delta, and sort"""
    result_df = stations_df.groupby('subarea')['Delta'].apply(lambda x: x.abs().mean()).reset_index(name='mean_absolute_Delta')
    result_df = result_df.sort_values('mean_absolute_Delta', ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df


def get_latest_available_bikes(stations_df):
    # Sortiere die Daten nach prediction_time_utc, um sicherzustellen, dass der letzte Wert genommen wird
    stations_df_sorted = stations_df.sort_values(by='time_utc', ascending=True)

    # Gruppiere nach entityId und nehme den letzten Wert für availableBikeNumber
    latest_available_bikes = stations_df_sorted.groupby('entityId')['availableBikeNumber'].last()

    return latest_available_bikes


def add_current_capacity_to_stations_df(stations_df, data_df, color_map):
    """
    Adds current capacity, delta, priority, and color information to the stations DataFrame.

    Parameters:
    - stations_df: DataFrame containing information about stations.
    - data_df: DataFrame containing the latest data used to compute the current capacity.

    Returns:
    - Updated stations_df with added columns: 'current_capacity', 'Delta', 'Prio', 'color_info', and 'color'.
    """
    # Get the latest capacity values from data_df
    latest_available_bikes = get_latest_available_bikes(data_df)

    # Add the current capacity values to the stations_df
    stations_df['current_capacity'] = stations_df['entityId'].map(latest_available_bikes).round()

    # Calculate the Delta to max_capacity
    stations_df['Delta'] = stations_df['current_capacity'] - stations_df['maximum_capacity']

    # Define conditions for priority calculation
    conditions = [
        (stations_df['current_capacity'] > 0.9 * stations_df['maximum_capacity']) | 
        (stations_df['current_capacity'] < 0.1 * stations_df['maximum_capacity']),  # Very high or very low
        (stations_df['current_capacity'] > 0.8 * stations_df['maximum_capacity']) | 
        (stations_df['current_capacity'] < 0.2 * stations_df['maximum_capacity'])   # High or low
    ]

    # Define priority choices according to conditions
    choices = ['❗️❗️', '❗️']

    # Assign priority to stations
    stations_df['Prio'] = np.select(conditions, choices, default='')

    # Add a new column to indicate color based on station conditions
    stations_df['color_info'] = stations_df.apply(
        lambda row: 'no data' if pd.isna(row['current_capacity'])
                    else 'überfüllt' if row['current_capacity'] >= 0.8 * row['maximum_capacity']
                    else 'zu leer' if row['current_capacity'] <= 0.2 * row['maximum_capacity'] 
                    else 'okay',
        axis=1
    )

    # Map the colors to a new column
    stations_df['color'] = stations_df['color_info'].map(color_map)

    return stations_df


def add_predictions_to_stations_df(stations_df, predictions_df, color_map_predictions):
    """Adds 5 prediction columns to stations_df for each prediction time."""
    
    # Make sure both DataFrames have the necessary columns
    if 'entityId' not in stations_df.columns:
        raise ValueError("stations_df must contain entityId column")
    if 'entityId' not in predictions_df.columns or 'prediction_time_utc' not in predictions_df.columns or 'prediction_availableBikeNumber' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'entityId', 'prediction_time_utc', and 'prediction_availableBikeNumber' columns")
    
    # Pivot the predictions_df so that each prediction time gets its own column
    predictions_pivot = predictions_df.pivot(index='entityId', columns='prediction_time_utc', values='prediction_availableBikeNumber')
    
    # Round the pivoted prediction values to the nearest integer
    predictions_pivot = predictions_pivot.round(0)

    # Reset the column names of the pivoted DataFrame for clarity
    predictions_pivot.columns = [f'prediction_{i+1}h' for i in range(len(predictions_pivot.columns))]
    
    # Merge the predictions with the stations_df
    stations_df = stations_df.merge(predictions_pivot, how='left', on='entityId')

    # Define a function to determine the color based on current and future capacity status
    def determine_color(row):
        current = row['current_capacity']
        future = row['prediction_5h'] # Just example: using the 5th prediction column for simplicity

         # Check for NaN values
        if pd.isna(current) or pd.isna(future):
            return 'no data'  # Return 'no data' which you can map to grey
        
        condition_current_full = current >= 0.8 * row['maximum_capacity']
        condition_current_empty = current <= 0.2 * row['maximum_capacity']
        condition_current_okay = not (condition_current_full or condition_current_empty)
        
        condition_future_full = future >= 0.8 * row['maximum_capacity']
        condition_future_empty = future <= 0.2 * row['maximum_capacity']
        condition_future_okay = not (condition_future_full or condition_future_empty)
        
        if condition_current_empty:
            if condition_future_empty: return 'zu leer - zu leer'
            elif condition_future_okay: return 'zu leer - okay'
            elif condition_future_full: return 'zu leer - überfüllt'
        elif condition_current_full:
            if condition_future_empty: return 'überfüllt - zu leer'
            elif condition_future_okay: return 'überfüllt - okay'
            elif condition_future_full: return 'überfüllt - überfüllt'
        elif condition_current_okay:
            if condition_future_empty: return 'okay - zu leer'
            elif condition_future_okay: return 'okay - okay'
            elif condition_future_full: return 'okay - überfüllt'
        return 'no data'  # default fallback color

    # Apply the determine_color function
    stations_df['color_info_predictions'] = stations_df.apply(determine_color, axis=1)
    
    stations_df['color_predictions'] = stations_df['color_info_predictions'].map(color_map_predictions)
    
    return stations_df


def get_full_df_per_station(stations_df, predictions_df, subarea_df):
    # Concatenate die letzten 24h und die nächsten 5h zu einem DataFrame
    stations_df['time_utc'] = pd.to_datetime(stations_df['time_utc'])
    predictions_df['time_utc'] = pd.to_datetime(predictions_df['prediction_time_utc'])
    predictions_df['availableBikeNumber'] = predictions_df['prediction_availableBikeNumber']

    full_df = pd.concat([stations_df[['entityId','time_utc','availableBikeNumber']], predictions_df[['entityId','time_utc','availableBikeNumber']]], ignore_index=True)
    full_df = full_df.sort_values(by=['entityId','time_utc']).reset_index(drop=True)
    full_df = full_df.merge(subarea_df[['entityId', 'subarea', 'station_name', 'maximum_capacity']], on='entityId', how='left')
    full_df['deutsche_timezone'] = full_df['time_utc'] + pd.Timedelta(hours=1)

    return full_df


# Berechnet absolute Prio - Muss noch in relative prio umberechnet werden
def measures_prio_of_subarea(stations_df:pd.DataFrame, predictions_df:pd.DataFrame, subareas_df) -> pd.DataFrame:
    full_df = get_full_df_per_station(stations_df, predictions_df, subareas_df)
    # result_df = pd.DataFrame(columns=['subarea', 'Station', 'Prio'])
    first_iteration = True  # Flag zur Überwachung der ersten Iteration

    stations = full_df['entityId'].unique()
    for station in stations:
        prio = 0

        teilbereich = full_df[full_df['entityId'] == station]['subarea'].unique()[0]
        max_capacity = subareas_df[subareas_df['subarea'] == teilbereich]['maximum_capacity'].unique()[0]
        station_data = full_df[full_df['entityId'] == station]
        availableBikes = station_data['availableBikeNumber']

        if availableBikes.iloc[-1] >= (0.8 * max_capacity):
            prio += 0.5
        if len(availableBikes) >= 5 and availableBikes.iloc[-5:].mean() >= (0.8 * max_capacity):  # Mean of last 5 values
            prio += 0.5
        if len(availableBikes) >= 9 and availableBikes.iloc[-9:].mean() >= (0.8 * max_capacity):  # Mean of last 9 values
            prio += 0.5
        if len(availableBikes) >= 25 and availableBikes.iloc[-25:].mean() >= (0.8 * max_capacity):  # Mean of last 25 values
            prio += 1
        
        if availableBikes.iloc[-1] <= (0.2 * max_capacity):
            prio += 0.5
        if len(availableBikes) >= 5 and availableBikes.iloc[-5:].mean() <= (0.2 * max_capacity):  # Mean of last 5 values
            prio += 0.5
        if len(availableBikes) >= 9 and availableBikes.iloc[-9:].mean() <= (0.2 * max_capacity):  # Mean of last 9 values
            prio += 0.5
        if len(availableBikes) >= 25 and availableBikes.iloc[-25:].mean() <= (0.2 * max_capacity):  # Mean of last 25 values
            prio += 1
    
        temp_df = pd.DataFrame({'subarea': [teilbereich], 'Station': [station], 'Prio': [prio]})

        # Überprüfen, ob es die erste Iteration ist
        if first_iteration:
            result_df = temp_df  # Setze result_df direkt für die erste Iteration
            first_iteration = False
        else:
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
    
    result_df = result_df.groupby('subarea')['Prio'].apply(lambda x: x.mean()).reset_index(name='subarea_prio')
    result_df = result_df.sort_values('subarea_prio', ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df
