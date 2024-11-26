#!/usr/bin/env python3

# requirements.txt !
import os
import re
from datetime import datetime, timedelta, timezone

import dotenv
import pandas as pd
import requests
import streamlit as st


### Functions
def request_access_token(USERNAME_EMAIL, PASSWORD, CLIENT_SECRET):
    """
    Requests an access token using user credentials and client information from an authorization server.

    This function posts user credentials and client details to the specified token URL of an OAuth authentication server,
    and tries to retrieve an access token. If successful, the access token is returned.

    :param USERNAME_EMAIL: The username or email associated with the user account.
    :type USERNAME_EMAIL: str
    :param PASSWORD: The password for the user account.
    :type PASSWORD: str
    :param CLIENT_SECRET: The secret key associated with the client application.
    :type CLIENT_SECRET: str

    :return: The access token if successfully requested, otherwise None.
    :rtype: str or None
    """

    token_url = 'https://accounts.kielregion.addix.io/realms/infoportal/protocol/openid-connect/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'password',
        'username': USERNAME_EMAIL, 
        'password': PASSWORD,
        'client_id': 'quantumleap',
        'client_secret': CLIENT_SECRET
    }

    response = requests.post(token_url, headers=headers, data=data)

    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get('access_token')
        if access_token:
            print("Access Token successfully requested")
            return access_token
        else:
            print("Access token is not available in the response.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def fetch_station_data(station_id, from_date, to_date, BASE_URL, ACCESS_TOKEN):
    """
    Retrieves data for a specified bike hire docking station over a given time period.

    This function connects to a specified URL to access data regarding the availability of bikes. It
    makes an HTTP GET request with authentication and specific query parameters to gather
    aggregated data across the specified dates.

    :param station_id: Unique identifier for the bike hire docking station.
    :type station_id: str
    :param from_date: The start date and time from when the data is to be fetched.
    :type from_date: datetime
    :param to_date: The end date and time until when the data is to be fetched.
    :type to_date: datetime
    :param BASE_URL: The base URL of the API where the bike data is hosted.
    :type BASE_URL: str
    :param ACCESS_TOKEN: The access token for authenticating the API request.
    :type ACCESS_TOKEN: str

    :return: A JSON object containing the response data if successful, None otherwise. The JSON includes
             aggregated available bike numbers per hour. If the request fails, an error message and status
             code are printed.
    :rtype: dict or None
    
    :raises Exception: Raises an error with status code and text if the response is unsuccessful.
    """

    url = f"{BASE_URL}{station_id}"
    headers = {
        'NGSILD-Tenant': 'infoportal',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    params = {
        'type': 'BikeHireDockingStation',
        'fromDate': from_date.isoformat(),
        'toDate': to_date.isoformat(),
        'attrs': 'availableBikeNumber',
        'aggrPeriod': 'hour',
        'aggrMethod': 'avg'
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        print(f'Got a response for station_id: {station_id}')
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def create_dataframe_from_api_data(data):
    """
    Converts data received from an API response into a structured pandas DataFrame.

    This function parses data from a JSON-like dictionary that includes time indices,
    entity identifiers, and various attributes into a DataFrame that can be used for 
    further data analysis or visualization.

    :param data: The data received from an API request, expected to include keys like 
                 'index', 'entityId', and 'attributes' which contain measurement values.
    :type data: dict

    :return: A DataFrame with time indices, an entity id extracted from 'entityId', and columns 
             for each attribute contained within 'attributes'. This DataFrame is restructured to 
             place 'entityId' and 'time_utc' columns first.
    :rtype: pandas.DataFrame

    :raises ValueError: If essential keys such as 'index', 'entityId', or 'attributes' are missing in the data.
    """

    if not all(key in data for key in ['index', 'entityId', 'attributes']):
        raise ValueError("Data missing one of the essential keys: 'index', 'entityId', 'attributes'")
    
    # Dictionary to store attribute values
    # attribute_data = {}

    # Extract the index from the response
    time_index = pd.to_datetime(data['index'])

    # Extract entityId and entityType
    entity_id = data['entityId']

    # Extract the number after "KielRegion" from the entityId
    match = re.search(r'KielRegion:(\d+)', entity_id)
    entity_id_number = match.group(1) if match else ''  # Get the number or set to empty if not found

    # Loop through each attribute dictionary in 'attributes'
    # for attribute in data['attributes']:
    #     attr_name = attribute['attrName']
    #     attribute_data[attr_name] = attribute.get('values', [])

    # Dictionary to accumulate attribute values
    attribute_data = {attr['attrName']: attr.get('values', []) for attr in data['attributes']}
    
    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(attribute_data)
    # Add the entityId number and index values as new columns
    df['entityId'] = entity_id_number
    df['time_utc'] = time_index

    # Reorder the columns to have 'entityId' first, then 'time', followed by the rest
    column_order = ['entityId', 'time_utc'] + [col for col in df.columns if col not in ['entityId', 'time_utc']]
    df = df[column_order]

    return df

@st.cache_data
def update_and_save_station_data(DATA_FILENAME, STATIONS_FILENAME, START_DATE, END_DATE, BASE_URL, ACCESS_TOKEN):
    """
    Updates and saves bike station data by fetching new data for specified stations and dates, then combining it with existing data.

    This function first checks if there exists previous data in a specified CSV file,
    and loads it if available. It then identifies any gaps in the data between START_DATE and END_DATE,
    and makes API requests to fetch missing data for those specific time periods and station IDs.
    The newly fetched data is then combined with the previously existing data, the combined data is sorted and saved back to the CSV file.

    Parameters:
    - DATA_FILENAME (str): The file path for reading and writing data.
    - STATIONS_FILENAME (str): The file path for reading and writing station data.
    - START_DATE (datetime): The starting datetime from which data needs to be fetched.
    - END_DATE (datetime): The ending datetime until which data needs to be fetched.
    - BASE_URL (str): The base URL to which API requests should be made.
    - ACCESS_TOKEN (str): The token used for authenticating API requests.

    Returns:
    - None: This function does not return any value but prints a message to the console about the process completion or any errors encountered.

    No return value is expected. The function logs messages indicating successful or failed data fetching attempts,
    and shows the number of new records fetched and total unique stations updated.
    If no new data is fetched, it notifies that existing data is used.

    Side Effects:
    - Read and write operations on a CSV file.
    - API requests sent to a remote service.
    - Potentially modifies global state if global variables or mutable data types are passed and manipulated.
    """

    # Prüfen, ob data_temp.csv vorhanden ist
    if os.path.exists(DATA_FILENAME):
        # Laden des existierenden DataFrame
        old_data_temp = pd.read_csv(DATA_FILENAME)
        # make 'time_utc' in datetime
        old_data_temp['time_utc'] = pd.to_datetime(old_data_temp['time_utc'])
        # lösche alle daten vor START_DATE
        old_data_temp = old_data_temp[old_data_temp['time_utc'] >= START_DATE]
    else:
        # Erstellen eines leeren DataFrame, wenn die Datei nicht existiert
        old_data_temp = pd.DataFrame(columns=['entityId', 'time_utc'])

    # prüfen ob station_data.csv vorhanden
    try:
        # Laden des existierenden DataFrame
        stations_data = pd.read_csv(STATIONS_FILENAME)
        # make entity id list
        STATION_IDS = stations_data['entityId'].tolist()
    except Exception as e:
        print(f'No {STATIONS_FILENAME} file exists. Please provide such file with these columns:\nentityId, station_name, maximum_capacity, longitude, latitude, subarea')
        print(f'Error: {e}')

    # - timedelta(hours=1), damit der request_start_date nicht gleich END_DATE ist
    # full_date_range = all timestamps (until now) of the timewindow needed for the model for prediction
    full_date_range = pd.date_range(start=START_DATE, end=END_DATE - timedelta(hours=1), freq='h') 
    # Liste von DataFrames
    dataframes = []

    # entfernen der einträge der station ids, die nicht in STATION_IDS ist
    # Maske, die True ist für jede Zeile, deren entityId NICHT in STATION_IDS ist
    mask = ~old_data_temp['entityId'].isin(STATION_IDS)
    # Entfernen dieser Zeilen
    old_data_temp = old_data_temp[~mask]

    for station_id in STATION_IDS:
        # überprüfe für station_id, ob der zeitraum von START_DATE bis END_DATE in old_data_temp vorhanden ist:
        # select one station
        station_data = old_data_temp[old_data_temp['entityId'] == station_id]
        # extract available dates
        available_dates = station_data['time_utc']
        # Ermitteln der fehlenden Daten
        missing_dates = full_date_range[~full_date_range.isin(available_dates)]

        # wenn ja, skip diese station_id
        # wenn nein, mache ein request_start_date

        # Daten nur für fehlende Zeiten anfordern
        if not missing_dates.empty:
            request_start_date = missing_dates[0]
            # und requeste die nicht vorhandenen stunden bis zum END_DATE
            try:
                data = fetch_station_data(station_id, request_start_date, END_DATE, BASE_URL, ACCESS_TOKEN)
                if data:
                    df = create_dataframe_from_api_data(data)
                    # und appende sie an das dataframe
                    dataframes.append(df)
            except Exception as e:
                print(f'There was a problem retrieving the data for station {station_id}.')
                print(f'Error: {e}')

    if dataframes:
        # Alle neuen DataFrames der Stationen zusammenführen
        new_data_temp = pd.concat(dataframes)
        # make the entitiy_id a number 
        new_data_temp['entityId'] = new_data_temp['entityId'].astype('int64')
        # Zusammenführen des alten DataFrames mit dem neuen
        combined_data_temp = pd.concat([old_data_temp, new_data_temp])
        # Sortieren, nach entitiyId und time_utc
        combined_data_temp = combined_data_temp.sort_values(by=['entityId', 'time_utc'])
        # resete index
        combined_data_temp = combined_data_temp.reset_index(drop=True)
        # DataFrame in eine CSV-Datei speichern
        combined_data_temp.to_csv(DATA_FILENAME, index=False)

        # count new records and unique Ids 
        total_new_records = len(new_data_temp)
        unique_stations = new_data_temp['entityId'].nunique()

        print(f'{total_new_records} new records fetched for {unique_stations} stations.')
        print(f'Data successfully loaded and saved for STATION_IDS:{STATION_IDS}')
    else:
        print('No new data to process, data for every station is available. Existing data used.')

    print('-------------')
    print(f'Time in UTC:\nStart Date: {START_DATE}\nEnd Date: {END_DATE}')


### Configurations
# .env
# .env file anpassen, für application mit password, username email for access token
config = dotenv.dotenv_values('.env')

# PASSWORD = config['PASSWORD']
# CLIENT_SECRET = config['CLIENT_SECRET']
# USERNAME_EMAIL = config['USERNAME_EMAIL']

PASSWORD = st.secrets['PASSWORD']
CLIENT_SECRET = st.secrets['CLIENT_SECRET']
USERNAME_EMAIL = st.secrets['USERNAME_EMAIL']

ACCESS_TOKEN = request_access_token(USERNAME_EMAIL, PASSWORD, CLIENT_SECRET)
BASE_URL = "https://apis.kielregion.addix.io/ql/v2/entities/urn:ngsi-ld:BikeHireDockingStation:KielRegion:"

# API mit UTC time steps
# Calculate the end date by rounding down to the closest whole hour in UTC !,
# to make sure to get hourly averages for whole hours with API request
END_DATE = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=1) # timedelta anpassen an model sliding window length (=24 hours)

DATA_FILENAME = 'data/data_temp.csv'
STATIONS_FILENAME = 'data/stations.csv' # station ids from file einlesen, um änderungen zu haben


### Usage
update_and_save_station_data(DATA_FILENAME, STATIONS_FILENAME, START_DATE, END_DATE, BASE_URL, ACCESS_TOKEN)