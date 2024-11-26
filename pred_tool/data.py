# This will be the file for the backend

import random
import pandas as pd
import streamlit as st

global_station_names = ['Wilhelmsplatz', 'Dreiecksplatz', 'Alter Markt', 'Fähranleger Dietrichsdorf', 'Hauptbahnhof', 'CAU', 'Uni Sportstätten', 'Reventlou', 'Wik', 'Holtenau']

@st.cache_data
def get_predictions():
    # Create a dictionary with actual and predicted values
    data = {
        'Teilbereich': [random.randint(1, 4) for _ in range(10)],
        'Station': global_station_names,    
        'jetzt': [random.randint(1, 100) for _ in range(10)],
        'in_einer_Stunde': [random.randint(1, 100) for _ in range(10)],
    }
    df = pd.DataFrame(data).set_index('Teilbereich')
    df['delta'] = df['in_einer_Stunde'] - df['jetzt'] 
    df['Teilbereich_delta'] = df.groupby('Teilbereich')['delta'].transform('mean')
    # Convert the dictionary to a DataFrame
    return df



'''# TODO 
def get_sprottenflotte_data():
    
# TODO 
def get_weather_data():

# TODO
def data_to_dataframe() -> pd.DataFrame:

'''


