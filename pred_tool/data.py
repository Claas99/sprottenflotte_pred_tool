# This will be the file for the backend

import random
import pandas as pd

def get_predictions():
    # Create a dictionary with actual and predicted values
    data = {
        'actual': [random.randint(1, 100) for _ in range(10)],
        'predicted': [random.randint(1, 100) for _ in range(10)]
    }
    # Convert the dictionary to a DataFrame
    return pd.DataFrame(data)


'''# TODO 
def get_sprottenflotte_data():
    
# TODO 
def get_weather_data():

# TODO
def data_to_dataframe() -> pd.DataFrame:

'''


