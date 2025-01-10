# Sprottenflotte Prediction Tool

- Current Version available with this link: [https://sprottenflottev102.streamlit.app/](https://sprottenflottev102.streamlit.app/)

## Overview

The Sprottenflotte Prediction Tool is an advanced application built using Streamlit that predicts the availability of bicycles using machine learning techniques. By integrating live data from various APIs, the tool provides real-time predictions and insights. It leverages different predictive models, including deep learning models (e.g., BiLSTM) and ensemble techniques like Random Forest. The primary aim is to enhance service efficiency by anticipating bicycle availability across different stations, aiding in better resource allocations.

## Structure

```
.streamlit
 └── config.toml		- UI theming and configuration
.devcontainer
 └── devcontainer.json		- Sets up a consistent development environment
data
 ├── bike_to_weather_station.csv
 ├── data_temp.csv			- Stores temporary station data fetched from the API
 ├── predictions_dl.csv			- Contains prediction data from the deep learning model
 ├── predictions_rf.csv			- Contains prediction data from the random forest model
 ├── stations.csv			- Contains metadata about each bike station, such as location and capacity
 ├── weather_data_temp.csv		- Stores weather data related to the stations
 └── weather_stations.csv		- Contains metadata about each weather station, such as location
models
 ├── 5pred_biLSTM_whole_weights.pth	- Weights for the Bidirectional LSTM model used in predictions
 ├── model_rf.joblib			- Serialized Random Forest model for making predictions
 ├── scaler_rf.joblib			- Scaler for Random Forest data used in predictions
 ├── scaler_X.joblib			- Scaler for LSTM feature data used in predictions
 └── scaler_y.joblib			- Scaler for LSTM target data used in predictions

app_functions.py		- Collection of helper functions for data manipulation and UI functions
app.py				- Main application script to launch the Streamlit app
data.py				- Manages data fetching from external APIs and synchronization with GitHub
predictions_dl.py		- Manages deep learning model predictions
predictions_rf.py		- Manages random forest model predictions

AUTHORS				- Lists the contributors to the repository
poetry.lock			- Contains the exact versions of all dependencies used in the project to ensure consistency across different environments
pyproject.toml			- Specifies the project metadata and dependencies. Functions as a configuration file for the Poetry dependency manager
README.md
.gitignore
```

## Architecture

The architecture of the Sprottenflotte Prediction Tool is centered around several interconnected Python modules, with the main application script `app.py` orchestrating the overall operation of the Streamlit web application. `app.py` serves as the core of the tool, initializing the Streamlit interface, managing user interactions, and setting up the page configuration. It handles model selection between Random Forest and Deep Learning, loads necessary data, and executes the prediction process. Supporting this, `app_functions.py` contains utility functions essential for data manipulation and interface operations, such as displaying messages, filtering data frames, and organizing station data for visualization.

Data management is handled by `data.py`, which retrieves and synchronizes data from external APIs and updates datasets on GitHub, ensuring that the application always uses the most recent data. Predictions are generated by the `predictions_dl.py` and `predictions_rf.py` modules, which process data through a BiLSTM model and a Random Forest model, respectively. These modules perform necessary data transformations and update the results for integration back into the main application. Together, these components form a cohesive infrastructure that supports real-time predictions and dynamic data visualization, ensuring the application is both scalable and maintainable.

## Functionality

Here are some functionalities of the backend of the Application:

- To add a new station, insert a new line in the stations.csv file with all the column information. The data is automatically requested and integrated into the application
  - Go to the `data/stations.csv` file on github
  - Press `e` or click the pen in the top right corner
  - Add a newline with comma separeted values of:
    - entityId: the ID for the request to the addix server
    - station_name: the name od the station, which will be displayed
    - maximum_capacity: the maximum capacity of the station, which is used for calculating priority
    - latitude: the latitude of the station
    - longitude: the longitude of the station
    - subarea: choose a appropriate subarea for the station, which is used to display the station
  - Click the green button `commit changes` in the top right corner
  - You can add a commit message like `Added Station x` and a description
  - Choose `Commit directly to the main branch`
  - Click the green button `commit changes` in the bottom right corner
- Data is only stored for the last 24 hours and forecasts are only made for the next 5 hours
- Data is only requested once per hour and forecasts are also only created once per hour
- The user can choose between two forecast models

## Subareas

Used Subareas are:

- Eckernförde
- Eckernförde Umland Norden
- Felde
- Kiel Innenstadt
- Kiel Norden
- Kiel Osten
- Kiel Umland
- Kiel Westen
- Owschlag
- Plön
- Preetz
- Rendsburg
- Schönberg

## Requirements

- Python 3.10 or above.
- Poetry is used for requirements settings
- Streamlit, Pandas, NumPy, PyTorch, etc., are specified in `pyproject.toml`.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your ideas or improvements.

- After pushes to this repository, please reboot the Streamlit app to make sure all changes are applied

## License

This project is licensed under ...
