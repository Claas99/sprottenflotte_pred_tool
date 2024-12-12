# Sprottenflotte Prediction Tool

## Overview

The Sprottenflotte Prediction Tool is a Streamlit application designed to predict bike availability using machine learning models. It retrieves live data from relevant APIs and uses predictive models to offer useful insights.

## Structure


```

data
 ├── bike_to_weather_station.csv
 ├── data_temp.csv			- Stores temporary station data fetched from the API
 ├── predictions_dl.csv			- Contains prediction data from the deep learning model
 ├── predictions_rf.csv			- Contains prediction data from the random forest model
 ├── stations.csv			- Contains metadata about each bike station, such as location and capacity
 ├── test_data_for_model.csv
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
.streamlit
 └── config.toml		- UI theming and configuration
.devcontainer
 └── devcontainer.json		- Sets up a consistent development environment
```

## Usage

1. Clone the repository.
2. Set up your environment using the `devcontainer.json` or manually install required dependencies.
3. Run `app.py` using Streamlit to launch the application interface.

## Requirements

- Python 3.10 or above.
- Streamlit, Pandas, NumPy, PyTorch, etc., are specified in `pyproject.toml`.

## Data Flow

- Data is fetched using API requests in `data.py`.
- Predictions are processed by either `predictions_dl.py` or `predictions_rf.py`.
- UI updates are handled within `app.py` by calling `app_functions.py`.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your ideas or improvements.

## License

This project is licensed under ...
