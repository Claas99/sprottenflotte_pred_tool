# Sprottenflotte Prediction Tool

## Overview

The Sprottenflotte Prediction Tool is a Streamlit application designed to predict bike availability using machine learning models. It retrieves live data from relevant APIs and uses predictive models to offer useful insights.

## Structure


```
data
 ├── 
 └──

app_functions.py
app.py
data.py
data_temp.csv
predictions_dl.py
predictions_rf.py

README.md
.gitignore
```


- **app.py**: Main application script to launch the Streamlit app.
- **config.toml**: UI theming and configuration.
- **app_functions.py**: Collection of helper functions for data manipulation and UI functions.
- **data.py**: Manages data fetching from external APIs and synchronization with GitHub.
- **predictions_dl.py**: Manages deep learning model predictions.
- **predictions_rf.py**: Manages random forest model predictions.
- **devcontainer.json**: Sets up a consistent development environment.

## Usage

1. Clone the repository.
2. Set up your environment using the `devcontainer.json` or manually install required dependencies.
3. Run `app.py` using Streamlit to launch the application interface.

## Requirements

- Python 3.10 or above.
- Streamlit, Pandas, NumPy, PyTorch, etc., are specified in `requirements.txt` or `pyproject.toml`.

## Getting Started

1. Clone the repository: `git clone git@github.com:yourusername/sprottenflotte_pred_tool.git`
2. Create a virtual environment and activate it.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the app: `streamlit run app.py`.

## Data Flow

- Data is fetched using API requests in `data.py`.
- Predictions are processed by either `predictions_dl.py` or `predictions_rf.py`.
- UI updates are handled within `app.py` by calling `app_functions.py`.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with your ideas or improvements.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
