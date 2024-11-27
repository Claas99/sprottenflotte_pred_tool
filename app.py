# Here will be the code for the frontend


from enum import Enum, auto

import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import plotly.express as px
import data as data
import plotly.graph_objects as go
import numpy as np


# --- Streamlit Configuration ---
st.set_page_config(page_title="Sprottenflotte prediction model", page_icon="🚲", layout="wide")

def reset_app():
    """Resets the app."""

    # clear session state
    ss.clear()
    initialize_session_state()

    # clear cache
    st.cache_data.clear()
    st.cache_resource.clear()


# --- App Stages ---
class AppStage(Enum):
    """Enum for the different stages of the app."""

    START = auto()
    ANALYSIS_START = auto()

    # define greater or equal
    def __ge__(self, other):
        return self.value >= other.value


# --- Session State Initialization ---
def initialize_session_state():
    """Initializes the session state."""
    if "app_stage" not in ss:
        ss["app_stage"] = AppStage.START
    if "df" not in ss:
        ss["df"] = pd.DataFrame()
    if "analysis_started" not in ss:
        ss["analysis_started"] = False
    if "analysis_done" not in ss:
        ss["analysis_done"] = False
    if "base_rule_expr" not in ss:
        ss["base_rule_expr"] = ""
    if "generated_rules" not in ss:
        ss["generated_rules"] = False
    if "edit_table_id" not in ss:
        ss["edit_table_id"] = 0
    if "download_enabled" not in ss:
        ss["download_enabled"] = False
    if "show_visuals" not in ss:
        ss["show_visuals"] = False  # Default to False (using full dataset)
    if "stations" not in ss:
        ss["stations"] = list()  # Default to False (using full dataset)
    if "subareas" not in ss:
            ss["subareas"] = list()  # Default to False (using full dataset)

def setup():
    if "initialized" not in ss:
        initialize_session_state()
        ss["initialized"] = True

# initialize_session_state()


# --- Helper Functions ---
def get_table_height(rows, max_rows=20, offset=18):
    """Calculates the height of a table."""

    def calc(rows, offset):
        return int(35 + offset + rows * 35)

    return calc(rows, offset) if rows <= max_rows else calc(max_rows, offset)


def increment_edit_table_id():
    """Increments the edit table ID."""
    ss["edit_table_id"] = ss["edit_table_id"] + 1


# --- Main App Logic ---
def main():
    setup() 

    file_station_name = "data/stations.csv"
    hist_bike_data_name = "data/data_temp.csv"

    st.title("Sprottenflotte prediction model 🚲 x 🤖")
    st.write(
        "Thank you for using the Sprottenflotte prediciton model! This model is still in beta - We are happy to hear your feedback. Please report any issues to Claas Resow."
    )
    tab1, tab2, tab3 = st.tabs(["Tabellenansicht", "Kartenansicht", "Historische_Analyse"])



    with tab1:

        # Load data into a DataFrame
        # df = data.get_predictions()
        stations = pd.read_csv(file_station_name)
        st.dataframe(stations)

        #ss['stations'] = list(np.unique(df['Station']))
        ss['subareas'] = list(np.unique(stations['subarea']))
        # Show data preview
        st.write("### Vorhersage")
        #st.dataframe(df.sort_values(['Teilbereich_delta', 'Teilbereich']))
        
    with tab2:
        selected_option = st.selectbox("Wähle ein Teilgebiet aus:", ss["subareas"])

        subarea_df = stations[stations['subarea']==selected_option]

        # Plot the map
        fig = px.scatter_mapbox(
            subarea_df, 
            lat='latitude', 
            lon='longitude', 
            hover_name='station_name', 
            zoom=10,
            height=600
        )

        # Set the Mapbox style (requires an internet connection)
        fig.update_layout(mapbox_style="open-street-map")

        # Show the map
        st.plotly_chart(fig)

    ''' if "jetzt" in df.columns and "in_einer_Stunde" in df.columns:

            station_df = df[df['Station']==selected_option]

            # jetzt vs in_einer_Stunde Plot
            st.write("### jetzt vs in_einer_Stunde Plot")
            fig_jetzt_vs_in_einer_Stunde = go.Figure()
            fig_jetzt_vs_in_einer_Stunde.add_trace(go.Bar(base=station_df, y=station_df['jetzt'], name='jetzt'))
            fig_jetzt_vs_in_einer_Stunde.add_trace(go.Bar(base=station_df, y=station_df['in_einer_Stunde'], name='in_einer_Stunde'))
            fig_jetzt_vs_in_einer_Stunde.update_layout(
                title=f"jetzt vs in einer Stunde für Station {selected_option}",
                xaxis_title="Index",
                yaxis_title="Values",
                legend_title="Legend"
            )
            st.plotly_chart(fig_jetzt_vs_in_einer_Stunde)

            # Error Distribution Plot
            st.write("### Error Distribution Plot")
            fig_error_dist = px.histogram(station_df, x="delta", nbins=10, title="Error Distribution")
            fig_error_dist.update_layout(
                xaxis_title="Error",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_error_dist)

        else:
            st.error("The uploaded file must contain 'jetzt' and 'in_einer_Stunde' columns.")
    '''

    with tab3:
        st.write("### Historische Analyse")
        # data.update_and_save_station_data(data.DATA_FILENAME, data.STATIONS_FILENAME, data.START_DATE, data.END_DATE, data.BASE_URL, data.ACCESS_TOKEN)
        # hist_df = pd.read_csv(hist_bike_data_name)
        # st.dataframe(hist_df)

        hist_df = data.update_station_data()
        if hist_df is not None:
            st.dataframe(hist_df)
        else:
            st.error("Failed to load historical data.")

    st.button("Reset App", on_click=reset_app)

# --- Entry Point ---
if __name__ == "__main__":
    main()

