# Here will be the code for the frontend


from enum import Enum, auto

import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import plotly.express as px
import data as data
import predictions as predictions
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


initialize_session_state() # needs to be here?


# --- Helper Functions ---
def get_table_height(rows, max_rows=20, offset=18):
    """Calculates the height of a table."""

    def calc(rows, offset):
        return int(35 + offset + rows * 35)

    return calc(rows, offset) if rows <= max_rows else calc(max_rows, offset)


def increment_edit_table_id():
    """Increments the edit table ID."""
    ss["edit_table_id"] = ss["edit_table_id"] + 1

def print_message(message_type, message_text):
    if message_type == 'info':
        return st.info(message_text)
    elif message_type == 'success':
        return st.success(message_text)
    elif message_type == 'error':
        return st.error(message_text)


@st.cache_data
def make_dataframe_of_subarea(selected_option, stations_df):
    """Creates a DataFrame for the selected subarea based on the 'Teilbereich' column."""
    subarea_df = stations_df[stations_df['Teilbereich'] == selected_option]
    subarea_df = subarea_df.sort_values(
        ['Prio', 'Delta'],
        ascending=[False, True],  # Sortiere 'Prio' absteigend und 'Delta' in Bezug auf den absoluten Wert
        key=lambda col: (col if col.name != 'Delta' else abs(col))
    )
    return subarea_df.reset_index(drop=True)

def make_subareas_dataframe(stations_df):
    """Creates a DataFrame for the subareas, mean delta, and sort"""
    result_df = stations_df.groupby('Teilbereich')['Delta'].apply(lambda x: x.abs().mean()).reset_index(name='Mean Absolute Delta')
    return result_df.sort_values('Mean Absolute Delta', ascending=False).reset_index(drop=True)

# Berechnet absolute Prio - Muss noch in relative prio umberechnet werden
def measures_prio_of_subarea(subarea_df:pd.DataFrame) -> int:
    predictions_df = predictions.update_predictions() 

    def measure_überfüllt(stationID:int) -> int:
        # get max capacity of station
        max_capacity = data.get_max_capacity(stationID)
        # return variable
        hours_überfüllt = 0

        for pred_value in predictions_df[predictions_df['entityID']==stationID]:
            if pred_value >= (max_capacity*0.8):
                hours_überfüllt += 1

        # return hours_überfüllt
        return hours_überfüllt
    
    def measure_zu_leer(stationID:int) -> int:
        # get max capacity of station
        max_capacity = data.get_max_capacity(stationID)
        # return variable
        hours_zu_leer = 0

        for pred_value in predictions_df[predictions_df['entityID']==stationID]:
            if pred_value <= (max_capacity*0.2):
                hours_zu_leer += 1

        # return hours_zu_leer
        return hours_zu_leer
    
    result_df = pd.DataFrame(columns=['Teilgebiet', 'Prio'])
    for station in subarea_df['entityId']:
        überfüllt = measure_überfüllt(station)
        zu_leer = measure_zu_leer(station)
        prio = überfüllt + zu_leer
        result_df = pd.concat([result_df, pd.DataFrame({'Teilgebiet': [station], 'Prio': [prio]})], ignore_index=True)
    
    return result_df


# --- Main App Logic ---
def main():
    stations_filename = "data/stations.csv"
    stations_df = pd.read_csv(stations_filename)

    data_df, data_message_type, data_message_text = data.update_station_data()
    predictions_df, pred_message_type, pred_message_text = predictions.update_predictions(data_df) # use data_df weil in der function sonst eine veraltete version von den daten eingelesen wird, wichtig bei stundenänderung

    stations_df['Teilbereich'] = stations_df['subarea'].str.replace('√∂', 'ö')

    # Generiere aktuelle Werte für jede Station
    # add code, to get the latest number for every station instead of random generating numbers
    stations_df['Aktuelle_Kapazität'] = np.random.randint(0, stations_df['maximum_capacity'] + 10, size=stations_df.shape[0])

    # Berechne das Delta zu max_capacity
    stations_df['Delta'] = stations_df['Aktuelle_Kapazität'] - stations_df['maximum_capacity']

    # Bedingungen und Priorität festlegen
    conditions = [
        stations_df['Delta'] >= 7,
        stations_df['Delta'] < -7,
        stations_df['Delta'] < -5,
        stations_df['Delta'] >= 5
    ]
    choices = ['❗️❗️❗️', '❗️❗️❗️', '❗️❗️', '❗️❗️']

    stations_df['Prio'] = np.select(conditions, choices, default='')

    # Add a new column to categorize Teilbereich_delta
    stations_df['Delta_color'] = stations_df['Delta'].apply(
        lambda x: 'überfüllt' if x >= 5 else ('zu leer' if x <= -5 else 'okay')
    )

    # --- initialise ---
    # Initialise Streamlit Interface
    st.title("Sprottenflotte prediction model 🚲 x 🤖")
    st.write("""Thank you for using the Sprottenflotte prediciton model! This model is still in beta
              - We are happy to hear your feedback.
             Please report any issues to Claas Resow.""")
    
    # initialise tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tabellenansicht", "Kartenansicht", "Historische_Analyse", "Predictions"])

    # --- tab 1 ---
    with tab1:
        st.write("### Vorhersage - Teilgebiete nach Handlungsbedarf")

        subareas = make_subareas_dataframe(stations_df)
        ss['subareas'] = subareas['Teilbereich'].tolist()

        st.dataframe(subareas, use_container_width=True)

        st.info('st.info')
        st.success('st.success')
        st.error('st.error')
        st.warning('st.warning')
    
    # --- tab 2 ---
    with tab2:
        st.write('Als Default ist hier das Teilgebiet ausgewählt, dass die höchste Prio hat. Die restlichen Teilgebiete sind nach absteigender Prio sortiert.')
        
        selected_option = st.selectbox("Wähle ein Teilgebiet aus:", ss['subareas'], index=0)

        subarea_df = make_dataframe_of_subarea(selected_option, stations_df)

        # Plot the map
        fig = px.scatter_mapbox(
            subarea_df, 
            lat='latitude', 
            lon='longitude', 
            hover_name='station_name',
            hover_data={
                'Aktuelle_Kapazität':True,
                'maximum_capacity': True,
                'Delta': True,
                'latitude': False,  # Disable latitude hover
                'longitude': False,  # Disable longitude hover
                'Delta_color': False
            },
            color='Delta_color',  # Use the new column for colors
            color_discrete_map={
                    'überfüllt': 'red',
                    'zu leer': 'blue',
                    'okay': 'green'
                },
            zoom=10.2,
            height=600,
            labels={
                'Delta_color': 'Station Info'  # Change title of the legend
            }
        )

        # Set the Mapbox style (requires an internet connection)
        fig.update_layout(mapbox_style="open-street-map")

        # Adjust the hoverlabel color # bgcolor=subarea_df['Delta_color'],
        fig.update_traces(marker=dict(size=12), 
                        hoverlabel=dict(font=dict(
                                            family='Arial', 
                                            size=12,
                                            color='black'
                                        )))

        # Show the map
        st.plotly_chart(fig)

        columns_to_show = ['Teilbereich', 'station_name', 'Aktuelle_Kapazität', 'maximum_capacity',  'Delta', 'Prio']
        st.dataframe(subarea_df[columns_to_show])

        st.dataframe(subarea_df)

        # ''' if "jetzt" in df.columns and "in_einer_Stunde" in df.columns:

        #         station_df = df[df['Station']==selected_option]

        #         # jetzt vs in_einer_Stunde Plot
        #         st.write("### jetzt vs in_einer_Stunde Plot")
        #         fig_jetzt_vs_in_einer_Stunde = go.Figure()
        #         fig_jetzt_vs_in_einer_Stunde.add_trace(go.Bar(base=station_df, y=station_df['jetzt'], name='jetzt'))
        #         fig_jetzt_vs_in_einer_Stunde.add_trace(go.Bar(base=station_df, y=station_df['in_einer_Stunde'], name='in_einer_Stunde'))
        #         fig_jetzt_vs_in_einer_Stunde.update_layout(
        #             title=f"jetzt vs in einer Stunde für Station {selected_option}",
        #             xaxis_title="Index",
        #             yaxis_title="Values",
        #             legend_title="Legend"
        #         )
        #         st.plotly_chart(fig_jetzt_vs_in_einer_Stunde)

        #         # Error Distribution Plot
        #         st.write("### Error Distribution Plot")
        #         fig_error_dist = px.histogram(station_df, x="delta", nbins=10, title="Error Distribution")
        #         fig_error_dist.update_layout(
        #             xaxis_title="Error",
        #             yaxis_title="Frequency"
        #         )
        #         st.plotly_chart(fig_error_dist)

        #     else:
        #         st.error("The uploaded file must contain 'jetzt' and 'in_einer_Stunde' columns.")
        # '''

    # --- tab 3 ---
    with tab3:
        st.write("### Historische Analyse")
        print_message(data_message_type, data_message_text)

        if data_df is not None:
            st.dataframe(data_df)
        else:
            st.error("Failed to load historical data.")

    # --- tab 4 ---
    with tab4:
        st.write("### Predictions")
        print_message(pred_message_type, pred_message_text)

        if predictions_df is not None:
            st.dataframe(predictions_df)
            pivot_df = predictions_df.pivot(index='entityId', columns='prediction_time_utc', values='prediction_availableBikeNumber')
            st.dataframe(pivot_df)
        else:
            st.error("Failed to load prediction data.")


    st.button("Reset App", on_click=reset_app)

# --- Entry Point ---
if __name__ == "__main__":
    main()

