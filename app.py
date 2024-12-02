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
def print_message(message_type, message_text):
    if message_type == 'info':
        return st.info(message_text)
    elif message_type == 'success':
        return st.success(message_text)
    elif message_type == 'error':
        return st.error(message_text)


@st.cache_data
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


def get_full_df_per_station(stations_df, predictions_df, subarea_df):
    # Concatenate die letzten 24h und die nächsten 5h zu einem DataFrame
    stations_df['time_utc'] = pd.to_datetime(stations_df['time_utc'])
    predictions_df['time_utc'] = pd.to_datetime(predictions_df['prediction_time_utc'])
    predictions_df['availableBikeNumber'] = predictions_df['prediction_availableBikeNumber']

    full_df = pd.concat([stations_df[['entityId','time_utc','availableBikeNumber']], predictions_df[['entityId','time_utc','availableBikeNumber']]], ignore_index=True)
    full_df = full_df.sort_values(by=['entityId','time_utc']).reset_index(drop=True)
    full_df = full_df.merge(subarea_df[['entityId', 'subarea']], on='entityId', how='left')

    return full_df

    

# Berechnet absolute Prio - Muss noch in relative prio umberechnet werden
def measures_prio_of_subarea(stations_df:pd.DataFrame, predictions_df:pd.DataFrame, subareas_df) -> pd.DataFrame:
    full_df = get_full_df_per_station(stations_df, predictions_df, subareas_df)
    result_df = pd.DataFrame(columns=['subarea', 'Station', 'Prio'])

    prio = 0

    stations = full_df['entityId'].unique()
    for station in stations:
        teilbereich = full_df[full_df['entityId'] == station]['subarea'].unique()[0]
        max_capacity = subareas_df[subareas_df['subarea'] == teilbereich]['maximum_capacity'].unique()[0]
        station_data = full_df[full_df['entityId'] == station]
        for pred in station_data['availableBikeNumber']:
            # Berechne die Differenz zwischen 
            if pred >= (0.8 * max_capacity):
                prio += 1
            elif pred <= (0.2 * max_capacity):
                prio += 1
    
        result_df = pd.concat([result_df, pd.DataFrame({'subarea': [teilbereich], 'Station': [station], 'Prio': [prio]})], ignore_index=True)
        prio = 0
    
    result_df = result_df.groupby('subarea')['Prio'].apply(lambda x: x.mean()).reset_index(name='subarea_prio')
    result_df = result_df.sort_values('subarea_prio', ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df


def check_duration_of_leerstand(stationID: int, station_data) -> dict:
    # Get max capacity of the station
    max_capacity = data.get_max_capacity(stationID)
    threshold = max_capacity * 0.2

    # Filter predictions for the specific station
    station_predictions = station_data[station_data['prediction_availableBikeNumber'] != None]

    # Sort by time to ensure correct duration calculation
    station_predictions = station_predictions.sort_values(by='prediction_time_utc')

    # Identify periods where predicted value is below the threshold
    station_predictions['below_threshold'] = station_predictions['predicted_value'] <= threshold

    # Calculate the difference in time between consecutive rows
    station_predictions['time_diff'] = station_predictions['prediction_time_utc'].diff()

    # Identify the start of each new period below the threshold
    station_predictions['new_period'] = (station_predictions['below_threshold'] != station_predictions['below_threshold'].shift()).cumsum()

    # Filter only periods where the condition is met
    below_threshold_periods = station_predictions[station_predictions['below_threshold']]

    # Calculate the duration of each period
    period_durations = below_threshold_periods.groupby('new_period')['time_diff'].sum()

    # Convert durations to hours
    period_durations_in_hours = period_durations.dt.total_seconds() / 3600

    # Check if any period exceeds the specified durations
    results = {
        'exceeds_4h': any(period_durations_in_hours > 4),
        'exceeds_8h': any(period_durations_in_hours > 8),
        'exceeds_24h': any(period_durations_in_hours > 24)
    }

    return results



# --- Main App Logic ---
def main():
    stations_filename = "data/stations.csv"
    stations_df = pd.read_csv(stations_filename)

    data_df, data_message_type, data_message_text = data.update_station_data()
    predictions_df, pred_message_type, pred_message_text = predictions.update_predictions(data_df) # use data_df weil in der function sonst eine veraltete version von den daten eingelesen wird, wichtig bei stundenänderung

    # Hole die aktuellen Kapazitätswerte aus predictions_df
    latest_available_bikes = get_latest_available_bikes(data_df)

    # Füge die Werte in die Spalte 'current_capacity' im stations_df ein
    stations_df['current_capacity'] = stations_df['entityId'].map(latest_available_bikes).round()    

    # Berechne das Delta zu max_capacity
    stations_df['Delta'] = stations_df['current_capacity'] - stations_df['maximum_capacity']

    # berechne prio Bedingungen für die Priorisierung
    conditions = [
        (stations_df['current_capacity'] > 0.9 * stations_df['maximum_capacity']) | 
        (stations_df['current_capacity'] < 0.1 * stations_df['maximum_capacity']),  # Sehr hoch oder sehr niedrig
        (stations_df['current_capacity'] > 0.8 * stations_df['maximum_capacity']) | 
        (stations_df['current_capacity'] < 0.2 * stations_df['maximum_capacity'])   # Hoch oder niedrig
    ]
    # Wahl der Prioritäten entsprechend den Bedingungen
    choices = ['❗️❗️', '❗️']
    # Zuweisung der Prioritäten
    stations_df['Prio'] = np.select(conditions, choices, default='')


    # Add a new column to color
    stations_df['color_info'] = stations_df.apply(
        lambda row: 'no data' if pd.isna(row['current_capacity'])
                    else 'überfüllt' if row['current_capacity'] >= 0.8 * row['maximum_capacity']
                    else 'zu leer' if row['current_capacity'] <= 0.2 * row['maximum_capacity'] 
                    else 'okay',
        axis=1
    )

    color_map = {
            'überfüllt': 'red',
            'zu leer': 'blue',
            'okay': 'green',
            'no data': 'grey'
        }

    stations_df['color'] = stations_df['color_info'].map(color_map)

    # --- initialise ---
    # Initialise Streamlit Interface
    st.title("Sprottenflotte prediction model 🚲 x 🤖")
    st.write("""Thank you for using the Sprottenflotte prediciton model! This model is still in beta
              - We are happy to hear your feedback.
             Please report any issues to Claas Resow.""")
    
    # initialise tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tabellenansicht", "Kartenansicht", "Historische_Analyse", "Predictions", "Testebene"])

    # --- tab 1 ---
    with tab1:
        st.write("### Vorhersage - Teilgebiete nach Handlungsbedarf")

        prio_df = measures_prio_of_subarea(data_df, predictions_df, stations_df)
        st.dataframe(prio_df, use_container_width=True)

        ss['subareas'] = prio_df['subarea'].tolist()
        ss['subareas'].append('Alle')  # Option hinzufügen

        with st.expander("ℹ️ Mehr Informationen zu der Berechnung der Prio anzeigen"):
            st.write("""Die Prio der Subareas wird wie folgt berechnet: """)

        # st.info('st.info')
        # st.success('st.success')
        # st.error('st.error')
        # st.warning('st.warning')
        # st.button('Show Info', help='helping', icon='ℹ️', disabled=True)
        # st.radio('Show Info', options=[], help='helping for sure')
    
    # --- tab 2 ---
    with tab2:
        with st.expander("ℹ️ Mehr Informationen zur Karte anzeigen"):
            st.write("""
                     Als Default ist hier das Teilgebiet ausgewählt, welches die höchste Prio hat. Die restlichen Teilgebiete sind nach absteigender Prio sortiert.

                     **Die Farben bedeuten:**
                     - **rot** - überfüllt - mehr als 80% der maximalen Kapazität
                     - **blau** - zu leer - weniger als 20% der maximalen Kapazität
                     - **grün** - okay - zwischen 20% und 80% der maximalen Kapazität
                     - **grau** - no data - keine aktuellen Kapazitätsdaten verfügbar
                    """)

        selected_option = st.selectbox("Wähle ein Teilgebiet aus:", ss['subareas'], index=0)

        subarea_df = make_dataframe_of_subarea(selected_option, stations_df)

        # Plot the map
        fig = px.scatter_mapbox(
            subarea_df, 
            lat='latitude', 
            lon='longitude',
            title=f"Teilgebiet: {selected_option}",
            hover_name='station_name',
            hover_data={
                'current_capacity':True,
                'maximum_capacity': True,
                'Delta': True,
                'latitude': False,  # Disable latitude hover
                'longitude': False,  # Disable longitude hover
                'color_info': False,
                'color': True
            },
            color='color_info',  # Use the new column for colors
            color_discrete_map=color_map,
            zoom=10.2,
            height=600,
            labels={
                'color_info': 'Station Info'  # Change title of the legend
            }
        )

        # Set the Mapbox style (requires an internet connection)
        fig.update_layout(mapbox_style="open-street-map")

        # Adjust the hoverlabel color # bgcolor=subarea_df['color'],
        fig.update_traces(marker=dict(size=12),
                        hoverlabel=dict(#font_family='Serif',
                                        font_size=12,
                                        font_color='#31333F',
                                        bgcolor='#FCFEF6',
                                        bordercolor='#9ec044'))

        # Show the map
        st.plotly_chart(fig)

        selected_station = st.selectbox("Wähle eine Station aus:", subarea_df['station_name'])
        station_data = subarea_df[subarea_df['station_name'] == selected_station].iloc[0]

        # Create a Google Maps URL
        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={station_data['latitude']},{station_data['longitude']}"
        st.markdown(f"[Klicken Sie hier, um {selected_station} in Google Maps zu öffnen]({google_maps_url})")

        columns_to_show = ['subarea', 'station_name', 'current_capacity', 'maximum_capacity',  'Delta', 'Prio']
        st.dataframe(subarea_df[columns_to_show])

        st.dataframe(subarea_df)

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

        # st.write('Als Default ist hier das Teilgebiet ausgewählt, dass die höchste Prio hat. Die restlichen Teilgebiete sind nach absteigender Prio sortiert.')
        
        # selected_option = st.selectbox("Wähle ein Teilgebiet aus:", ss['subareas'], index=0)

        # subarea_df = make_dataframe_of_subarea(selected_option, stations_df)

        # # Plot the map
        # fig = px.scatter_mapbox(
        #     subarea_df, 
        #     lat='latitude', 
        #     lon='longitude', 
        #     hover_name='station_name',
        #     hover_data={
        #         'current_capacity':True,
        #         'maximum_capacity': True,
        #         'Delta': True,
        #         'latitude': False,  # Disable latitude hover
        #         'longitude': False,  # Disable longitude hover
        #         'color': False
        #     },
        #     color='color',  # Use the new column for colors
        #     color_discrete_map={
        #             'überfüllt': 'red',
        #             'zu leer': 'blue',
        #             'okay': 'green'
        #         },
        #     zoom=10.2,
        #     height=600,
        #     labels={
        #         'color': 'Station Info'  # Change title of the legend
        #     }
        # )

        # # Set the Mapbox style (requires an internet connection)
        # fig.update_layout(mapbox_style="open-street-map")

        # # Adjust the hoverlabel color # bgcolor=subarea_df['color'],
        # fig.update_traces(marker=dict(size=12), 
        #                 hoverlabel=dict(font=dict(
        #                                     family='Arial', 
        #                                     size=12,
        #                                     color='black'
        #                                 )))

        # # Show the map
        # st.plotly_chart(fig)


        if predictions_df is not None:
            st.dataframe(predictions_df)
            pivot_df = predictions_df.pivot(index='entityId', columns='prediction_time_utc', values='prediction_availableBikeNumber')
            st.dataframe(pivot_df)
        else:
            st.error("Failed to load prediction data.")

    with tab5:
        full_df = get_full_df_per_station(data_df, predictions_df, stations_df)
        st.dataframe(full_df)

    st.button("Reset App", on_click=reset_app)

# --- Entry Point ---
if __name__ == "__main__":
    main()

